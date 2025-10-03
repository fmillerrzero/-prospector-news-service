#!/usr/bin/env python3
"""
News service v4 — tuned for Manhattan Class A office buildings
---------------------------------------------------------------
- Builds queries from building names & addresses
- (Optional) Adds office-context terms to the query to reduce noise
- Strict recency + confidence scoring tuned for Class A office context
- Extra boosts for office/leasing context and major NYC RE outlets
- Negative weights for non-office (residential/hospitality) if no office context
- Manhattan-aware penalties if other boroughs are mentioned without Manhattan
- JSON API supports min_score, max_age_days, limit, and optional debug reasons

ENV (optional):
  ALLOWED_SITES="therealdeal.com,commercialobserver.com,crainsnewyork.com,bisnow.com,nytimes.com,wsj.com,bloomberg.com,globest.com"
  EXCLUDE_TERMS='("Bank of America Stadium" OR Charlotte OR "North Carolina")'
  CITY_TERMS='("Manhattan" OR "New York" OR "NYC")'
  OFFICE_QUERY_TERMS='("office" OR "lease" OR "leases" OR "leasing" OR "tenant" OR "tenants" OR "office tower")'
  REQUEST_TIMEOUT=10
  CACHE_SECONDS=900
  DEFAULT_MIN_SCORE=7.5
  DEFAULT_MAX_AGE_DAYS=10
  NEWS_REFRESH_TOKEN="shared-secret"   # optional gate for refresh endpoints
  INCLUDE_REASONS=0                    # set 1 to include score reasons in API response
"""
import argparse, hashlib, os, re, time, sqlite3, json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Optional

import pandas as pd
import feedparser
import requests
import requests_cache
from flask import Flask, jsonify, request

# -------------------- Config --------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "900"))
CITY_TERMS = os.getenv("CITY_TERMS", '("Manhattan" OR "New York" OR "NYC")')
EXCLUDE_TERMS = os.getenv("EXCLUDE_TERMS", '("Bank of America Stadium" OR Charlotte OR "North Carolina")')
ALLOWED_SITES = [s.strip().lower() for s in os.getenv("ALLOWED_SITES", "").split(",") if s.strip()]
OFFICE_QUERY_TERMS = os.getenv("OFFICE_QUERY_TERMS", '("office" OR "lease" OR "leases" OR "leasing" OR "tenant" OR "tenants" OR "office tower")')
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "7.5"))
DEFAULT_MAX_AGE_DAYS = int(os.getenv("DEFAULT_MAX_AGE_DAYS", "10"))
REFRESH_TOKEN = os.getenv("NEWS_REFRESH_TOKEN", "").strip()
INCLUDE_REASONS = os.getenv("INCLUDE_REASONS", "0").strip() == "1"

requests_cache.install_cache("news_cache_v4", expire_after=CACHE_SECONDS)

# -------------------- Helpers --------------------
def now_utc():
    return datetime.now(timezone.utc)

def to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    return s

def normalize_simple(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)     # normalize dashes
    s = re.sub(r"[^a-z0-9\s\-\/&]", " ", s)    # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""

# Address canonicalization (very light: focus on number + main token)
ABBR = {
    "street":"st", "st":"st",
    "avenue":"ave","av":"ave","ave":"ave","av.":"ave",
    "road":"rd","rd":"rd",
    "boulevard":"blvd","blvd":"blvd",
    "place":"pl","pl":"pl",
    "parkway":"pkwy","pkwy":"pkwy",
    "square":"sq","sq":"sq",
    "west":"w","w":"w","east":"e","e":"e","north":"n","n":"n","south":"s","s":"s",
    "street.":"st","avenue.":"ave"
}
def canon_addr(addr: str):
    s = normalize_simple(addr)
    toks = s.split()
    num = None
    keep = []
    for t in toks:
        if t.isdigit() and num is None:
            num = t
        t2 = ABBR.get(t, t)
        keep.append(t2)
    # normalize avenue names
    s_norm = " ".join(keep)
    s_norm = s_norm.replace("avenue of the americas", "6 ave").replace("sixth avenue","6 ave")
    s_norm = s_norm.replace("seventh avenue","7 ave").replace("fifth avenue","5 ave")
    base_tokens = [t for t in s_norm.split() if t not in {"new","york","ny","nyc"}]
    main = None
    for t in base_tokens:
        if t not in {"w","e","n","s","st","ave","rd","blvd","pkwy","pl","sq"} and not t.isdigit():
            main = t
            break
    return num, main, set(base_tokens)

NEIGHBORHOOD_TOKENS = {
    "midtown","hudson yards","grand central","times square","penn station",
    "financial district","fidi","tribeca","chelsea","nomad","flatiron",
    "upper east side","upper west side","union square","meatpacking","soho","nolita"
}
BOROUGH_POS = {"manhattan","nyc","new york","new york city"}
BOROUGH_NEG = {"brooklyn","queens","bronx","staten island","long island city","lic","williamsburg","downtown brooklyn"}

NEGATIVE_PHRASES = [
    "bank of america stadium","charlotte","north carolina",
]

# Context tokens for Class A office buildings
OFFICE_TOKENS = {
    "office","lease","leases","leasing","sublease","sublet",
    "tenant","tenants","landlord","relet","office tower","office building",
    "class a","trophy"
}
NON_OFFICE_NEG = {
    "apartment","apartments","condo","condominium","co-op","residential",
    "hotel","hospitality","hostel","resort"
}
# Light brand/broker boosts (names appear in real office news frequently)
BROKER_BRANDS = {"cbre","jll","cushman","newmark","savills"}
OWNER_OPERATORS = {"sl green","vornado","brookfield","related","rxr","tishman speyer","silverstein","durst","esrt","empire state realty","hines"}

# -------------------- Load buildings --------------------
def load_buildings(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    name_cols = [c for c in df.columns if c.startswith("alternative_name")] + \
                [c for c in ["primary_building_name"] if c in df.columns]
    addr_cols = ["main_address"] + [c for c in df.columns if c.startswith("alternate_address_")]
    bbl_col = "bbl" if "bbl" in df.columns else None
    out = []
    for _, row in df.iterrows():
        names = sorted({normalize_text(row.get(c)) for c in name_cols if normalize_text(row.get(c))})
        addrs = sorted({normalize_text(row.get(c)) for c in addr_cols if normalize_text(row.get(c))})
        if not names and not addrs:
            continue
        b_id = None
        if bbl_col:
            bbl_val = normalize_text(row.get(bbl_col))
            if bbl_val:
                b_id = f"bbl-{bbl_val}"
        if not b_id:
            key = (names[0] if names else "") + "|" + (addrs[0] if addrs else "")
            b_id = "b-" + sha1(key)[:10]
        out.append({"id": b_id, "names": names, "addresses": addrs})
    return out

# -------------------- Query building --------------------
def make_query(b: Dict) -> str:
    parts = []
    if b["names"]:
        parts.append("(" + " OR ".join(f'"{n}"' for n in b["names"]) + ")")
    if b["addresses"]:
        parts.append("(" + " OR ".join(f'"{a}"' for a in b["addresses"]) + ")")
    if CITY_TERMS:
        parts.append(CITY_TERMS)
    if OFFICE_QUERY_TERMS:
        parts.append(OFFICE_QUERY_TERMS)
    if EXCLUDE_TERMS:
        parts.append(f"-{EXCLUDE_TERMS}")
    if ALLOWED_SITES:
        sites = " OR ".join([f"site:{s}" for s in ALLOWED_SITES])
        parts.append("(" + sites + ")")
    return " AND ".join(parts)

def fetch_feed(url: str):
    headers = {"User-Agent": "nyc-odcv-prospector-news/1.0"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content

def google_news_rss(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"

# -------------------- Scoring --------------------
def contains_phrase(hay: str, needle: str) -> bool:
    return needle.lower() in hay.lower()

def tokens(s: str) -> set:
    return set(normalize_simple(s).split())

def score_match(building: Dict, title: str, summary: str, url: str):
    """Returns (score, reasons)"""
    t_norm = normalize_simple(title or "")
    s_norm = normalize_simple(summary or "")
    both = t_norm + " " + s_norm

    score = 0.0
    reasons = []

    # 1) Building names (exact phrase in title or summary)
    for name in building["names"]:
        if not name: continue
        if contains_phrase(title, name):
            score += 4.0; reasons.append(f"name_title:{name}")
        elif contains_phrase(summary, name):
            score += 2.0; reasons.append(f"name_sum:{name}")
        else:
            ntoks = tokens(name) - {"the","and","of","at","on","tower","building","center"}
            if len(ntoks) >= 2 and ntoks.issubset(tokens(both)):
                score += 1.5; reasons.append(f"name_tokens:{' '.join(sorted(ntoks))}")

    # 2) Addresses — require number + distinctive street token
    for addr in building["addresses"]:
        if not addr: continue
        num, main, base = canon_addr(addr)
        if not main: 
            continue
        hit_num = (num and num in both)
        hit_main = (main in both)
        if hit_num and hit_main:
            score += 4.0; reasons.append(f"addr:{num}+{main}")
        elif hit_main:
            score += 1.5; reasons.append(f"addr_main:{main}")
        elif hit_num:
            score += 0.5; reasons.append(f"addr_num_only:{num}")

    # 3) Manhattan/Borough + neighborhoods
    if any(tok in both for tok in BOROUGH_POS):
        score += 1.2; reasons.append("borough_pos")
    if any(tok in both for tok in NEIGHBORHOOD_TOKENS):
        score += 1.0; reasons.append("neighborhood")
    if any(tok in both for tok in BOROUGH_NEG) and ("manhattan" not in both):
        score -= 1.5; reasons.append("borough_neg")

    # 4) Office/Class A context
    office_hits = [tok for tok in OFFICE_TOKENS if tok in both]
    if office_hits:
        score += 1.8; reasons.append("office_ctx:" + ",".join(office_hits))
    non_office_hits = [tok for tok in NON_OFFICE_NEG if tok in both]
    if non_office_hits and not office_hits:
        score -= 2.0; reasons.append("non_office_ctx:" + ",".join(non_office_hits))

    # 5) Source allowlist boost
    dom = extract_domain(url)
    if dom in ALLOWED_SITES:
        score += 1.5; reasons.append(f"src:{dom}")

    # 6) Brand/Broker light boosts
    if any(b in both for b in OWNER_OPERATORS):
        score += 0.8; reasons.append("owner_op")
    if any(b in both for b in BROKER_BRANDS):
        score += 0.5; reasons.append("broker")

    # 7) Negatives
    for neg in NEGATIVE_PHRASES:
        if contains_phrase(both, neg):
            score -= 10.0; reasons.append(f"neg:{neg}")

    # Penalize if only address number matched but not main token
    if any(r.startswith("addr_num_only") for r in reasons) and not any(r.startswith("addr:") or r.startswith("addr_main:") for r in reasons):
        score -= 0.5

    return round(score, 2), reasons

# -------------------- Fetch & parse --------------------
def parse_entry(entry, building_id: str) -> Dict:
    dt_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    dt = datetime(*dt_struct[:6], tzinfo=timezone.utc) if dt_struct else now_utc()
    url = entry.link
    title = entry.title
    summary = getattr(entry, "summary", "")
    source = getattr(getattr(entry, "source", None), "title", "") or getattr(entry, "author", "") or "Unknown"
    uid = sha1(f"{url}|{title}")
    return {
        "uid": uid,
        "building_id": building_id,
        "title": title,
        "url": url,
        "summary": summary,
        "source": source,
        "published_at": to_iso(dt),
    }

def fetch_for_building(b: Dict) -> List[Dict]:
    q = make_query(b)
    feed_url = google_news_rss(q)
    data = fetch_feed(feed_url)
    feed = feedparser.parse(data)
    items = []
    for e in feed.entries:
        item = parse_entry(e, b["id"])
        sc, rs = score_match(b, item["title"], item.get("summary",""), item["url"])
        item["score"] = sc
        if INCLUDE_REASONS:
            item["reasons"] = rs
        items.append(item)
    return items

# -------------------- Store --------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  uid TEXT PRIMARY KEY,
  building_id TEXT,
  title TEXT,
  url TEXT,
  summary TEXT,
  source TEXT,
  published_at TEXT,
  score REAL
);
CREATE INDEX IF NOT EXISTS idx_items_building ON items(building_id);
CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_at);
CREATE INDEX IF NOT EXISTS idx_items_score ON items(score);
"""

def ensure_columns(conn):
    cur = conn.execute("PRAGMA table_info(items)")
    cols = {r[1] for r in cur.fetchall()}
    if "score" not in cols:
        conn.execute("ALTER TABLE items ADD COLUMN score REAL")
        conn.commit()

class Store:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        ensure_columns(self._conn)
        self._conn.commit()

    def upsert_many(self, items: List[Dict]) -> int:
        if not items: return 0
        cur = self._conn.cursor()
        cur.executemany("""
            INSERT INTO items(uid, building_id, title, url, summary, source, published_at, score)
            VALUES(:uid, :building_id, :title, :url, :summary, :source, :published_at, :score)
            ON CONFLICT(uid) DO UPDATE SET
              building_id=excluded.building_id,
              title=excluded.title,
              url=excluded.url,
              summary=excluded.summary,
              source=excluded.source,
              published_at=excluded.published_at,
              score=excluded.score
        """, items)
        self._conn.commit()
        return cur.rowcount

    def list(self, building_id: Optional[str], limit: int, min_score: float, max_age_days: int):
        qs = """
            SELECT uid, building_id, title, url, summary, source, published_at, score
            FROM items
            WHERE 1=1
        """
        args = []
        if building_id:
            qs += " AND building_id = ?"
            args.append(building_id)
        if max_age_days is not None:
            cutoff = (now_utc() - timedelta(days=max_age_days)).isoformat()
            qs += " AND published_at >= ?"
            args.append(cutoff)
        if min_score is not None:
            qs += " AND score >= ?"
            args.append(min_score)
        qs += " ORDER BY published_at DESC LIMIT ?"
        args.append(limit)
        cur = self._conn.execute(qs, args)
        rows = cur.fetchall()
        cols = ["uid","building_id","title","url","summary","source","published_at","score"]
        return [dict(zip(cols, r)) for r in rows]

# -------------------- Service --------------------
def build_service(buildings: List[Dict], db_path: str):
    app = Flask(__name__)
    store = Store(db_path)
    bmap = {b["id"]: b for b in buildings}

    @app.after_request
    def cors(resp):
        origin = request.headers.get("Origin","*") or "*"
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Refresh-Token"
        return resp

    @app.route("/api/news", methods=["GET","OPTIONS"])
    def all_news():
        if request.method == "OPTIONS": return ("",204)
        limit = int(request.args.get("limit", "50"))
        min_score = float(request.args.get("min_score", str(DEFAULT_MIN_SCORE)))
        max_age_days = int(request.args.get("max_age_days", str(DEFAULT_MAX_AGE_DAYS)))
        debug = request.args.get("debug","0") == "1"
        items = store.list(None, limit, min_score, max_age_days)
        if debug and INCLUDE_REASONS:
            return jsonify(items)
        # Strip reasons if present and debug not requested
        return jsonify([{k:v for k,v in it.items() if k != "reasons"} for it in items])

    @app.route("/api/news/<building_id>", methods=["GET","OPTIONS"])
    def by_building(building_id):
        if request.method == "OPTIONS": return ("",204)
        limit = int(request.args.get("limit", "10"))
        min_score = float(request.args.get("min_score", str(DEFAULT_MIN_SCORE)))
        max_age_days = int(request.args.get("max_age_days", str(DEFAULT_MAX_AGE_DAYS)))
        debug = request.args.get("debug","0") == "1"
        items = store.list(building_id, limit, min_score, max_age_days)
        if debug and INCLUDE_REASONS:
            return jsonify(items)
        return jsonify([{k:v for k,v in it.items() if k != "reasons"} for it in items])

    @app.route("/api/news/refresh", methods=["POST","OPTIONS"])
    def refresh_all():
        if request.method == "OPTIONS": return ("",204)
        if REFRESH_TOKEN and request.headers.get("X-Refresh-Token","") != REFRESH_TOKEN:
            return jsonify({"error":"forbidden"}), 403
        total = 0
        for b in buildings:
            try:
                its = fetch_for_building(b)
                total += store.upsert_many(its)
            except Exception as e:
                print("refresh error:", b["id"], "->", e)
        return jsonify({"inserted_or_updated": total})

    @app.route("/api/news/refresh/<building_id>", methods=["POST","OPTIONS"])
    def refresh_one(building_id):
        if request.method == "OPTIONS": return ("",204)
        if REFRESH_TOKEN and request.headers.get("X-Refresh-Token","") != REFRESH_TOKEN:
            return jsonify({"error":"forbidden"}), 403
        b = bmap.get(building_id)
        if not b:
            return jsonify({"error":"unknown building"}), 404
        try:
            its = fetch_for_building(b)
            n = store.upsert_many(its)
            return jsonify({"inserted_or_updated": n})
        except Exception as e:
            print("refresh error:", b["id"], "->", e)
            return jsonify({"error": str(e)}), 500

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to all_building_addresses.csv")
    parser.add_argument("--db", default="news_v4.db", help="SQLite file path")
    parser.add_argument("--bind", default="0.0.0.0:8080", help="host:port")
    parser.add_argument("--interval", type=int, default=900, help="Background refresh interval seconds (0 disable if 0)")
    args = parser.parse_args()

    buildings = load_buildings(args.csv)
    print(f"Loaded {len(buildings)} buildings")

    app = build_service(buildings, args.db)

    # initial pass to seed - run in background to avoid blocking port binding
    import threading
    def initial_fetch():
        total = 0
        for b in buildings:
            try:
                its = fetch_for_building(b)
                total += len(its)
                Store(args.db).upsert_many(its)
            except Exception as e:
                print("initial fetch error", b["id"], "->", e)
        print(f"Initial gathered ~{total} items (pre-filter)")

    threading.Thread(target=initial_fetch, daemon=True).start()

    if args.interval and args.interval > 0:
        def loop():
            st = Store(args.db)
            while True:
                try:
                    for b in buildings:
                        its = fetch_for_building(b)
                        st.upsert_many(its)
                except Exception as e:
                    print("refresh loop error:", e)
                time.sleep(args.interval)
        threading.Thread(target=loop, daemon=True).start()

    host, port = args.bind.split(":")
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=int(port))

if __name__ == "__main__":
    main()
