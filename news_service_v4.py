#!/usr/bin/env python3
"""
News service v4 — Manhattan Class A office, PRIMARY-FIRST preference
--------------------------------------------------------------------
- Reads buildings from CSV, preserving PRIMARY vs ALTERNATE fields:
    * primary_building_name       -> primary_name
    * alternative_name_*          -> alt_names[]
    * main_address                -> primary_address
    * alternate_address_*         -> alt_addresses[]
- Builds TWO search tiers per building:
    TIER 1 (primary, strict): (primary_name OR primary_address) AND city AND office terms
    TIER 2 (expanded): (primary+alt names/addresses) AND city AND office terms
- Confidence scoring PREFERENCES:
    * PRIMARY NAME in title: +6.0; in summary: +3.0; token overlap: +2.0
    * ALT NAME in title: +3.5; in summary: +1.5; token overlap: +1.0
    * PRIMARY ADDRESS num+main: +6.0; main-only: +2.5; num-only: +1.0
    * ALT ADDRESS num+main: +4.0; main-only: +1.5; num-only: +0.5
    * Manhattan/NYC +1.2; other boroughs w/o Manhattan −1.5
    * Office context (office/lease/tenant/"class a") +1.8
    * Allow‑listed outlets +1.5
    * Owner-operator +0.8; Broker +0.5
    * Non-office (residential/hospitality) −2.0 (only if no office context)
    * Strong negatives (e.g., "Bank of America Stadium") −10.0
- JSON API filters: min_score, max_age_days, limit; optional ?debug=1 returns reasons if INCLUDE_REASONS=1

ENV (recommended defaults for Class A):
  ALLOWED_SITES="therealdeal.com,commercialobserver.com,crainsnewyork.com,bisnow.com,nytimes.com,wsj.com,bloomberg.com,globest.com"
  CITY_TERMS='("Manhattan" OR "New York" OR "NYC")'
  OFFICE_QUERY_TERMS='("office" OR "lease" OR "leases" OR "leasing" OR "tenant" OR "tenants" OR "office tower" OR "class A")'
  EXCLUDE_TERMS='("Bank of America Stadium" OR Charlotte OR "North Carolina")'
  DEFAULT_MIN_SCORE=7.5
  DEFAULT_MAX_AGE_DAYS=10
  CACHE_SECONDS=900
  REQUEST_TIMEOUT=10
  INCLUDE_REASONS=0        # set 1 to include reasons in output when ?debug=1
  NEWS_REFRESH_TOKEN=""    # optional shared secret for POST /refresh endpoints (public if used from client JS)
"""

import argparse, hashlib, os, re, time, sqlite3
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Optional, Tuple

import pandas as pd
import feedparser
import requests
import requests_cache
from flask import Flask, jsonify, request

# -------------------- Config --------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "900"))
CITY_TERMS = os.getenv("CITY_TERMS", '("Manhattan" OR "New York" OR "NYC")')
OFFICE_QUERY_TERMS = os.getenv("OFFICE_QUERY_TERMS", '("office" OR "lease" OR "leases" OR "leasing" OR "tenant" OR "tenants" OR "office tower" OR "class A")')
EXCLUDE_TERMS = os.getenv("EXCLUDE_TERMS", '("Bank of America Stadium" OR Charlotte OR "North Carolina")')
ALLOWED_SITES = [s.strip().lower() for s in os.getenv("ALLOWED_SITES", "").split(",") if s.strip()]
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "7.5"))
DEFAULT_MAX_AGE_DAYS = int(os.getenv("DEFAULT_MAX_AGE_DAYS", "10"))
REFRESH_TOKEN = os.getenv("NEWS_REFRESH_TOKEN", "").strip()
INCLUDE_REASONS = os.getenv("INCLUDE_REASONS", "0").strip() == "1"
ENABLE_REFRESH_ALL = os.getenv("ENABLE_REFRESH_ALL", "0") == "1"

requests_cache.install_cache("news_cache_v4", expire_after=CACHE_SECONDS)

# -------------------- Utils --------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_text(x: Optional[str]) -> Optional[str]:
    if x is None: return None
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

# Address canonicalization (light)
ABBR = {
    "street":"st","st":"st",
    "avenue":"ave","av":"ave","ave":"ave","av.":"ave",
    "road":"rd","rd":"rd",
    "boulevard":"blvd","blvd":"blvd",
    "place":"pl","pl":"pl",
    "parkway":"pkwy","pkwy":"pkwy",
    "square":"sq","sq":"sq",
    "west":"w","w":"w","east":"e","e":"e","north":"n","n":"n","south":"s","s":"s",
    "street.":"st","avenue.":"ave"
}
def canon_addr(addr: str) -> Tuple[Optional[str], Optional[str], set]:
    s = normalize_simple(addr)
    toks = s.split()
    num = None
    keep = []
    for t in toks:
        if t.isdigit() and num is None:
            num = t
        keep.append(ABBR.get(t, t))
    s_norm = " ".join(keep)
    s_norm = s_norm.replace("avenue of the americas","6 ave").replace("sixth avenue","6 ave")
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

NEGATIVE_PHRASES = ["bank of america stadium","charlotte","north carolina"]

OFFICE_TOKENS = {
    "office","lease","leases","leasing","sublease","sublet",
    "tenant","tenants","landlord","relet","office tower","office building",
    "class a","trophy"
}
NON_OFFICE_NEG = {
    "apartment","apartments","condo","condominium","co-op","residential",
    "hotel","hospitality","hostel","resort"
}
BROKER_BRANDS = {"cbre","jll","cushman","newmark","savills"}
OWNER_OPERATORS = {"sl green","vornado","brookfield","related","rxr","tishman speyer","silverstein","durst","esrt","empire state realty","hines"}

# -------------------- Buildings --------------------
def load_buildings(csv_path: str) -> List[Dict]:
    # Use the News Search Addresses CSV with exact search terms
    df = pd.read_csv("data/news_search_addresses.csv", sep=',')
    
    out: List[Dict] = []
    for i, row in df.iterrows():
        primary_name = normalize_text(row.get("primary_building_name"))
        primary_address = normalize_text(row.get("main_address"))

        if not (primary_name or primary_address):
            continue

        # Create stable building ID from address
        addr_clean = re.sub(r'[^a-zA-Z0-9]', '', primary_address or primary_name or "")
        b_id = f"bld-{addr_clean.lower()}"

        out.append({
            "id": b_id,
            "primary_name": primary_name,
            "alt_names": [],
            "primary_address": primary_address,
            "alt_addresses": [],
        })
    return out

# -------------------- Query building --------------------
def group_or(phrases: List[str]) -> Optional[str]:
    phrases = [p for p in phrases if p]
    if not phrases: return None
    return "(" + " OR ".join(f'"{p}"' for p in phrases) + ")"

def make_query(b: Dict, tier: str) -> str:
    # Search for full street address in quotes
    search_terms = []
    
    if b.get("primary_address"):
        # Use the FULL address in quotes (e.g. "4 Times Square")
        full_addr = b["primary_address"].strip()
        search_terms.append(f'"{full_addr}"')
    
    if b.get("primary_name"):
        search_terms.append(f'"{b["primary_name"]}"')
    
    if not search_terms:
        return ""
    
    # Just the address/name + nyc
    return f"({' OR '.join(search_terms)}) nyc"

def google_news_rss(query: str) -> str:
    # Use Bing News RSS - less aggressive blocking than Google
    return f"https://www.bing.com/news/search?q={quote_plus(query)}&format=rss"

def fetch_feed(url: str) -> bytes:
    headers = {"User-Agent": "nyc-odcv-prospector-news/1.0"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content

# -------------------- Scoring --------------------
def contains_phrase(hay: str, needle: str) -> bool:
    return needle and needle.lower() in (hay or "").lower()

def tokens(s: str) -> set:
    return set(normalize_simple(s or "").split())

def score_names(title: str, summary: str, name: str, primary: bool) -> Tuple[float, List[str]]:
    sc, rs = 0.0, []
    if not name: return sc, rs
    if contains_phrase(title, name):
        sc += 6.0 if primary else 3.5; rs.append(("primary_name_title" if primary else "alt_name_title") + f":{name}")
    elif contains_phrase(summary, name):
        sc += 3.0 if primary else 1.5; rs.append(("primary_name_sum" if primary else "alt_name_sum") + f":{name}")
    else:
        ntoks = tokens(name) - {"the","and","of","at","on","tower","building","center"}
        if len(ntoks) >= 2 and ntoks.issubset(tokens(title + " " + summary)):
            sc += 2.0 if primary else 1.0; rs.append(("primary_name_tokens" if primary else "alt_name_tokens") + ":" + " ".join(sorted(ntoks)))
    return sc, rs

def score_addr(title: str, summary: str, addr: str, primary: bool) -> Tuple[float, List[str]]:
    sc, rs = 0.0, []
    if not addr: return sc, rs
    both = normalize_simple(title + " " + summary)
    num, main, _ = canon_addr(addr)
    hit_num = bool(num and num in both)
    hit_main = bool(main and main in both)
    if hit_num and hit_main:
        sc += 6.0 if primary else 4.0; rs.append(("primary_addr" if primary else "alt_addr") + f":{num}+{main}")
    elif hit_main:
        sc += 2.5 if primary else 1.5; rs.append(("primary_addr_main" if primary else "alt_addr_main") + f":{main}")
    elif hit_num:
        sc += 1.0 if primary else 0.5; rs.append(("primary_addr_num_only" if primary else "alt_addr_num_only") + f":{num}")
    return sc, rs

def score_match(b: Dict, title: str, summary: str, url: str) -> Tuple[float, List[str]]:
    t_norm = normalize_simple(title or "")
    s_norm = normalize_simple(summary or "")
    both = t_norm + " " + s_norm

    score = 0.0
    reasons: List[str] = []

    # Names
    pn = b.get("primary_name")
    if pn:
        sc, rs = score_names(title, summary, pn, primary=True); score += sc; reasons += rs
    for an in b.get("alt_names", []):
        sc, rs = score_names(title, summary, an, primary=False); score += sc; reasons += rs

    # Addresses
    pa = b.get("primary_address")
    if pa:
        sc, rs = score_addr(title, summary, pa, primary=True); score += sc; reasons += rs
    for aa in b.get("alt_addresses", []):
        sc, rs = score_addr(title, summary, aa, primary=False); score += sc; reasons += rs

    # Borough/neighborhood
    if any(tok in both for tok in BOROUGH_POS): score += 1.2; reasons.append("borough_pos")
    if any(tok in both for tok in NEIGHBORHOOD_TOKENS): score += 1.0; reasons.append("neighborhood")
    if any(tok in both for tok in BOROUGH_NEG) and ("manhattan" not in both): score -= 1.5; reasons.append("borough_neg")

    # Office/Class A context
    office_hits = [tok for tok in OFFICE_TOKENS if tok in both]
    if office_hits: score += 1.8; reasons.append("office_ctx:" + ",".join(office_hits))
    non_office_hits = [tok for tok in NON_OFFICE_NEG if tok in both]
    if non_office_hits and not office_hits: score -= 2.0; reasons.append("non_office_ctx:" + ",".join(non_office_hits))

    # Source allowlist
    dom = extract_domain(url)
    if dom in ALLOWED_SITES: score += 1.5; reasons.append(f"src:{dom}")

    # Owners/Brokers
    if any(bd in both for bd in OWNER_OPERATORS): score += 0.8; reasons.append("owner_op")
    if any(bb in both for bb in BROKER_BRANDS): score += 0.5; reasons.append("broker")

    # Strong negatives
    for neg in NEGATIVE_PHRASES:
        if neg in both: score -= 10.0; reasons.append(f"neg:{neg}")

    # Address-only-number penalty (no main street)
    if any(r.endswith("addr_num_only") for r in reasons) and not any("addr:" in r or "addr_main" in r for r in reasons):
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
    import re
    img_match = re.search(r'<img[^>]+src="([^"]+)"', summary)
    thumbnail_url = img_match.group(1) if img_match else None
    return {
        "uid": uid,
        "building_id": building_id,
        "title": title,
        "url": url,
        "summary": summary,
        "source": source,
        "published_at": to_iso(dt),
        "thumbnail_url": thumbnail_url,
    }

def fetch_for_building(b: Dict) -> List[Dict]:
    items = []
    seen = set()
    
    q = make_query(b, "primary")
    if not q:
        return items
        
    # Try both Google and Bing News sources
    sources = [
        f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en",
        f"https://www.bing.com/news/search?q={quote_plus(q)}&format=rss"
    ]
    
    for feed_url in sources:
        try:
            data = fetch_feed(feed_url)
            feed = feedparser.parse(data)
            if feed.entries:  # Got results, use this source
                break
        except Exception as e:
            print("feed error", b["id"], "->", e)
            continue
    else:
        # No sources worked
        return items
        
    for e in getattr(feed, "entries", []):
        item = parse_entry(e, b["id"])
        if item["uid"] in seen: continue
        sc, rs = score_match(b, item["title"], item.get("summary",""), item["url"])
        item["score"] = sc
        if INCLUDE_REASONS: item["reasons"] = rs
        items.append(item)
        seen.add(item["uid"])

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
            qs += " AND building_id = ?"; args.append(building_id)
        if max_age_days is not None:
            cutoff = (now_utc() - timedelta(days=max_age_days)).isoformat()
            qs += " AND published_at >= ?"; args.append(cutoff)
        if min_score is not None:
            qs += " AND score >= ?"; args.append(min_score)
        qs += " ORDER BY published_at DESC LIMIT ?"; args.append(limit)
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

    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify({"ok": True, "time": to_iso(now_utc())})

    @app.route("/api/news", methods=["GET","OPTIONS"])
    def all_news():
        if request.method == "OPTIONS": return ("",204)
        limit = int(request.args.get("limit", "50"))
        min_score = float(request.args.get("min_score", str(DEFAULT_MIN_SCORE)))
        max_age_days = int(request.args.get("max_age_days", str(DEFAULT_MAX_AGE_DAYS)))
        debug = request.args.get("debug","0") == "1"
        items = store.list(None, limit, min_score, max_age_days)
        if debug and INCLUDE_REASONS: return jsonify(items)
        return jsonify([{k:v for k,v in it.items() if k != "reasons"} for it in items])

    @app.route("/api/news/<building_id>", methods=["GET","OPTIONS"])
    def by_building(building_id):
        if request.method == "OPTIONS":
            return ("", 204)

        limit = int(request.args.get("limit", "10"))
        min_score = float(request.args.get("min_score", str(DEFAULT_MIN_SCORE)))
        max_age_days = int(request.args.get("max_age_days", str(DEFAULT_MAX_AGE_DAYS)))
        debug = request.args.get("debug", "0") == "1"
        force_refresh = request.args.get("refresh", "0") == "1"

        # Handle BBL format by looking up address in CSV
        if building_id.startswith("bbl-"):
            bbl = building_id[4:]  # Remove 'bbl-' prefix
            try:
                import pandas as pd
                df = pd.read_csv("data/news_search_addresses.csv", sep=',')
                df['bbl_str'] = df['bbl'].astype(str)  
                match = df[df['bbl_str'] == str(bbl)]
                if not match.empty:
                    address = match.iloc[0]["main_address"]
                    building_name = match.iloc[0].get("primary_building_name", "")
                    # Create a synthetic building for this BBL using the ADDRESS for news search
                    b = {"id": building_id, "main_address": address, "primary_building_name": building_name}
                    print(f"BBL {bbl} -> Address: {address}, Building: {building_name}")
                else:
                    return jsonify({"error": f"BBL {bbl} not found in {len(df)} rows"}), 404
            except Exception as e:
                return jsonify({"error": f"BBL lookup failed: {e}"}), 500
        else:
            b = bmap.get(building_id)

        # Get whatever we have cached first
        items = store.list(building_id, limit, min_score, max_age_days)

        # If nothing cached (or caller asked), do a synchronous fetch -> upsert -> read again
        if (not items or force_refresh) and b:
            try:
                new_items = fetch_for_building(b)
                store.upsert_many(new_items)
                items = store.list(building_id, limit, min_score, max_age_days)
            except Exception as e:
                print("lazy fetch error:", building_id, "->", e)

        if debug and INCLUDE_REASONS:
            return jsonify(items)
        return jsonify([{k: v for k, v in it.items() if k != "reasons"} for it in items])

    if ENABLE_REFRESH_ALL:
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

    # Skip initial fetch - only fetch on-demand
    st = Store(args.db)
    print("Service ready - will fetch news on-demand when requested")

    # Disable background refresh to prevent batch operations
    # if args.interval and args.interval > 0:
    #     import threading
    #     def loop():
    #         while True:
    #             try:
    #                 for b in buildings:
    #                     st.upsert_many(fetch_for_building(b))
    #             except Exception as e:
    #                 print("refresh loop error:", e)
    #             time.sleep(args.interval)
    #     threading.Thread(target=loop, daemon=True).start()

    host, port_str = args.bind.split(":")
    # Use PORT environment variable for Render compatibility
    port = int(os.getenv("PORT", port_str))
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)

# Create app instance for gunicorn
app = None

if __name__ == "__main__":
    main()
else:
    # For gunicorn deployment
    import argparse
    buildings = load_buildings("data/all_building_addresses.csv")
    app = build_service(buildings, "news_v4.db")