#!/usr/bin/env python3
"""
News service v5 - IMPROVED version for better building coverage
----------------------------------------------------------------
Key improvements:
1. Multiple search strategies (not just exact match)
2. Broader time window (30 days instead of 10)
3. Lower minimum score threshold (3.0 instead of 7.5)
4. More flexible address matching
5. Additional search terms for real estate context
6. Fallback searches if primary search fails
"""

import argparse, hashlib, os, re, time, sqlite3, json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Optional, Tuple
from functools import lru_cache

import pandas as pd
import feedparser
import requests
import requests_cache
from flask import Flask, jsonify, request

# -------------------- Config --------------------
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
CACHE_SECONDS = int(os.getenv("CACHE_SECONDS", "900"))

# More flexible search - removed mandatory office terms
CITY_TERMS = os.getenv("CITY_TERMS", '("Manhattan" OR "New York" OR "NYC" OR "NY")')

# Broader office/real estate terms
OFFICE_QUERY_TERMS = os.getenv("OFFICE_QUERY_TERMS",
    '("office" OR "lease" OR "tenant" OR "building" OR "real estate" OR "property" OR "tower" OR "space" OR "floor" OR "renovation" OR "sale" OR "acquisition")')

# Expanded allowed sites
ALLOWED_SITES = [s.strip().lower() for s in os.getenv("ALLOWED_SITES",
    "therealdeal.com,commercialobserver.com,crainsnewyork.com,bisnow.com,nytimes.com,wsj.com,bloomberg.com,globest.com,ny1.com,nypost.com,rebusinessonline.com,newyorkyimby.com").split(",") if s.strip()]

# More lenient defaults
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "3.0"))  # Lowered from 7.5
DEFAULT_MAX_AGE_DAYS = int(os.getenv("DEFAULT_MAX_AGE_DAYS", "30"))  # Increased from 10

REFRESH_TOKEN = os.getenv("NEWS_REFRESH_TOKEN", "").strip()
INCLUDE_REASONS = os.getenv("INCLUDE_REASONS", "1").strip() == "1"  # Enable by default for debugging

requests_cache.install_cache("news_cache_v5", expire_after=CACHE_SECONDS)

# ---------- Thumbnails (same as v4) ----------
def get_thumbnail_for_source(source_name: str) -> dict:
    """Get thumbnail with fallback hierarchy"""
    local_logos = {
        "The Real Deal": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/therealdeal.png",
        "Commercial Observer": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/commercialobserver.png",
        "Crain's New York Business": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/crainsnewyork.png",
        "New York Times": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/nytimes.png",
        "Wall Street Journal": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/wsj.png",
        "Bloomberg": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/Bloomberg.png",
        "Bisnow": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/bisnow.png",
        "GlobeSt": "https://nyc-odcv-images.s3.us-east-2.amazonaws.com/logos/globest.png"
    }

    if source_name in local_logos:
        return {"image": local_logos[source_name], "source": "local_logo"}

    # Fallback
    return {"image": "https://rzero.com/wp-content/themes/rzero/build/images/favicons/favicon.png", "source": "rzero"}

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
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9\s\-\/&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except:
        return ""

# -------------------- Improved Scoring --------------------
POSITIVE_KEYWORDS = {
    # Building activities
    "lease", "leases", "leasing", "tenant", "tenants", "rent", "rental", "occupancy",
    "renovation", "upgrade", "construction", "development", "redevelopment",
    "sale", "sold", "acquisition", "purchase", "investment", "investor",

    # Companies/organizations
    "company", "firm", "corporation", "headquarters", "office", "offices",
    "expand", "expansion", "relocate", "relocation", "move", "moving",

    # Real estate terms
    "square feet", "sq ft", "floor", "floors", "space", "building", "tower",
    "property", "real estate", "commercial", "class a", "class b",

    # Market terms
    "market", "deal", "transaction", "broker", "brokerage", "landlord", "owner"
}

def score_match_v5(building: Dict, title: str, summary: str, url: str) -> Tuple[float, List[str]]:
    """Improved scoring that's more lenient and comprehensive"""
    t_norm = normalize_simple(title or "")
    s_norm = normalize_simple(summary or "")
    combined = t_norm + " " + s_norm

    score = 0.0
    reasons = []

    # 1. Check for building name (more lenient)
    if building.get("primary_name"):
        name_norm = normalize_simple(building["primary_name"])
        name_parts = name_norm.split()

        # Full name match
        if name_norm in combined:
            score += 5.0
            reasons.append(f"full_name:{building['primary_name']}")
        # Partial name match (at least 2 significant words)
        elif len(name_parts) > 1:
            matching_parts = [p for p in name_parts if len(p) > 3 and p in combined]
            if len(matching_parts) >= 2:
                score += 3.0
                reasons.append(f"partial_name:{' '.join(matching_parts)}")

    # 2. Check for address (more flexible)
    if building.get("primary_address"):
        addr = building["primary_address"]
        addr_norm = normalize_simple(addr)

        # Extract street number and street name
        addr_parts = addr.split()
        street_num = None
        street_name = None

        if addr_parts and addr_parts[0].isdigit():
            street_num = addr_parts[0]
            if len(addr_parts) > 1:
                # Get the main street name (not direction like W/E)
                for part in addr_parts[1:]:
                    if part.lower() not in ['w', 'e', 'n', 's', 'west', 'east', 'north', 'south']:
                        street_name = part
                        break

        # Full address match
        if addr_norm in combined:
            score += 5.0
            reasons.append(f"full_address:{addr}")
        # Street number + street name
        elif street_num and street_name:
            street_name_norm = normalize_simple(street_name)
            if street_num in combined and street_name_norm in combined:
                # Check if they appear near each other (within 50 chars)
                num_pos = combined.find(street_num)
                name_pos = combined.find(street_name_norm)
                if abs(num_pos - name_pos) < 50:
                    score += 4.0
                    reasons.append(f"street_match:{street_num} {street_name}")
                else:
                    score += 2.0
                    reasons.append(f"street_parts:{street_num}+{street_name}")
        # Just street name (common for famous buildings)
        elif street_name:
            street_name_norm = normalize_simple(street_name)
            if street_name_norm in combined and len(street_name_norm) > 3:
                score += 1.5
                reasons.append(f"street_name:{street_name}")

    # 3. Check for NYC context
    nyc_terms = ["manhattan", "nyc", "new york", "midtown", "downtown", "financial district",
                 "times square", "wall street", "park avenue", "fifth avenue", "madison avenue"]
    for term in nyc_terms:
        if term in combined:
            score += 0.5
            reasons.append(f"nyc:{term}")
            break

    # 4. Check for real estate context (more keywords)
    re_keywords_found = []
    for keyword in POSITIVE_KEYWORDS:
        if keyword in combined:
            re_keywords_found.append(keyword)

    if re_keywords_found:
        score += min(2.0, len(re_keywords_found) * 0.3)  # Up to 2 points for context
        reasons.append(f"re_context:{','.join(re_keywords_found[:5])}")  # Show first 5

    # 5. Trusted source bonus
    domain = extract_domain(url)
    if domain in ALLOWED_SITES:
        score += 1.0
        reasons.append(f"trusted_source:{domain}")

    # 6. Negative signals (less aggressive)
    negative_terms = ["residential", "apartment", "condo", "hotel"]
    if not any(term in ["office", "commercial", "tenant", "lease"] for term in combined.split()):
        # Only apply negative if NO commercial context
        for neg in negative_terms:
            if neg in combined:
                score -= 1.0
                reasons.append(f"negative:{neg}")
                break

    return round(score, 2), reasons

# -------------------- Query Building --------------------
def make_queries_v5(building: Dict) -> List[str]:
    """Generate multiple search queries with different strategies"""
    queries = []

    # Extract components
    name = building.get("primary_name", "")
    address = building.get("primary_address", "")

    # Strategy 1: Exact name + NYC (if name exists)
    if name:
        queries.append(f'"{name}" Manhattan OR NYC')

    # Strategy 2: Address + NYC
    if address:
        queries.append(f'"{address}" Manhattan OR NYC')

    # Strategy 3: Simplified name (remove common suffixes)
    if name:
        simple_name = name.replace(" Building", "").replace(" Tower", "").replace(" Center", "")
        if simple_name != name:
            queries.append(f'"{simple_name}" Manhattan building')

    # Strategy 4: Street name only + building/tower
    if address:
        parts = address.split()
        if len(parts) >= 2:
            # Find the main street name
            street_parts = []
            for part in parts[1:]:  # Skip number
                if part.lower() not in ['st', 'street', 'ave', 'avenue', 'pl', 'place']:
                    street_parts.append(part)
            if street_parts:
                street_name = ' '.join(street_parts)
                queries.append(f'"{street_name}" Manhattan office building')

    # Strategy 5: Combined partial match
    if name and address:
        # Use first number from address and first significant word from name
        addr_parts = address.split()
        name_parts = [p for p in name.split() if len(p) > 3]
        if addr_parts and addr_parts[0].isdigit() and name_parts:
            queries.append(f'{addr_parts[0]} {name_parts[0]} NYC')

    # Remove duplicates while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique_queries.append(q)

    return unique_queries[:3]  # Return top 3 queries

def fetch_feed(url: str) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.content

def parse_entry(entry, building_id: str) -> Dict:
    """Parse RSS entry into our format"""
    dt_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    dt = datetime(*dt_struct[:6], tzinfo=timezone.utc) if dt_struct else now_utc()

    url = entry.link
    title = entry.title
    summary = getattr(entry, "summary", "")
    source = getattr(getattr(entry, "source", None), "title", "") or "Unknown"

    # Clean up title
    if " - " in title:
        title = title.rsplit(" - ", 1)[0].strip()

    uid = sha1(f"{url}|{title}")
    thumb = get_thumbnail_for_source(source)

    return {
        "uid": uid,
        "building_id": building_id,
        "title": title,
        "url": url,
        "summary": summary,
        "source": source,
        "published_at": to_iso(dt),
        "thumbnail_url": thumb["image"],
    }

def fetch_for_building_v5(building: Dict) -> List[Dict]:
    """Fetch news using multiple strategies"""
    items = []
    seen_urls = set()

    queries = make_queries_v5(building)

    for query in queries:
        if not query:
            continue

        # Try multiple news sources
        sources = [
            f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en",
            f"https://www.bing.com/news/search?q={quote_plus(query)}&format=rss",
        ]

        for feed_url in sources:
            try:
                data = fetch_feed(feed_url)
                feed = feedparser.parse(data)

                for entry in getattr(feed, "entries", [])[:10]:  # Process more entries
                    item = parse_entry(entry, building["id"])

                    # Skip if we've seen this URL
                    if item["url"] in seen_urls:
                        continue
                    seen_urls.add(item["url"])

                    # Score the match
                    score, reasons = score_match_v5(building, item["title"], item.get("summary", ""), item["url"])
                    item["score"] = score
                    item["reasons"] = reasons
                    item["search_query"] = query  # Track which query found it

                    items.append(item)

                # If we found good results, we might stop
                good_items = [it for it in items if it["score"] >= 3.0]
                if len(good_items) >= 3:
                    break  # Have enough good results

            except Exception as e:
                print(f"Feed error for {building['id']} query '{query}': {e}")
                continue

        # Small delay between queries
        if queries.index(query) < len(queries) - 1:
            time.sleep(0.2)

    # Sort by score and return
    items.sort(key=lambda x: x["score"], reverse=True)
    return items

# -------------------- Database --------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  uid TEXT PRIMARY KEY,
  building_id TEXT,
  title TEXT,
  url TEXT,
  summary TEXT,
  source TEXT,
  published_at TEXT,
  score REAL,
  thumbnail_url TEXT,
  search_query TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_building ON items(building_id);
CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_at);
CREATE INDEX IF NOT EXISTS idx_items_score ON items(score);
"""

class Store:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def upsert_many(self, items: List[Dict]) -> int:
        if not items: return 0
        cur = self._conn.cursor()
        cur.executemany("""
            INSERT INTO items(uid, building_id, title, url, summary, source, published_at, score, thumbnail_url, search_query)
            VALUES(:uid, :building_id, :title, :url, :summary, :source, :published_at, :score, :thumbnail_url, :search_query)
            ON CONFLICT(uid) DO UPDATE SET
              score=excluded.score,
              search_query=excluded.search_query
        """, [
            {**item, 'search_query': item.get('search_query', '')}
            for item in items
        ])
        self._conn.commit()
        return cur.rowcount

    def list(self, building_id: Optional[str], limit: int, min_score: float, max_age_days: int):
        qs = """
            SELECT uid, building_id, title, url, summary, source, published_at, score, thumbnail_url
            FROM items WHERE 1=1
        """
        args = []
        if building_id:
            qs += " AND building_id = ?"; args.append(building_id)
        if max_age_days is not None:
            cutoff = (now_utc() - timedelta(days=max_age_days)).isoformat()
            qs += " AND published_at >= ?"; args.append(cutoff)
        if min_score is not None:
            qs += " AND score >= ?"; args.append(min_score)
        qs += " ORDER BY score DESC, published_at DESC LIMIT ?"; args.append(limit)

        cur = self._conn.execute(qs, args)
        rows = cur.fetchall()
        cols = ["uid","building_id","title","url","summary","source","published_at","score","thumbnail_url"]
        return [dict(zip(cols, r)) for r in rows]

# -------------------- Load Buildings --------------------
def load_buildings(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    buildings = []

    for _, row in df.iterrows():
        primary_name = normalize_text(row.get("primary_building_name"))
        primary_address = normalize_text(row.get("main_address"))

        if not (primary_name or primary_address):
            continue

        addr_clean = re.sub(r'[^a-zA-Z0-9]', '', primary_address or primary_name or "")
        b_id = f"bld-{addr_clean.lower()}"

        buildings.append({
            "id": b_id,
            "primary_name": primary_name,
            "primary_address": primary_address,
            "bbl": str(row.get("bbl", ""))
        })

    return buildings

# -------------------- Service --------------------
def build_service(buildings: List[Dict], db_path: str):
    app = Flask(__name__)
    store = Store(db_path)
    bmap = {b["id"]: b for b in buildings}

    # Also create BBL lookup
    bbl_map = {}
    for b in buildings:
        if b.get("bbl"):
            bbl_map[b["bbl"]] = b

    @app.after_request
    def cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/healthz")
    def healthz():
        return jsonify({"ok": True, "version": "v5", "time": to_iso(now_utc())})

    @app.route("/api/news", methods=["GET"])
    def all_news():
        limit = int(request.args.get("limit", "50"))
        min_score = float(request.args.get("min_score", str(DEFAULT_MIN_SCORE)))
        max_age_days = int(request.args.get("max_age_days", str(DEFAULT_MAX_AGE_DAYS)))

        items = store.list(None, limit, min_score, max_age_days)
        return jsonify(items)

    @app.route("/api/news/<building_id>", methods=["GET"])
    def by_building(building_id):
        limit = int(request.args.get("limit", "20"))
        min_score = float(request.args.get("min_score", str(DEFAULT_MIN_SCORE)))
        max_age_days = int(request.args.get("max_age_days", str(DEFAULT_MAX_AGE_DAYS)))
        force_refresh = request.args.get("refresh", "0") == "1"

        # Handle BBL format
        building = None
        if building_id.startswith("bbl-"):
            bbl = building_id[4:]
            building = bbl_map.get(bbl)
        else:
            building = bmap.get(building_id)

        if not building:
            return jsonify({"error": f"Building not found: {building_id}"}), 404

        # Check cache first
        items = store.list(building["id"], limit, min_score, max_age_days)

        # If no results or force refresh, fetch new
        if not items or force_refresh:
            try:
                new_items = fetch_for_building_v5(building)
                if new_items:
                    store.upsert_many(new_items)
                    items = store.list(building["id"], limit, min_score, max_age_days)
            except Exception as e:
                print(f"Fetch error for {building_id}: {e}")

        # Include reasons if requested
        if request.args.get("debug") == "1" and INCLUDE_REASONS:
            return jsonify(items)
        else:
            # Strip reasons from response
            clean_items = []
            for item in items:
                clean_item = {k: v for k, v in item.items() if k != "reasons"}
                clean_items.append(clean_item)
            return jsonify(clean_items)

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--db", default="news_v5.db", help="SQLite database path")
    parser.add_argument("--bind", default="0.0.0.0:8080", help="host:port")
    args = parser.parse_args()

    buildings = load_buildings(args.csv)
    print(f"Loaded {len(buildings)} buildings (v5 improved)")

    app = build_service(buildings, args.db)

    host, port_str = args.bind.split(":")
    port = int(os.getenv("PORT", port_str))
    print(f"Starting improved news service v5 on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()