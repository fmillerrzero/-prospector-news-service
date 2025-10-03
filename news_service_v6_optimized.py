#!/usr/bin/env python3
"""
News service v6 - OPTIMIZED version with alternate addresses and better coverage
--------------------------------------------------------------------------------
Key improvements over v5:
1. Integrates alternate addresses from all_building_addresses.csv
2. Enhanced search strategies using multiple addresses per building
3. Negative scoring for theatrical Broadway content
4. Lower threshold (2.5) and smarter scoring
5. Parallel processing support
6. Coverage tracking and monitoring
"""

import argparse, hashlib, os, re, time, sqlite3, json, logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, quote_plus
from typing import List, Dict, Optional, Tuple, Set
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import feedparser
import requests
import requests_cache
from flask import Flask, jsonify, request

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    "therealdeal.com,commercialobserver.com,crainsnewyork.com,bisnow.com,nytimes.com,wsj.com,bloomberg.com,globest.com,ny1.com,nypost.com,rebusinessonline.com,newyorkyimby.com,ny.curbed.com,6sqft.com,archpaper.com").split(",") if s.strip()]

# More lenient defaults
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "2.5"))  # Lowered from 3.0
DEFAULT_MAX_AGE_DAYS = int(os.getenv("DEFAULT_MAX_AGE_DAYS", "180"))  # 6 months max

REFRESH_TOKEN = os.getenv("NEWS_REFRESH_TOKEN", "").strip()
INCLUDE_REASONS = os.getenv("INCLUDE_REASONS", "1").strip() == "1"  # Enable by default for debugging

# Parallel processing settings
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "10"))

requests_cache.install_cache("news_cache_v6", expire_after=CACHE_SECONDS)

# ---------- Thumbnails (same as v5) ----------
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
    if not s or s.lower() in {"nan", "none", "", "n/a"}:
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

# -------------------- Enhanced Scoring with Theatrical Filter --------------------
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

# Theatrical terms to penalize for Broadway addresses
THEATRICAL_KEYWORDS = {
    "musical", "theater", "theatre", "broadway show", "performance", "actor",
    "actress", "play", "stage", "curtain", "ticket", "matinee", "production",
    "tony award", "opening night", "revival", "cast", "rehearsal", "premiere"
}

def score_match_v6(building: Dict, title: str, summary: str, url: str) -> Tuple[float, List[str]]:
    """Enhanced scoring with alternate addresses and theatrical filtering"""
    t_norm = normalize_simple(title or "")
    s_norm = normalize_simple(summary or "")
    combined = t_norm + " " + s_norm

    score = 0.0
    reasons = []
    addresses_matched = set()

    # 1. Check primary building name
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

    # 2. Check alternate name if exists
    if building.get("alternative_name"):
        alt_name_norm = normalize_simple(building["alternative_name"])
        if alt_name_norm and alt_name_norm in combined:
            score += 4.0
            reasons.append(f"alt_name:{building['alternative_name']}")

    # 3. Check ALL addresses (primary and alternates)
    all_addresses = []
    if building.get("primary_address"):
        all_addresses.append(building["primary_address"])

    # Add alternate addresses
    for i in range(100):  # Check up to 100 alternate address columns
        addr_key = f"alternate_address_{i}"
        if building.get(addr_key):
            all_addresses.append(building[addr_key])

    # Score each address
    for addr in all_addresses[:10]:  # Check top 10 addresses
        if not addr:
            continue

        addr_norm = normalize_simple(addr)

        # Extract street number and street name
        addr_parts = addr.split()
        street_num = None
        street_name = None

        if addr_parts and addr_parts[0].replace('-', '').replace('/', '').isdigit():
            street_num = addr_parts[0]
            if len(addr_parts) > 1:
                # Get the main street name
                for part in addr_parts[1:]:
                    if part.lower() not in ['w', 'e', 'n', 's', 'west', 'east', 'north', 'south', 'st', 'ave', 'pl']:
                        street_name = part
                        break

        # Full address match
        if addr_norm in combined:
            addresses_matched.add(addr)
            score += 4.0
            reasons.append(f"full_addr:{addr}")
        # Street number + street name
        elif street_num and street_name:
            street_name_norm = normalize_simple(street_name)
            if street_num in combined and street_name_norm in combined:
                # Check if they appear near each other
                num_pos = combined.find(street_num)
                name_pos = combined.find(street_name_norm)
                if abs(num_pos - name_pos) < 50:
                    addresses_matched.add(addr)
                    score += 3.0
                    reasons.append(f"street_match:{street_num} {street_name}")

    # 4. Bonus for multiple address matches
    if len(addresses_matched) > 1:
        score += 2.0
        reasons.append(f"multi_addr:{len(addresses_matched)}")

    # 5. NYC context
    nyc_terms = ["manhattan", "nyc", "new york", "midtown", "downtown", "financial district",
                 "times square", "wall street", "park avenue", "fifth avenue", "madison avenue",
                 "hudson yards", "chelsea", "tribeca", "soho", "flatiron", "nomad", "fidi"]
    for term in nyc_terms:
        if term in combined:
            score += 0.5
            reasons.append(f"nyc:{term}")
            break

    # 6. Real estate context
    re_keywords_found = []
    for keyword in POSITIVE_KEYWORDS:
        if keyword in combined:
            re_keywords_found.append(keyword)

    if re_keywords_found:
        score += min(2.0, len(re_keywords_found) * 0.3)
        reasons.append(f"re_context:{','.join(re_keywords_found[:5])}")

    # 7. Trusted source bonus
    domain = extract_domain(url)
    if domain in ALLOWED_SITES:
        score += 1.5
        reasons.append(f"trusted_source:{domain}")

    # 8. THEATRICAL PENALTY for Broadway addresses
    is_broadway = any("broadway" in addr.lower() for addr in all_addresses if addr)
    if is_broadway:
        theatrical_found = []
        for term in THEATRICAL_KEYWORDS:
            if term in combined:
                theatrical_found.append(term)

        if theatrical_found:
            penalty = min(5.0, len(theatrical_found) * 1.5)  # Stronger penalty
            score -= penalty
            reasons.append(f"theatrical_penalty:-{penalty}:{','.join(theatrical_found[:3])}")

    # 9. Other negative signals
    if not is_broadway:  # Only apply if not Broadway (to avoid double penalty)
        negative_terms = ["residential", "apartment", "condo", "hotel"]
        if not any(term in ["office", "commercial", "tenant", "lease"] for term in combined.split()):
            for neg in negative_terms:
                if neg in combined:
                    score -= 1.0
                    reasons.append(f"negative:{neg}")
                    break

    return round(score, 2), reasons

# -------------------- Enhanced Query Building --------------------
def make_queries_v6(building: Dict) -> List[str]:
    """Generate queries using primary and alternate addresses - ALL IN QUOTES WITH NYC"""
    queries = []
    seen_queries = set()

    # Collect all addresses
    all_addresses = []
    if building.get("primary_address"):
        all_addresses.append(building["primary_address"])

    # Add alternate addresses
    for i in range(20):  # Use top 20 alternate addresses for queries
        addr_key = f"alternate_address_{i}"
        if building.get(addr_key):
            addr = building[addr_key]
            if addr and addr not in all_addresses:
                all_addresses.append(addr)

    # Strategy 1: Primary name + NYC (always in quotes)
    if building.get("primary_name"):
        name = building["primary_name"]
        # Special handling for Broadway buildings
        if any("Broadway" in addr for addr in all_addresses if addr):
            query = f'"{name}" NYC Manhattan office -theater -musical -show -Broadway'
        else:
            query = f'"{name}" NYC Manhattan office'
        if query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    # Strategy 2: Alternate name if exists (always with NYC)
    if building.get("alternative_name"):
        alt_name = building["alternative_name"]
        query = f'"{alt_name}" NYC Manhattan building'
        if query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    # Strategy 3: Top addresses with NYC and quotes ALWAYS
    for addr in all_addresses[:3]:  # Use top 3 addresses
        if not addr:
            continue

        # Broadway special handling - strong theatrical exclusion
        if "Broadway" in addr:
            query = f'"{addr}" NYC Manhattan office -theater -musical -show -play -actress -actor'
        else:
            query = f'"{addr}" NYC Manhattan'

        if query not in seen_queries:
            queries.append(query)
            seen_queries.add(query)

    # Strategy 4: Address ranges if available (with NYC)
    for i in range(10):
        range_key = f"alternate_address_{i}_range"
        if building.get(range_key):
            addr_range = building[range_key]
            if addr_range:
                query = f'"{addr_range}" NYC Manhattan office'
                if query not in seen_queries:
                    queries.append(query)
                    seen_queries.add(query)
                break  # Use only first range

    # Strategy 5: Building name with street name (if different strategies haven't worked)
    if building.get("primary_name") and len(all_addresses) > 0:
        # Extract street name from address
        addr_parts = all_addresses[0].split()
        street_name = None
        for part in addr_parts[1:]:  # Skip number
            if part.lower() not in ['st', 'ave', 'pl', 'lane', 'way', 'street', 'avenue', 'place']:
                street_name = part
                break

        if street_name:
            # Combine building name with street for precision
            query = f'"{building["primary_name"]}" "{street_name}" NYC'
            if query not in seen_queries:
                queries.append(query)
                seen_queries.add(query)

    return queries[:5]  # Return top 5 unique queries

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

def fetch_for_building_v6(building: Dict) -> List[Dict]:
    """Fetch news using enhanced strategies with alternate addresses"""
    items = []
    seen_urls = set()

    queries = make_queries_v6(building)

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

                for entry in getattr(feed, "entries", [])[:15]:  # Process more entries
                    item = parse_entry(entry, building["id"])

                    # Skip if we've seen this URL
                    if item["url"] in seen_urls:
                        continue
                    seen_urls.add(item["url"])

                    # Score the match with v6 algorithm
                    score, reasons = score_match_v6(building, item["title"], item.get("summary", ""), item["url"])
                    item["score"] = score
                    item["reasons"] = reasons
                    item["search_query"] = query

                    items.append(item)

                # If we found good results, we might stop
                good_items = [it for it in items if it["score"] >= DEFAULT_MIN_SCORE]
                if len(good_items) >= 5:
                    break  # Have enough good results

            except Exception as e:
                logger.error(f"Feed error for {building['id']} query '{query}': {e}")
                continue

        # Small delay between queries
        if queries.index(query) < len(queries) - 1:
            time.sleep(0.1)

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
  search_query TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_items_building ON items(building_id);
CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_at);
CREATE INDEX IF NOT EXISTS idx_items_score ON items(score);
CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at);

CREATE TABLE IF NOT EXISTS search_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  building_id TEXT,
  search_time TEXT,
  items_found INTEGER,
  max_score REAL,
  success BOOLEAN
);
CREATE INDEX IF NOT EXISTS idx_log_building ON search_log(building_id);
CREATE INDEX IF NOT EXISTS idx_log_time ON search_log(search_time);
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

    def log_search(self, building_id: str, items_found: int, max_score: float, success: bool):
        """Log search attempt for monitoring"""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO search_log(building_id, search_time, items_found, max_score, success)
            VALUES(?, datetime('now'), ?, ?, ?)
        """, (building_id, items_found, max_score, success))
        self._conn.commit()

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

    def get_coverage_stats(self) -> Dict:
        """Get statistics on building coverage"""
        cur = self._conn.execute("""
            SELECT
                COUNT(DISTINCT building_id) as buildings_with_news,
                COUNT(*) as total_articles,
                AVG(score) as avg_score,
                MAX(score) as max_score,
                COUNT(CASE WHEN score >= ? THEN 1 END) as good_articles
            FROM items
            WHERE score >= ?
        """, (DEFAULT_MIN_SCORE, 0))
        row = cur.fetchone()
        return {
            "buildings_with_news": row[0],
            "total_articles": row[1],
            "avg_score": round(row[2] or 0, 2),
            "max_score": round(row[3] or 0, 2),
            "good_articles": row[4]
        }

# -------------------- Load Buildings with Alternate Addresses --------------------
def load_buildings_with_alternates(addresses_csv: str, clean_csv: str) -> List[Dict]:
    """Load buildings with all their alternate addresses"""
    # Load the clean CSV for basic info
    df_clean = pd.read_csv(clean_csv)

    # Load the full addresses CSV
    df_addresses = pd.read_csv(addresses_csv)

    # Merge on BBL
    df = pd.merge(
        df_clean[['bbl', 'main_address', 'primary_building_name']],
        df_addresses,
        on='bbl',
        how='left'
    )

    buildings = []

    for _, row in df.iterrows():
        # Get primary info
        primary_name = normalize_text(row.get("primary_building_name"))
        primary_address = normalize_text(row.get("main_address"))
        alternative_name = normalize_text(row.get("alternative_name_1"))

        if not (primary_name or primary_address):
            continue

        addr_clean = re.sub(r'[^a-zA-Z0-9]', '', primary_address or primary_name or "")
        b_id = f"bld-{addr_clean.lower()}"

        building = {
            "id": b_id,
            "primary_name": primary_name,
            "primary_address": primary_address,
            "alternative_name": alternative_name,
            "bbl": str(row.get("bbl", ""))
        }

        # Add all alternate addresses
        for i in range(100):  # Check up to 100 alternate address columns
            addr_col = f"alternate_address_{i}"
            if addr_col in row:
                addr_value = normalize_text(row.get(addr_col))
                if addr_value:
                    building[addr_col] = addr_value

            # Also check for address ranges
            range_col = f"alternate_address_{i}_range"
            if range_col in row:
                range_value = normalize_text(row.get(range_col))
                if range_value:
                    building[range_col] = range_value

        buildings.append(building)

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
        return jsonify({"ok": True, "version": "v6_optimized", "time": to_iso(now_utc())})

    @app.route("/api/stats")
    def stats():
        """Get coverage statistics"""
        stats = store.get_coverage_stats()
        stats["total_buildings"] = len(buildings)
        stats["coverage_percent"] = round(stats["buildings_with_news"] / len(buildings) * 100, 2)
        return jsonify(stats)

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
                new_items = fetch_for_building_v6(building)
                if new_items:
                    store.upsert_many(new_items)
                    # Log the search
                    max_score = max(item["score"] for item in new_items) if new_items else 0
                    store.log_search(building["id"], len(new_items), max_score, True)
                    items = store.list(building["id"], limit, min_score, max_age_days)
                else:
                    store.log_search(building["id"], 0, 0, False)
            except Exception as e:
                logger.error(f"Fetch error for {building_id}: {e}")
                store.log_search(building["id"], 0, 0, False)

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
    parser.add_argument("--addresses-csv",
                       default="/Users/forrestmiller/Desktop/New/data/all_building_addresses.csv",
                       help="Path to all_building_addresses.csv")
    parser.add_argument("--clean-csv",
                       default="/Users/forrestmiller/Desktop/-prospector-news-service/data/news_search_addresses_clean.csv",
                       help="Path to news_search_addresses_clean.csv")
    parser.add_argument("--db", default="news_v6.db", help="SQLite database path")
    parser.add_argument("--bind", default="0.0.0.0:8080", help="host:port")
    args = parser.parse_args()

    logger.info("Loading buildings with alternate addresses...")
    buildings = load_buildings_with_alternates(args.addresses_csv, args.clean_csv)
    logger.info(f"Loaded {len(buildings)} buildings (v6 optimized with alternates)")

    # Log sample building to show alternate addresses
    if buildings:
        sample = buildings[0]
        alt_count = sum(1 for k in sample.keys() if k.startswith("alternate_address_"))
        logger.info(f"Sample building has {alt_count} alternate addresses")

    app = build_service(buildings, args.db)

    host, port_str = args.bind.split(":")
    port = int(os.getenv("PORT", port_str))
    logger.info(f"Starting optimized news service v6 on {host}:{port}")
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()