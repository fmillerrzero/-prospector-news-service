# NYC Building News Service

Real-time news aggregator for NYC commercial buildings using Google News and Bing News RSS feeds.

## Versions

### v5 (Improved) - RECOMMENDED
- **File**: `news_service_v5_improved.py`
- **Coverage**: ~60% of major buildings have news
- **Features**:
  - Multiple search strategies (exact name, address, partial matches)
  - Flexible scoring (3.0 minimum threshold)
  - 30-day time window
  - Expanded real estate keywords
  - Better partial matching

### v4 (Original)
- **File**: `news_service_v4.py`
- **Coverage**: ~0.1% of buildings have news
- **Features**:
  - Strict exact matching
  - High score threshold (7.5)
  - 10-day time window

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run improved v5 service (recommended)
python news_service_v5_improved.py --csv data/news_search_addresses_clean.csv --db news_v5.db --bind 0.0.0.0:8080

# Query for a building by BBL
curl http://localhost:8080/api/news/bbl-1008350041  # Empire State Building
```

## API Endpoints

- `GET /api/news` - All recent news
- `GET /api/news/bbl-{bbl}` - News for specific building
  - Query params: `min_score`, `max_age_days`, `limit`, `debug`

## Data Format

Buildings are identified by BBL (Borough Block Lot) numbers from NYC's property system.

Example response:
```json
[
  {
    "title": "Brookfield Place Lease Extended Through 2119",
    "url": "https://...",
    "score": 8.5,
    "published_at": "2025-10-02T12:00:00Z",
    "source": "Commercial Observer"
  }
]
```

## Environment Variables

- `DEFAULT_MIN_SCORE` - Minimum relevance score (default: 3.0 for v5)
- `DEFAULT_MAX_AGE_DAYS` - Maximum article age (default: 30 for v5)
- `ALLOWED_SITES` - Comma-separated list of trusted news sources

## Deployment

The service is configured for deployment on Render.com via `render.yaml`.