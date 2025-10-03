#!/usr/bin/env python3
"""
Batch Processor for News Service v6
------------------------------------
Proactively searches all buildings in parallel to maintain fresh news coverage.
Designed to run as a scheduled job (cron/systemd timer).
"""

import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import signal
import sys

# Import functions from the main service
from news_service_v6_optimized import (
    load_buildings_with_alternates,
    fetch_for_building_v6,
    Store,
    now_utc,
    to_iso,
    DEFAULT_MIN_SCORE,
    DEFAULT_MAX_AGE_DAYS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class BatchProcessor:
    """Batch processor for searching all buildings"""

    def __init__(self, db_path: str, max_workers: int = 10, batch_size: int = 50):
        self.store = Store(db_path)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.stats = {
            "buildings_processed": 0,
            "buildings_with_news": 0,
            "total_articles": 0,
            "good_articles": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }

    def process_building(self, building: Dict) -> Tuple[str, int, float, bool]:
        """Process a single building and return results"""
        building_id = building["id"]

        try:
            # Check if we've searched this building recently (within 6 hours)
            recent_search = self._has_recent_search(building_id, hours=6)
            if recent_search and not shutdown_requested:
                logger.debug(f"Skipping {building_id} - searched recently")
                return building_id, 0, 0, False

            logger.info(f"Searching for news: {building_id} ({building.get('primary_name', 'Unknown')})")

            # Fetch news for this building
            items = fetch_for_building_v6(building)

            if items:
                # Filter to good articles
                good_items = [item for item in items if item["score"] >= DEFAULT_MIN_SCORE]

                # Store results
                if good_items:
                    self.store.upsert_many(good_items)
                    max_score = max(item["score"] for item in good_items)
                    logger.info(f"Found {len(good_items)} articles for {building_id} (max score: {max_score})")
                    self.store.log_search(building_id, len(good_items), max_score, True)
                    return building_id, len(good_items), max_score, True
                else:
                    logger.debug(f"No good articles for {building_id}")
                    self.store.log_search(building_id, 0, 0, False)
                    return building_id, 0, 0, False
            else:
                logger.debug(f"No articles found for {building_id}")
                self.store.log_search(building_id, 0, 0, False)
                return building_id, 0, 0, False

        except Exception as e:
            logger.error(f"Error processing {building_id}: {e}")
            self.store.log_search(building_id, 0, 0, False)
            return building_id, 0, 0, False

    def _has_recent_search(self, building_id: str, hours: int) -> bool:
        """Check if building was searched recently"""
        try:
            conn = self.store._conn
            cutoff = (now_utc() - timedelta(hours=hours)).isoformat()
            cur = conn.execute("""
                SELECT COUNT(*) FROM search_log
                WHERE building_id = ? AND search_time > ? AND success = 1
            """, (building_id, cutoff))
            count = cur.fetchone()[0]
            return count > 0
        except:
            return False

    def process_batch(self, buildings: List[Dict], resume_from: int = 0) -> Dict:
        """Process a batch of buildings in parallel"""
        self.stats["start_time"] = now_utc()
        buildings_to_process = buildings[resume_from:]
        total = len(buildings_to_process)

        logger.info(f"Starting batch processing of {total} buildings (starting from index {resume_from})")
        logger.info(f"Using {self.max_workers} workers, batch size {self.batch_size}")

        # Process in smaller chunks to show progress
        for chunk_start in range(0, total, self.batch_size):
            if shutdown_requested:
                logger.info("Shutdown requested, stopping batch processing")
                break

            chunk_end = min(chunk_start + self.batch_size, total)
            chunk = buildings_to_process[chunk_start:chunk_end]

            logger.info(f"Processing chunk {chunk_start//self.batch_size + 1} "
                       f"(buildings {resume_from + chunk_start + 1}-{resume_from + chunk_end} of {len(buildings)})")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_building = {
                    executor.submit(self.process_building, building): building
                    for building in chunk
                }

                # Process results as they complete
                for future in as_completed(future_to_building):
                    if shutdown_requested:
                        executor.shutdown(wait=False)
                        break

                    building = future_to_building[future]
                    try:
                        building_id, articles_found, max_score, success = future.result(timeout=30)
                        self.stats["buildings_processed"] += 1

                        if success:
                            self.stats["buildings_with_news"] += 1
                            self.stats["total_articles"] += articles_found
                            if articles_found > 0:
                                self.stats["good_articles"] += articles_found

                        # Log progress every 10 buildings
                        if self.stats["buildings_processed"] % 10 == 0:
                            self._log_progress()

                    except Exception as e:
                        self.stats["errors"] += 1
                        logger.error(f"Error processing building: {e}")

            # Brief pause between chunks to avoid overwhelming services
            if chunk_end < total and not shutdown_requested:
                time.sleep(2)

        self.stats["end_time"] = now_utc()
        return self.stats

    def _log_progress(self):
        """Log current progress"""
        processed = self.stats["buildings_processed"]
        with_news = self.stats["buildings_with_news"]
        coverage = (with_news / processed * 100) if processed > 0 else 0

        logger.info(f"Progress: {processed} buildings processed, "
                   f"{with_news} with news ({coverage:.1f}% coverage), "
                   f"{self.stats['total_articles']} articles found, "
                   f"{self.stats['errors']} errors")

    def get_final_report(self) -> str:
        """Generate final report"""
        if not self.stats["start_time"]:
            return "No processing performed"

        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        coverage = (self.stats["buildings_with_news"] / self.stats["buildings_processed"] * 100) \
                  if self.stats["buildings_processed"] > 0 else 0

        report = f"""
========================================
Batch Processing Report
========================================
Start Time:          {to_iso(self.stats['start_time'])}
End Time:            {to_iso(self.stats['end_time'])}
Duration:            {duration:.1f} seconds

Buildings Processed: {self.stats['buildings_processed']}
Buildings with News: {self.stats['buildings_with_news']}
Coverage Rate:       {coverage:.1f}%

Total Articles:      {self.stats['total_articles']}
Good Articles:       {self.stats['good_articles']}
Average per Building:{self.stats['total_articles'] / max(1, self.stats['buildings_with_news']):.1f}

Errors:              {self.stats['errors']}
Processing Rate:     {self.stats['buildings_processed'] / max(1, duration):.1f} buildings/second
========================================
"""
        return report

    def get_coverage_summary(self) -> Dict:
        """Get current database coverage summary"""
        try:
            conn = self.store._conn

            # Total unique buildings with news
            cur = conn.execute("""
                SELECT COUNT(DISTINCT building_id) FROM items WHERE score >= ?
            """, (DEFAULT_MIN_SCORE,))
            buildings_with_news = cur.fetchone()[0]

            # Articles from last 6 months
            cutoff = (now_utc() - timedelta(days=180)).isoformat()
            cur = conn.execute("""
                SELECT COUNT(*) FROM items
                WHERE published_at >= ? AND score >= ?
            """, (cutoff, DEFAULT_MIN_SCORE))
            recent_articles = cur.fetchone()[0]

            # Buildings searched in last 24 hours
            search_cutoff = (now_utc() - timedelta(hours=24)).isoformat()
            cur = conn.execute("""
                SELECT COUNT(DISTINCT building_id) FROM search_log
                WHERE search_time >= ?
            """, (search_cutoff,))
            recent_searches = cur.fetchone()[0]

            return {
                "buildings_with_news": buildings_with_news,
                "recent_articles": recent_articles,
                "recent_searches": recent_searches
            }
        except Exception as e:
            logger.error(f"Error getting coverage summary: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description="Batch processor for news service")
    parser.add_argument("--addresses-csv",
                       default="/Users/forrestmiller/Desktop/New/data/all_building_addresses.csv",
                       help="Path to all_building_addresses.csv")
    parser.add_argument("--clean-csv",
                       default="/Users/forrestmiller/Desktop/-prospector-news-service/data/news_search_addresses_clean.csv",
                       help="Path to news_search_addresses_clean.csv")
    parser.add_argument("--db", default="news_v6.db", help="SQLite database path")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from building index")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without processing")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("NEWS SERVICE BATCH PROCESSOR v6")
    logger.info("=" * 50)

    # Load buildings
    logger.info("Loading buildings with alternate addresses...")
    buildings = load_buildings_with_alternates(args.addresses_csv, args.clean_csv)
    logger.info(f"Loaded {len(buildings)} buildings")

    # Initialize processor
    processor = BatchProcessor(
        db_path=args.db,
        max_workers=args.workers,
        batch_size=args.batch_size
    )

    # Show current coverage
    coverage = processor.get_coverage_summary()
    logger.info(f"Current coverage: {coverage.get('buildings_with_news', 0)} buildings have news")
    logger.info(f"Recent activity: {coverage.get('recent_searches', 0)} buildings searched in last 24h")
    logger.info(f"Recent articles: {coverage.get('recent_articles', 0)} articles from last 6 months")

    if args.dry_run:
        logger.info("Dry run mode - exiting without processing")
        sys.exit(0)

    # Process buildings
    try:
        stats = processor.process_batch(buildings, resume_from=args.resume_from)

        # Print final report
        print(processor.get_final_report())

        # Save report to file
        report_file = f"batch_report_{to_iso(now_utc()).replace(':', '-')}.txt"
        with open(report_file, 'w') as f:
            f.write(processor.get_final_report())
        logger.info(f"Report saved to {report_file}")

        # Exit with appropriate code
        if stats["errors"] > stats["buildings_processed"] * 0.1:  # More than 10% errors
            logger.error("High error rate detected")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error during batch processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()