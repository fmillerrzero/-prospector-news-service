#!/usr/bin/env python3
"""
Coverage Monitor for News Service v6
-------------------------------------
Tracks and reports coverage statistics, identifies gaps, and suggests improvements.
"""

import argparse
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import pandas as pd

from news_service_v6_optimized import (
    load_buildings_with_alternates,
    DEFAULT_MIN_SCORE,
    now_utc
)

class CoverageMonitor:
    """Monitor and analyze news coverage"""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)

    def get_overall_stats(self) -> Dict:
        """Get overall coverage statistics"""
        cur = self.conn.cursor()

        # Total articles and buildings
        cur.execute("""
            SELECT
                COUNT(DISTINCT building_id) as buildings_with_news,
                COUNT(*) as total_articles,
                AVG(score) as avg_score,
                MAX(score) as max_score,
                MIN(published_at) as oldest_article,
                MAX(published_at) as newest_article
            FROM items
            WHERE score >= ?
        """, (DEFAULT_MIN_SCORE,))

        row = cur.fetchone()
        stats = {
            "buildings_with_news": row[0] or 0,
            "total_articles": row[1] or 0,
            "avg_score": round(row[2] or 0, 2),
            "max_score": round(row[3] or 0, 2),
            "oldest_article": row[4],
            "newest_article": row[5]
        }

        # Score distribution
        cur.execute("""
            SELECT
                CASE
                    WHEN score >= 8 THEN 'Excellent (8+)'
                    WHEN score >= 6 THEN 'Good (6-8)'
                    WHEN score >= 4 THEN 'Fair (4-6)'
                    WHEN score >= 2.5 THEN 'Minimum (2.5-4)'
                    ELSE 'Below Threshold'
                END as score_range,
                COUNT(*) as count
            FROM items
            GROUP BY score_range
            ORDER BY MIN(score) DESC
        """)

        stats["score_distribution"] = {row[0]: row[1] for row in cur.fetchall()}

        # Articles by time period
        cutoffs = [
            (7, "Last Week"),
            (30, "Last Month"),
            (90, "Last 3 Months"),
            (180, "Last 6 Months")
        ]

        for days, label in cutoffs:
            cutoff = (now_utc() - timedelta(days=days)).isoformat()
            cur.execute("""
                SELECT COUNT(*) FROM items
                WHERE published_at >= ? AND score >= ?
            """, (cutoff, DEFAULT_MIN_SCORE))
            stats[f"articles_{label.lower().replace(' ', '_')}"] = cur.fetchone()[0]

        return stats

    def get_building_coverage(self, buildings: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Analyze which buildings have/don't have coverage"""
        cur = self.conn.cursor()

        # Get all buildings with news
        cur.execute("""
            SELECT
                building_id,
                COUNT(*) as article_count,
                MAX(score) as max_score,
                MAX(published_at) as latest_article
            FROM items
            WHERE score >= ?
            GROUP BY building_id
        """, (DEFAULT_MIN_SCORE,))

        covered = {}
        for row in cur.fetchall():
            covered[row[0]] = {
                "article_count": row[1],
                "max_score": row[2],
                "latest_article": row[3]
            }

        # Categorize buildings
        buildings_with_news = []
        buildings_without_news = []

        for building in buildings:
            b_id = building["id"]
            if b_id in covered:
                buildings_with_news.append({
                    "id": b_id,
                    "bbl": building.get("bbl"),
                    "name": building.get("primary_name"),
                    "address": building.get("primary_address"),
                    **covered[b_id]
                })
            else:
                buildings_without_news.append({
                    "id": b_id,
                    "bbl": building.get("bbl"),
                    "name": building.get("primary_name"),
                    "address": building.get("primary_address")
                })

        return buildings_with_news, buildings_without_news

    def get_search_activity(self) -> Dict:
        """Analyze search activity patterns"""
        cur = self.conn.cursor()

        # Searches by time period
        periods = [
            (1, "Last Hour"),
            (6, "Last 6 Hours"),
            (24, "Last 24 Hours"),
            (168, "Last Week")
        ]

        activity = {}
        for hours, label in periods:
            cutoff = (now_utc() - timedelta(hours=hours)).isoformat()
            cur.execute("""
                SELECT
                    COUNT(DISTINCT building_id) as buildings_searched,
                    COUNT(*) as total_searches,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_searches,
                    AVG(items_found) as avg_items_found
                FROM search_log
                WHERE search_time >= ?
            """, (cutoff,))

            row = cur.fetchone()
            activity[label] = {
                "buildings_searched": row[0] or 0,
                "total_searches": row[1] or 0,
                "successful_searches": row[2] or 0,
                "success_rate": round((row[2] or 0) / max(1, row[1]) * 100, 1),
                "avg_items_found": round(row[3] or 0, 1)
            }

        return activity

    def identify_high_value_gaps(self, buildings_without_news: List[Dict]) -> List[Dict]:
        """Identify high-value buildings without news (famous buildings, large buildings)"""
        high_value = []

        # Keywords that indicate important buildings
        important_keywords = [
            "plaza", "tower", "center", "building", "exchange",
            "one", "world", "park", "square", "hudson"
        ]

        for building in buildings_without_news:
            name = (building.get("name") or "").lower()
            address = (building.get("address") or "").lower()

            # Check if it's likely an important building
            is_important = False

            # Named buildings are usually important
            if building.get("name"):
                is_important = True

            # Check for important keywords
            for keyword in important_keywords:
                if keyword in name or keyword in address:
                    is_important = True
                    break

            # Check for prime locations
            prime_streets = ["wall st", "broadway", "park ave", "fifth ave", "madison ave"]
            for street in prime_streets:
                if street in address:
                    is_important = True
                    break

            if is_important:
                high_value.append(building)

        return high_value[:50]  # Top 50 gaps

    def generate_report(self, buildings: List[Dict]) -> str:
        """Generate comprehensive coverage report"""
        stats = self.get_overall_stats()
        with_news, without_news = self.get_building_coverage(buildings)
        activity = self.get_search_activity()
        high_value_gaps = self.identify_high_value_gaps(without_news)

        coverage_pct = len(with_news) / len(buildings) * 100

        report = f"""
================================================================================
NEWS SERVICE COVERAGE REPORT - {now_utc().strftime('%Y-%m-%d %H:%M:%S UTC')}
================================================================================

OVERALL COVERAGE
----------------
Total Buildings:        {len(buildings)}
Buildings with News:    {len(with_news)} ({coverage_pct:.1f}%)
Buildings without News: {len(without_news)} ({100-coverage_pct:.1f}%)

ARTICLE STATISTICS
------------------
Total Articles:         {stats['total_articles']}
Average Score:          {stats['avg_score']}
Maximum Score:          {stats['max_score']}

Articles Last Week:     {stats.get('articles_last_week', 0)}
Articles Last Month:    {stats.get('articles_last_month', 0)}
Articles Last 3 Months: {stats.get('articles_last_3_months', 0)}
Articles Last 6 Months: {stats.get('articles_last_6_months', 0)}

SCORE DISTRIBUTION
------------------"""

        for range_name, count in sorted(stats.get('score_distribution', {}).items(),
                                       key=lambda x: x[0], reverse=True):
            report += f"\n{range_name:20s}: {count:5d} articles"

        report += """

SEARCH ACTIVITY
---------------"""

        for period, data in activity.items():
            report += f"""
{period}:
  Buildings Searched: {data['buildings_searched']}
  Success Rate:       {data['success_rate']}%
  Avg Items Found:    {data['avg_items_found']}"""

        report += """

TOP COVERAGE GAPS (High-Value Buildings Without News)
------------------------------------------------------"""

        for i, building in enumerate(high_value_gaps[:20], 1):
            name = building.get('name') or 'Unnamed'
            address = building.get('address') or 'No address'
            report += f"\n{i:2d}. {name:40s} | {address}"

        report += """

RECOMMENDATIONS
---------------"""

        if coverage_pct < 50:
            report += "\n⚠️  Coverage below 50% - Run batch processor to improve"
        if stats.get('articles_last_week', 0) < 100:
            report += "\n⚠️  Low recent article count - Check if searches are running"
        if len(high_value_gaps) > 20:
            report += f"\n⚠️  {len(high_value_gaps)} high-value buildings lack coverage"

        report += "\n\n" + "=" * 80 + "\n"

        return report

    def export_gaps_csv(self, buildings: List[Dict], output_file: str):
        """Export buildings without news to CSV for analysis"""
        _, without_news = self.get_building_coverage(buildings)

        df = pd.DataFrame(without_news)
        df.to_csv(output_file, index=False)
        return len(without_news)


def main():
    parser = argparse.ArgumentParser(description="Monitor news service coverage")
    parser.add_argument("--db", default="news_v6.db", help="SQLite database path")
    parser.add_argument("--addresses-csv",
                       default="/Users/forrestmiller/Desktop/New/data/all_building_addresses.csv",
                       help="Path to all_building_addresses.csv")
    parser.add_argument("--clean-csv",
                       default="/Users/forrestmiller/Desktop/-prospector-news-service/data/news_search_addresses_clean.csv",
                       help="Path to news_search_addresses_clean.csv")
    parser.add_argument("--export-gaps", help="Export buildings without news to CSV")
    parser.add_argument("--brief", action="store_true", help="Show brief summary only")
    args = parser.parse_args()

    # Load buildings
    print("Loading buildings...")
    buildings = load_buildings_with_alternates(args.addresses_csv, args.clean_csv)
    print(f"Loaded {len(buildings)} buildings")

    # Initialize monitor
    monitor = CoverageMonitor(args.db)

    if args.brief:
        # Brief summary
        stats = monitor.get_overall_stats()
        with_news, without_news = monitor.get_building_coverage(buildings)
        coverage_pct = len(with_news) / len(buildings) * 100

        print(f"\nCoverage: {len(with_news)}/{len(buildings)} buildings ({coverage_pct:.1f}%)")
        print(f"Articles: {stats['total_articles']} (avg score: {stats['avg_score']})")
        print(f"Recent: {stats.get('articles_last_week', 0)} articles in last week")
    else:
        # Full report
        report = monitor.generate_report(buildings)
        print(report)

        # Save to file
        report_file = f"coverage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_file}")

    # Export gaps if requested
    if args.export_gaps:
        count = monitor.export_gaps_csv(buildings, args.export_gaps)
        print(f"Exported {count} buildings without news to {args.export_gaps}")


if __name__ == "__main__":
    main()