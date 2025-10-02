#!/usr/bin/env python3
"""
Script to add missing BBLs from all_building_addresses.csv to news_search_addresses_clean.csv
"""
import pandas as pd
import os

# File paths
NEWS_SEARCH_FILE = "/Users/forrestmiller/Desktop/-prospector-news-service/data/news_search_addresses_clean.csv"
ALL_BUILDINGS_FILE = "/Users/forrestmiller/Desktop/New/data/all_building_addresses.csv"

# Read the existing news search addresses
print(f"Reading {NEWS_SEARCH_FILE}...")
news_df = pd.read_csv(NEWS_SEARCH_FILE)
print(f"  Found {len(news_df)} existing entries")

# Read all building addresses
print(f"\nReading {ALL_BUILDINGS_FILE}...")
all_df = pd.read_csv(ALL_BUILDINGS_FILE)
print(f"  Found {len(all_df)} total buildings")

# Find BBLs that are in all_df but not in news_df
existing_bbls = set(news_df['bbl'])
all_bbls = set(all_df['bbl'])
missing_bbls = all_bbls - existing_bbls

print(f"\nFound {len(missing_bbls)} BBLs to add")

# Extract the missing BBLs with their addresses and names
missing_df = all_df[all_df['bbl'].isin(missing_bbls)][['bbl', 'main_address', 'primary_building_name']]

# Fill NaN values with empty strings
missing_df = missing_df.fillna('')

print(f"  Extracted {len(missing_df)} new entries")

# Combine existing and new data
combined_df = pd.concat([news_df, missing_df], ignore_index=True)

# Sort by BBL for consistency
combined_df = combined_df.sort_values('bbl')

# Save the updated file
print(f"\nSaving updated file to {NEWS_SEARCH_FILE}...")
combined_df.to_csv(NEWS_SEARCH_FILE, index=False)
print(f"  Saved {len(combined_df)} total entries")

# Print summary
print("\n" + "="*50)
print("SUMMARY:")
print(f"  Original entries: {len(news_df)}")
print(f"  New entries added: {len(missing_df)}")
print(f"  Total entries now: {len(combined_df)}")
print("="*50)