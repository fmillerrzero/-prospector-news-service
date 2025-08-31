#!/usr/bin/env python3
"""
Super simple news service - just returns mock data for now
"""
import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Simple News Service", "status": "running"}

@app.route("/api/news/<building_id>")
def get_news(building_id):
    # Return mock news data for any building
    return jsonify([
        {
            "uid": "mock1",
            "building_id": building_id,
            "title": "Major Office Lease Signed in Midtown Manhattan",
            "url": "https://therealdeal.com/example",
            "summary": "A major technology company has signed a 50,000 square foot lease in this prime Manhattan office building.",
            "source": "The Real Deal",
            "published_at": "2025-08-30T12:00:00Z",
            "score": 8.5
        },
        {
            "uid": "mock2", 
            "building_id": building_id,
            "title": "Class A Office Building Sees Strong Leasing Activity",
            "url": "https://commercialobserver.com/example",
            "summary": "The building continues to attract high-quality tenants in the current market.",
            "source": "Commercial Observer",
            "published_at": "2025-08-29T15:30:00Z", 
            "score": 7.8
        }
    ])

@app.route("/api/news")
def get_all_news():
    return jsonify([{"message": "Use /api/news/<building_id> for specific building news"}])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)