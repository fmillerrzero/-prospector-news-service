#!/usr/bin/env python3
import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return {"message": "Minimal Flask app working!", "port": os.getenv("PORT", "unknown")}

@app.route("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    print(f"Starting minimal Flask app on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)