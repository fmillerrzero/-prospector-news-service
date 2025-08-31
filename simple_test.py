#!/usr/bin/env python3
import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from Render!"

@app.route("/test")  
def test():
    return {"message": "News service is working", "port": os.getenv("PORT", "unknown")}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Starting on port {port}")
    app.run(host="0.0.0.0", port=port)