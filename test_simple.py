#!/usr/bin/env python3
import requests
import json

# Simple test data
data = {
    "players": [
        {"id": 1, "xPct": 80, "yPct": 30, "role": "red", "label": "Attacker1"},
        {"id": 2, "xPct": 85, "yPct": 40, "role": "red", "label": "Attacker2"}
    ],
    "cornerPosition": {"x": 95, "y": 5},
    "goalPosition": {"x": 95, "y": 50},
    "setPiece": "Corner Kick"
}

print("Testing API with simple data...")
response = requests.post("http://localhost:5000/api/simulate", json=data)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"Primary Player: {result['prediction']['primaryPlayer']}")
    print(f"Shot Confidence: {result['prediction']['shotConfidence']}%")
else:
    print(f"Error: {response.text}")
