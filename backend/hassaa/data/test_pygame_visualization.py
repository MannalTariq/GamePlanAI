#!/usr/bin/env python3
"""
Test script for the Pygame corner kick visualization
"""

import json
from corner_visualization_pygame import CornerKickVisualization

def create_test_strategy():
    """Create a test strategy for visualization"""
    return {
        "corner_id": 1,
        "timestamp": "2025-10-08T22:00:00Z",
        "best_strategy": "Far Post - Target #244553",
        "confidence": 0.81,
        "corner_flag": {"x": 105, "y": 0},
        "primary": {
            "id": 244553,
            "start": [90, 36],
            "target": [94, 34]
        },
        "alternates": [
            {"id": 232293, "position": {"x": 88, "y": 30}, "start": [88, 30], "target": [92, 30]},
            {"id": 221992, "position": {"x": 86, "y": 42}, "start": [86, 42], "target": [90, 36]}
        ],
        "players": [
            {"id": 244553, "team": "attacker", "intent": "primary_target", "position": {"x": 90, "y": 36}, "start": [90, 36], "target": [94, 34]},
            {"id": 232293, "team": "attacker", "intent": "support_runner", "position": {"x": 88, "y": 30}, "start": [88, 30], "target": [92, 30]},
            {"id": 221992, "team": "attacker", "intent": "support_runner", "position": {"x": 86, "y": 42}, "start": [86, 42], "target": [90, 36]},
            {"id": 900001, "team": "defender", "intent": "marking", "position": {"x": 96, "y": 32}, "start": [96, 32], "target": [93, 33]},
            {"id": 900002, "team": "defender", "intent": "shift", "position": {"x": 97, "y": 38}, "start": [97, 38], "target": [94, 36]},
            {"id": 900003, "team": "defender", "intent": "edge_box", "position": {"x": 93, "y": 44}, "start": [93, 44], "target": [90, 42]},
            {"id": 100000, "team": "keeper", "intent": "goalkeeping", "position": {"x": 105, "y": 34}, "start": [105, 34], "target": [103, 34]},
            {"id": 888888, "team": "attacker", "intent": "corner_taker", "position": {"x": 103, "y": 6}, "start": [103, 6], "target": [99, 6]}
        ],
        "cluster_zone": [93, 34],
        "tactical_setup": {
            "total_players": 8,
            "attackers": 4,
            "defenders": 3,
            "keeper": 1
        }
    }

def main():
    """Test the Pygame visualization"""
    print("Testing Pygame corner kick visualization...")
    
    # Create test strategy
    strategy = create_test_strategy()
    
    # Save to file for reference
    with open("test_strategy.json", "w") as f:
        json.dump(strategy, f, indent=2)
    print("Test strategy saved to test_strategy.json")
    
    # Create and run visualization
    print("Starting Pygame visualization...")
    print("Controls: F - Cycle Camera, R - Reset, ESC - Quit, +/- - Zoom")
    
    try:
        visualization = CornerKickVisualization(strategy)
        visualization.run()
        print("Pygame visualization completed successfully!")
    except Exception as e:
        print(f"Error running Pygame visualization: {e}")
        return False
        
    return True

if __name__ == "__main__":
    main()