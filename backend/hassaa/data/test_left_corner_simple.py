#!/usr/bin/env python3
"""
Simple focused test: Left corner with nearby attackers SHOULD select a receiver
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_maker import StrategyMaker

def main():
    print("\n" + "="*70)
    print("🎯 FOCUSED TEST: Left Corner with Valid Nearby Attackers")
    print("="*70 + "\n")
    
    strategy_maker = StrategyMaker()
    
    # Left corner (0, 0) with attackers VERY CLOSE to left goal (0, 34)
    # These should definitely be selected!
    players = [
        {"id": 101, "x": 12, "y": 34, "team": "attacker"},  # 12m from left goal - PERFECT!
        {"id": 102, "x": 18, "y": 30, "team": "attacker"},  # Also very good
        {"id": 103, "x": 15, "y": 38, "team": "attacker"},  # Good position
        {"id": 201, "x": 50, "y": 34, "team": "defender"},  # Midfield
        {"id": 202, "x": 55, "y": 25, "team": "defender"},
        {"id": 301, "x": 3, "y": 34, "team": "keeper"},     # Left goal keeper
    ]
    
    corner_position = (0, 0)  # Bottom-left corner
    
    print(f"Setup:")
    print(f"  Corner: {corner_position}")
    print(f"  Target Goal: (0, 34) - LEFT GOAL")
    print(f"  Attackers:")
    for p in [p for p in players if p['team'] == 'attacker']:
        dist_to_left_goal = ((p['x'] - 0)**2 + (p['y'] - 34)**2)**0.5
        dist_to_corner = ((p['x'] - 0)**2 + (p['y'] - 0)**2)**0.5
        print(f"    Player #{p['id']}: ({p['x']}, {p['y']}) - {dist_to_left_goal:.1f}m from left goal, {dist_to_corner:.1f}m from corner")
    
    print(f"\n{'='*70}")
    print("Running strategy prediction...")
    print(f"{'='*70}\n")
    
    strategy = strategy_maker.predict_strategy(players, corner_position)
    
    primary = strategy["predictions"]["primary_receiver"]
    decision = strategy["predictions"]["tactical_decision"]
    
    print(f"\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Primary Receiver ID: {primary['player_id']}")
    print(f"Receiver Score: {primary['score']:.0%}")
    print(f"Tactical Decision: {decision}")
    
    if primary['position']:
        print(f"Receiver Position: ({primary['position']['x']}, {primary['position']['y']})")
        dist = ((primary['position']['x'] - 0)**2 + (primary['position']['y'] - 34)**2)**0.5
        print(f"Distance to left goal: {dist:.1f}m")
    
    print(f"\n" + "="*70)
    
    if primary['player_id'] is not None:
        print("✅ TEST PASSED - Receiver selected for left corner attack")
        print("✅ Players in left attacking third were properly recognized")
        return True
    else:
        print("❌ TEST FAILED - No receiver selected!")
        print("❌ This indicates the attacking zone detection is still not working")
        print(f"❌ Decision reason: {strategy.get('debug_info', {}).get('decision_reason', 'Unknown')}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
