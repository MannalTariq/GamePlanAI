#!/usr/bin/env python3
"""
Test Anti-Spam & Dynamic Receiver Selection
Verifies that different formations produce different receivers and the system 
doesn't get stuck on a single hardcoded receiver.
"""

import os
import sys
import time

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_maker import StrategyMaker

def test_dynamic_receiver_selection():
    """Test that different formations produce different receivers"""
    
    print("="*70)
    print("  TESTING DYNAMIC RECEIVER SELECTION")
    print("="*70)
    
    strategy_maker = StrategyMaker()
    
    # Test different formations
    formations = {
        "Left Wing Focus": [
            {"id": 1001, "x": 85, "y": 20, "team": "attacker"},
            {"id": 1002, "x": 90, "y": 30, "team": "attacker"},
            {"id": 1003, "x": 95, "y": 35, "team": "attacker"},
            {"id": 2001, "x": 88, "y": 25, "team": "defender"},
            {"id": 3001, "x": 100, "y": 34, "team": "keeper"},
        ],
        
        "Right Wing Focus": [
            {"id": 1011, "x": 85, "y": 48, "team": "attacker"},
            {"id": 1012, "x": 90, "y": 38, "team": "attacker"},
            {"id": 1013, "x": 95, "y": 33, "team": "attacker"},
            {"id": 2011, "x": 88, "y": 43, "team": "defender"},
            {"id": 3011, "x": 100, "y": 34, "team": "keeper"},
        ],
        
        "Central Cluster": [
            {"id": 1021, "x": 92, "y": 32, "team": "attacker"},
            {"id": 1022, "x": 94, "y": 34, "team": "attacker"},
            {"id": 1023, "x": 90, "y": 36, "team": "attacker"},
            {"id": 2021, "x": 91, "y": 33, "team": "defender"},
            {"id": 3021, "x": 100, "y": 34, "team": "keeper"},
        ]
    }
    
    strategies = {}
    
    print(f"\n🧪 Testing {len(formations)} different formations:")
    
    for formation_name, players in formations.items():
        print(f"\n--- Testing: {formation_name} ---")
        
        # Show formation
        attackers = [p for p in players if p['team'] == 'attacker']
        attacker_info = [f"#{p['id']} at ({p['x']}, {p['y']})" for p in attackers]
        print(f"Attackers: {attacker_info}")
        
        # Generate strategy
        strategy = strategy_maker.predict_strategy(players)
        strategies[formation_name] = strategy
        
        # Extract key results
        primary = strategy["predictions"]["primary_receiver"]
        decision = strategy["predictions"]["tactical_decision"]
        
        print(f"✅ Result: Player #{primary['player_id']} - '{decision}'")
        
        # Small delay to ensure different timestamps
        time.sleep(0.1)
    
    # Analysis
    print(f"\n" + "="*70)
    print("  ANALYSIS RESULTS")
    print("="*70)
    
    receivers = [s["predictions"]["primary_receiver"]["player_id"] for s in strategies.values()]
    decisions = [s["predictions"]["tactical_decision"] for s in strategies.values()]
    
    print(f"\n📊 RECEIVER SELECTION:")
    for formation_name, strategy in strategies.items():
        primary = strategy["predictions"]["primary_receiver"]
        print(f"   {formation_name:20}: Player #{primary['player_id']} (Score: {primary['score']:.3f})")
    
    print(f"\n📋 TACTICAL DECISIONS:")
    for formation_name, strategy in strategies.items():
        decision = strategy["predictions"]["tactical_decision"]
        print(f"   {formation_name:20}: {decision}")
    
    # Check for variety
    unique_receivers = len(set(receivers))
    unique_decisions = len(set(decisions))
    
    print(f"\n🔍 VARIETY CHECK:")
    print(f"   Unique receivers: {unique_receivers}/{len(formations)} formations")
    print(f"   Unique decisions: {unique_decisions}/{len(formations)} formations")
    print(f"   All receivers: {receivers}")
    print(f"   All decisions: {decisions}")
    
    # Verdict
    if unique_receivers >= 2:
        print(f"\n✅ SUCCESS: System produces different receivers for different formations!")
        print(f"   No hardcoded receiver detected.")
    else:
        print(f"\n⚠️  WARNING: Same receiver ({receivers[0]}) selected for all formations.")
        print(f"   Possible hardcoded receiver or insufficient formation variation.")
    
    if unique_decisions >= 2:
        print(f"✅ SUCCESS: Dynamic tactical decisions working!")
    else:
        print(f"⚠️  WARNING: Same decision ('{decisions[0]}') for all formations.")
        print(f"   Check tactical decision logic thresholds.")
    
    # Test anti-spam
    print(f"\n🚫 TESTING ANTI-SPAM:")
    print(f"   Running same formation twice...")
    
    strategy1 = strategy_maker.predict_strategy(formations["Central Cluster"])
    time.sleep(0.1)
    strategy2 = strategy_maker.predict_strategy(formations["Central Cluster"])
    
    receiver1 = strategy1["predictions"]["primary_receiver"]["player_id"]
    receiver2 = strategy2["predictions"]["primary_receiver"]["player_id"]
    decision1 = strategy1["predictions"]["tactical_decision"]
    decision2 = strategy2["predictions"]["tactical_decision"]
    
    if receiver1 == receiver2 and decision1 == decision2:
        print(f"   ✅ Consistent: Same formation produces same result")
        print(f"   Receiver: #{receiver1}, Decision: '{decision1}'")
    else:
        print(f"   ⚠️  Inconsistent: Same formation produced different results")
        print(f"   Run 1: #{receiver1} - '{decision1}'")
        print(f"   Run 2: #{receiver2} - '{decision2}'")
    
    print(f"\n" + "="*70)
    
    return {
        "formations_tested": len(formations),
        "unique_receivers": unique_receivers,
        "unique_decisions": unique_decisions,
        "receivers": receivers,
        "decisions": decisions,
        "passes_variety_check": unique_receivers >= 2 and unique_decisions >= 2
    }

if __name__ == "__main__":
    result = test_dynamic_receiver_selection()
    
    if result["passes_variety_check"]:
        print("🎉 ALL TESTS PASSED: Dynamic receiver selection working correctly!")
    else:
        print("❌ TESTS FAILED: System may have hardcoded receivers or decisions.")