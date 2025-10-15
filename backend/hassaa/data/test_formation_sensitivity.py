#!/usr/bin/env python3
"""
Test GNN Strategy Sensitivity - Validate Dynamic Model Outputs
Tests that different player formations produce different GNN predictions.
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_maker import StrategyMaker
import json

def test_formation_sensitivity():
    """Test that different formations produce different strategies"""
    
    print("="*70)
    print("  GNN STRATEGY SENSITIVITY TEST")
    print("="*70)
    
    # Initialize strategy maker
    strategy_maker = StrategyMaker()
    
    # Test Scenario 1: Compact formation (players close together)
    print("\n🏗️  TEST 1: COMPACT FORMATION")
    compact_formation = [
        # Compact attacking formation
        {"id": 1, "x": 92, "y": 32, "team": "attacker"},
        {"id": 2, "x": 94, "y": 34, "team": "attacker"},
        {"id": 3, "x": 90, "y": 36, "team": "attacker"},
        {"id": 4, "x": 88, "y": 34, "team": "attacker"},
        
        # Compact defending
        {"id": 5, "x": 91, "y": 30, "team": "defender"},
        {"id": 6, "x": 93, "y": 38, "team": "defender"},
        {"id": 7, "x": 89, "y": 32, "team": "defender"},
        
        # Goalkeeper
        {"id": 8, "x": 100, "y": 34, "team": "keeper"},
    ]
    
    strategy1 = strategy_maker.predict_strategy(compact_formation)
    
    # Test Scenario 2: Spread formation (players spread out)
    print("\n🏗️  TEST 2: SPREAD FORMATION")
    spread_formation = [
        # Spread attacking formation
        {"id": 11, "x": 85, "y": 25, "team": "attacker"},
        {"id": 12, "x": 95, "y": 45, "team": "attacker"},
        {"id": 13, "x": 80, "y": 35, "team": "attacker"},
        {"id": 14, "x": 98, "y": 30, "team": "attacker"},
        
        # Spread defending
        {"id": 15, "x": 82, "y": 20, "team": "defender"},
        {"id": 16, "x": 90, "y": 50, "team": "defender"},
        {"id": 17, "x": 75, "y": 40, "team": "defender"},
        
        # Goalkeeper
        {"id": 18, "x": 100, "y": 34, "team": "keeper"},
    ]
    
    strategy2 = strategy_maker.predict_strategy(spread_formation)
    
    # Test Scenario 3: Near post focus
    print("\n🏗️  TEST 3: NEAR POST FORMATION")
    near_post_formation = [
        # Near post attacking formation
        {"id": 21, "x": 96, "y": 30, "team": "attacker"},
        {"id": 22, "x": 98, "y": 32, "team": "attacker"},
        {"id": 23, "x": 94, "y": 28, "team": "attacker"},
        {"id": 24, "x": 92, "y": 31, "team": "attacker"},
        
        # Defending near post
        {"id": 25, "x": 95, "y": 29, "team": "defender"},
        {"id": 26, "x": 97, "y": 33, "team": "defender"},
        {"id": 27, "x": 93, "y": 27, "team": "defender"},
        
        # Goalkeeper
        {"id": 28, "x": 100, "y": 34, "team": "keeper"},
    ]
    
    strategy3 = strategy_maker.predict_strategy(near_post_formation)
    
    # Analyze differences
    print("\n" + "="*70)
    print("  COMPARISON RESULTS")
    print("="*70)
    
    strategies = [
        ("Compact", strategy1),
        ("Spread", strategy2),
        ("Near Post", strategy3)
    ]
    
    print(f"\n📊 RECEIVER PREDICTIONS:")
    for name, strategy in strategies:
        primary = strategy["predictions"]["primary_receiver"]
        print(f"   {name:12}: Player #{primary['player_id']} ({primary['score']:.4f})")
        
    print(f"\n🎯 SHOT CONFIDENCE:")
    for name, strategy in strategies:
        conf = strategy["predictions"]["shot_confidence"]
        print(f"   {name:12}: {conf:.4f}")
        
    print(f"\n📋 TACTICAL DECISIONS:")
    for name, strategy in strategies:
        decision = strategy["predictions"]["tactical_decision"]
        print(f"   {name:12}: {decision}")
        
    # Check for variability
    receiver_scores = [s[1]["predictions"]["primary_receiver"]["score"] for s in strategies]
    shot_confidences = [s[1]["predictions"]["shot_confidence"] for s in strategies]
    decisions = [s[1]["predictions"]["tactical_decision"] for s in strategies]
    
    print(f"\n🔍 VARIABILITY ANALYSIS:")
    print(f"   Receiver score range: {min(receiver_scores):.4f} - {max(receiver_scores):.4f}")
    print(f"   Shot confidence range: {min(shot_confidences):.4f} - {max(shot_confidences):.4f}")
    print(f"   Unique decisions: {len(set(decisions))}/3")
    
    # Verdict
    receiver_varied = (max(receiver_scores) - min(receiver_scores)) > 0.01
    shot_varied = (max(shot_confidences) - min(shot_confidences)) > 0.01
    decisions_varied = len(set(decisions)) > 1
    
    print(f"\n🏆 TEST RESULTS:")
    print(f"   ✅ Receiver scores vary: {receiver_varied}")
    print(f"   ✅ Shot confidence varies: {shot_varied}")
    print(f"   ✅ Tactical decisions vary: {decisions_varied}")
    
    if receiver_varied and shot_varied:
        print(f"\n🎉 SUCCESS: GNN models are responding dynamically to formation changes!")
        print(f"   The strategy maker is working correctly.")
    else:
        print(f"\n⚠️  WARNING: Models may be producing static outputs.")
        print(f"   Check graph construction and model loading.")
        
    # Save comparison report
    comparison_report = {
        "test_timestamp": strategy1["timestamp"],
        "scenarios": {
            "compact": {
                "formation": compact_formation,
                "strategy": strategy1
            },
            "spread": {
                "formation": spread_formation,
                "strategy": strategy2
            },
            "near_post": {
                "formation": near_post_formation,
                "strategy": strategy3
            }
        },
        "analysis": {
            "receiver_score_range": [min(receiver_scores), max(receiver_scores)],
            "shot_confidence_range": [min(shot_confidences), max(shot_confidences)],
            "unique_decisions": list(set(decisions)),
            "models_responsive": receiver_varied and shot_varied
        }
    }
    
    with open("formation_sensitivity_test.json", "w") as f:
        json.dump(comparison_report, f, indent=2)
        
    print(f"\n📁 Detailed comparison saved to: formation_sensitivity_test.json")
    print(f"="*70)

if __name__ == "__main__":
    test_formation_sensitivity()