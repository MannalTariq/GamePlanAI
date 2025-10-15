#!/usr/bin/env python3
"""
Test script to verify that GNN Strategy Maker produces dynamic, non-hardcoded results.
Tests multiple formations and verifies that receiver selection and tactical decisions vary.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_maker import StrategyMaker
import json

def create_test_formation(formation_name: str, players_data: list):
    """Create a test formation with named configuration"""
    print(f"\n{'='*80}")
    print(f"  TEST FORMATION: {formation_name}")
    print(f"{'='*80}")
    
    # Initialize strategy maker
    strategy_maker = StrategyMaker()
    
    # Generate strategy
    strategy = strategy_maker.predict_strategy(players_data, corner_position=(105, 0))
    
    # Extract key results
    primary = strategy["predictions"]["primary_receiver"]
    shot_conf = strategy["predictions"]["shot_confidence"]
    decision = strategy["predictions"]["tactical_decision"]
    
    result = {
        "formation": formation_name,
        "primary_receiver_id": primary["player_id"],
        "primary_receiver_score": primary["score"],
        "shot_confidence": shot_conf,
        "tactical_decision": decision,
        "num_players": len(players_data),
        "timestamp": strategy["timestamp"]
    }
    
    print(f"\n🎯 FORMATION RESULTS:")
    print(f"   Formation: {formation_name}")
    print(f"   Primary Receiver: Player #{primary['player_id']} (Score: {primary['score']:.4f})")
    print(f"   Shot Confidence: {shot_conf:.4f}")
    print(f"   Tactical Decision: '{decision}'")
    
    return result

def main():
    """Run comprehensive dynamic behavior test"""
    print("🧪 TESTING GNN STRATEGY MAKER DYNAMIC BEHAVIOR")
    print("=" * 80)
    print("This test verifies that the strategy maker:")
    print("1. Does NOT always select the same receiver (e.g., Player #100005)")
    print("2. Does NOT always show 'LAYOFF' decision")
    print("3. Produces varied outputs based on player positions")
    print("=" * 80)
    
    test_results = []
    
    # Test Formation 1: Crowded Near Post
    formation1 = [
        {"id": 100001, "x": 94, "y": 32, "team": "attacker"},
        {"id": 100002, "x": 96, "y": 30, "team": "attacker"},
        {"id": 100003, "x": 95, "y": 34, "team": "attacker"},
        {"id": 100004, "x": 93, "y": 36, "team": "attacker"},
        {"id": 100005, "x": 97, "y": 28, "team": "attacker"},
        {"id": 100011, "x": 88, "y": 30, "team": "defender"},
        {"id": 100012, "x": 90, "y": 35, "team": "defender"},
        {"id": 100013, "x": 85, "y": 40, "team": "defender"},
        {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
    ]
    test_results.append(create_test_formation("Crowded Near Post", formation1))
    
    # Test Formation 2: Far Post Focus
    formation2 = [
        {"id": 100001, "x": 92, "y": 45, "team": "attacker"},
        {"id": 100002, "x": 94, "y": 42, "team": "attacker"},
        {"id": 100003, "x": 96, "y": 40, "team": "attacker"},
        {"id": 100004, "x": 90, "y": 48, "team": "attacker"},
        {"id": 100005, "x": 88, "y": 35, "team": "attacker"},
        {"id": 100011, "x": 86, "y": 42, "team": "defender"},
        {"id": 100012, "x": 89, "y": 38, "team": "defender"},
        {"id": 100013, "x": 91, "y": 30, "team": "defender"},
        {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
    ]
    test_results.append(create_test_formation("Far Post Focus", formation2))
    
    # Test Formation 3: Central Concentration
    formation3 = [
        {"id": 100001, "x": 93, "y": 34, "team": "attacker"},
        {"id": 100002, "x": 91, "y": 34, "team": "attacker"},
        {"id": 100003, "x": 95, "y": 34, "team": "attacker"},
        {"id": 100004, "x": 89, "y": 34, "team": "attacker"},
        {"id": 100005, "x": 97, "y": 34, "team": "attacker"},
        {"id": 100011, "x": 92, "y": 32, "team": "defender"},
        {"id": 100012, "x": 94, "y": 36, "team": "defender"},
        {"id": 100013, "x": 90, "y": 36, "team": "defender"},
        {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
    ]
    test_results.append(create_test_formation("Central Concentration", formation3))
    
    # Test Formation 4: Wide Spread
    formation4 = [
        {"id": 100001, "x": 95, "y": 15, "team": "attacker"},
        {"id": 100002, "x": 93, "y": 50, "team": "attacker"},
        {"id": 100003, "x": 90, "y": 25, "team": "attacker"},
        {"id": 100004, "x": 88, "y": 45, "team": "attacker"},
        {"id": 100005, "x": 85, "y": 35, "team": "attacker"},
        {"id": 100011, "x": 87, "y": 20, "team": "defender"},
        {"id": 100012, "x": 89, "y": 48, "team": "defender"},
        {"id": 100013, "x": 91, "y": 30, "team": "defender"},
        {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
    ]
    test_results.append(create_test_formation("Wide Spread", formation4))
    
    # Analyze results for dynamic behavior
    print(f"\n{'='*80}")
    print(f"  DYNAMIC BEHAVIOR ANALYSIS")
    print(f"{'='*80}")
    
    receivers = [r["primary_receiver_id"] for r in test_results]
    decisions = [r["tactical_decision"] for r in test_results]
    scores = [r["primary_receiver_score"] for r in test_results]
    shot_confs = [r["shot_confidence"] for r in test_results]
    
    print(f"\n🔍 DIVERSITY CHECK:")
    print(f"   Primary Receivers: {receivers}")
    print(f"   Unique Receivers: {len(set(receivers))} out of {len(receivers)} formations")
    print(f"   Receiver Variation: {'✅ DYNAMIC' if len(set(receivers)) > 1 else '❌ STATIC'}")
    
    print(f"\n🎯 DECISION ANALYSIS:")
    print(f"   Tactical Decisions:")
    for i, decision in enumerate(decisions):
        print(f"     Formation {i+1}: '{decision}'")
    print(f"   Unique Decisions: {len(set(decisions))} out of {len(decisions)} formations")
    print(f"   Decision Variation: {'✅ DYNAMIC' if len(set(decisions)) > 1 else '❌ STATIC'}")
    
    print(f"\n📊 SCORE ANALYSIS:")
    print(f"   Receiver Scores: {[f'{s:.3f}' for s in scores]}")
    print(f"   Score Range: {min(scores):.3f} to {max(scores):.3f}")
    print(f"   Score Variation: {'✅ GOOD SPREAD' if (max(scores) - min(scores)) > 0.05 else '⚠️ LOW VARIATION'}")
    
    print(f"\n⚽ SHOT CONFIDENCE:")
    print(f"   Shot Confidences: {[f'{c:.3f}' for c in shot_confs]}")
    print(f"   Confidence Range: {min(shot_confs):.3f} to {max(shot_confs):.3f}")
    
    # Overall assessment
    dynamic_receivers = len(set(receivers)) > 1
    dynamic_decisions = len(set(decisions)) > 1
    no_hardcoded_layoff = not any('layoff' in d.lower() for d in decisions)
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   ✅ No hardcoded Player #100005: {'PASS' if 100005 not in receivers or len(set(receivers)) > 1 else 'FAIL'}")
    print(f"   ✅ Dynamic receiver selection: {'PASS' if dynamic_receivers else 'FAIL'}")
    print(f"   ✅ Dynamic tactical decisions: {'PASS' if dynamic_decisions else 'FAIL'}")
    print(f"   ✅ No hardcoded 'LAYOFF': {'PASS' if no_hardcoded_layoff else 'FAIL'}")
    
    all_tests_pass = dynamic_receivers and dynamic_decisions and no_hardcoded_layoff
    print(f"\n🎉 FINAL RESULT: {'✅ ALL TESTS PASS - GNN IS WORKING DYNAMICALLY!' if all_tests_pass else '❌ SOME ISSUES DETECTED'}")
    
    # Save detailed results
    with open("test_dynamic_results.json", "w") as f:
        json.dump({
            "test_summary": {
                "dynamic_receivers": dynamic_receivers,
                "dynamic_decisions": dynamic_decisions,
                "no_hardcoded_layoff": no_hardcoded_layoff,
                "all_tests_pass": all_tests_pass
            },
            "formation_results": test_results
        }, f, indent=2)
    
    print(f"\n📝 Detailed results saved to: test_dynamic_results.json")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()