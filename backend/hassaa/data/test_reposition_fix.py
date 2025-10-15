#!/usr/bin/env python3
"""
Test script to verify the "reposition" scenario fix
Tests both cases:
1. Valid receiver selection (normal)
2. No valid receiver - reposition scenario (fixed)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_maker import StrategyMaker

def test_left_corner_with_valid_receivers():
    """Test left corner with players in valid attacking positions"""
    print("="*70)
    print("TEST 1: Left Corner with Valid Receivers")
    print("="*70)
    
    strategy_maker = StrategyMaker()
    
    # Left corner setup with attackers in LEFT attacking third (x <= 35)
    players = [
        {"id": 100001, "x": 18, "y": 34, "team": "attacker"},  # Central, close to left goal
        {"id": 100002, "x": 25, "y": 25, "team": "attacker"},  # Good attacking position
        {"id": 100003, "x": 30, "y": 43, "team": "attacker"},  # Wing position
        {"id": 200001, "x": 45, "y": 34, "team": "defender"},  # Midfield defender
        {"id": 200002, "x": 40, "y": 20, "team": "defender"},
        {"id": 300001, "x": 8, "y": 34, "team": "keeper"},     # Left goal keeper
    ]
    
    corner_position = (0, 0)  # Bottom-left corner
    
    strategy = strategy_maker.predict_strategy(players, corner_position)
    
    primary = strategy["predictions"]["primary_receiver"]
    decision = strategy["predictions"]["tactical_decision"]
    
    print(f"\n✅ RESULTS:")
    print(f"   Primary Receiver: {primary['player_id']}")
    print(f"   Receiver Score: {primary['score']:.0%}")
    print(f"   Tactical Decision: {decision}")
    
    if primary['player_id'] is not None:
        print(f"\n✅ TEST PASSED - Valid receiver selected for left corner")
        return True
    else:
        print(f"\n❌ TEST FAILED - No receiver selected even though players are in position")
        return False

def test_left_corner_no_valid_receivers():
    """Test left corner with NO players in valid attacking positions"""
    print("\n" + "="*70)
    print("TEST 2: Left Corner with NO Valid Receivers (Reposition Scenario)")
    print("="*70)
    
    strategy_maker = StrategyMaker()
    
    # Left corner but attackers are in WRONG zone (x >= 70, right side)
    players = [
        {"id": 100001, "x": 85, "y": 34, "team": "attacker"},  # Right side - WRONG for left corner
        {"id": 100002, "x": 90, "y": 25, "team": "attacker"},  # Right side - WRONG
        {"id": 100003, "x": 95, "y": 43, "team": "attacker"},  # Right side - WRONG
        {"id": 200001, "x": 55, "y": 34, "team": "defender"},
        {"id": 200002, "x": 60, "y": 20, "team": "defender"},
        {"id": 300001, "x": 8, "y": 34, "team": "keeper"},     # Left goal keeper
    ]
    
    corner_position = (0, 0)  # Bottom-left corner
    
    strategy = strategy_maker.predict_strategy(players, corner_position)
    
    primary = strategy["predictions"]["primary_receiver"]
    decision = strategy["predictions"]["tactical_decision"]
    
    print(f"\n✅ RESULTS:")
    print(f"   Primary Receiver: {primary['player_id']}")
    print(f"   Receiver Score: {primary['score']:.0%}")
    print(f"   Tactical Decision: {decision}")
    print(f"   Decision Reason: {strategy.get('debug_info', {}).get('decision_reason', 'N/A')}")
    
    # In this case, primary_receiver_id should be None (reposition)
    if primary['player_id'] is None and "reposition" in decision.lower():
        print(f"\n✅ TEST PASSED - System correctly identified no valid receivers and chose reposition")
        return True
    else:
        print(f"\n⚠️ WARNING - Expected reposition decision, got: {decision}")
        return False

def test_right_corner_with_valid_receivers():
    """Test right corner with players in valid attacking positions"""
    print("\n" + "="*70)
    print("TEST 3: Right Corner with Valid Receivers")
    print("="*70)
    
    strategy_maker = StrategyMaker()
    
    # Right corner setup with attackers in RIGHT attacking third (x >= 70)
    players = [
        {"id": 100001, "x": 95, "y": 34, "team": "attacker"},  # Close to right goal
        {"id": 100002, "x": 88, "y": 25, "team": "attacker"},  # Good position
        {"id": 100003, "x": 92, "y": 43, "team": "attacker"},  # Wing position
        {"id": 200001, "x": 55, "y": 34, "team": "defender"},
        {"id": 200002, "x": 60, "y": 20, "team": "defender"},
        {"id": 300001, "x": 100, "y": 34, "team": "keeper"},   # Right goal keeper
    ]
    
    corner_position = (105, 0)  # Bottom-right corner
    
    strategy = strategy_maker.predict_strategy(players, corner_position)
    
    primary = strategy["predictions"]["primary_receiver"]
    decision = strategy["predictions"]["tactical_decision"]
    
    print(f"\n✅ RESULTS:")
    print(f"   Primary Receiver: {primary['player_id']}")
    print(f"   Receiver Score: {primary['score']:.0%}")
    print(f"   Tactical Decision: {decision}")
    
    if primary['player_id'] is not None:
        print(f"\n✅ TEST PASSED - Valid receiver selected for right corner")
        return True
    else:
        print(f"\n❌ TEST FAILED - No receiver selected even though players are in position")
        return False

def main():
    print("\n" + "🧪 REPOSITION SCENARIO FIX - COMPREHENSIVE TEST")
    print("="*70)
    print("Testing receiver selection for both valid and invalid scenarios")
    print("="*70 + "\n")
    
    results = []
    
    # Test 1: Left corner with valid receivers
    results.append(("Left corner - valid receivers", test_left_corner_with_valid_receivers()))
    
    # Test 2: Left corner with no valid receivers (reposition)
    results.append(("Left corner - reposition scenario", test_left_corner_no_valid_receivers()))
    
    # Test 3: Right corner with valid receivers
    results.append(("Right corner - valid receivers", test_right_corner_with_valid_receivers()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Receiver selection works correctly for both left and right corners")
        print("✅ Reposition scenario handled properly when no valid receivers")
    else:
        print("\n⚠️ SOME TESTS FAILED - Please review the output above")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
