#!/usr/bin/env python3
"""
Test script to verify receiver selection works for left corners
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock matplotlib to avoid GUI
import matplotlib
matplotlib.use('Agg')

from strategy_maker import StrategyMaker

def test_left_corner_receivers():
    """Test that left corner scenarios properly select receivers"""
    print("🧪 TESTING RECEIVER SELECTION FOR LEFT CORNERS")
    print("=" * 60)
    
    strategy_maker = StrategyMaker()
    
    # Test scenario: Left corner with players on left side
    print("\n🎯 Test Scenario: Left Corner (0, 0)")
    print("   Players positioned on LEFT side of pitch (x < 50)")
    
    players = [
        {'id': 100001, 'x': 10, 'y': 30, 'team': 'attacker'},  # Very close to left goal
        {'id': 100002, 'x': 15, 'y': 35, 'team': 'attacker'},  # Near left goal
        {'id': 100003, 'x': 20, 'y': 40, 'team': 'attacker'},  # Mid-range from left goal
        {'id': 100004, 'x': 25, 'y': 34, 'team': 'attacker'},  # Central position
        {'id': 100005, 'x': 12, 'y': 28, 'team': 'defender'},  # Defender
        {'id': 100006, 'x': 18, 'y': 42, 'team': 'defender'},  # Defender
        {'id': 100007, 'x': 2, 'y': 34, 'team': 'keeper'},     # Goalkeeper at left goal
    ]
    
    corner_position = (0, 0)  # Left corner
    
    print(f"\n📊 Player Positions:")
    for p in players:
        if p['team'] == 'attacker':
            print(f"   Attacker #{p['id']}: ({p['x']:.1f}, {p['y']:.1f})")
    
    try:
        strategy = strategy_maker.predict_strategy(players, corner_position)
        
        primary_receiver = strategy["predictions"]["primary_receiver"]
        
        print(f"\n✅ STRATEGY GENERATED SUCCESSFULLY!")
        print(f"   Primary Receiver: Player #{primary_receiver['player_id']}")
        print(f"   Score: {primary_receiver['score']:.2%}")
        print(f"   Tactical Decision: {strategy['predictions']['tactical_decision']}")
        
        # Verify receiver was selected
        if primary_receiver['player_id'] is not None:
            print(f"\n✅ TEST PASSED!")
            print(f"   A valid receiver was selected for left corner")
            
            # Verify it's one of our attackers
            if primary_receiver['player_id'] in [100001, 100002, 100003, 100004]:
                print(f"   Receiver is correctly from left-side attackers")
            else:
                print(f"   ⚠️  Warning: Receiver is not from expected attackers")
            
            return True
        else:
            print(f"\n❌ TEST FAILED!")
            print(f"   No receiver selected (got None)")
            print(f"   This means filtering is still too strict")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_right_corner_receivers():
    """Test that right corner scenarios still work"""
    print("\n\n🎯 Test Scenario: Right Corner (105, 0)")
    print("   Players positioned on RIGHT side of pitch (x > 70)")
    
    strategy_maker = StrategyMaker()
    
    players = [
        {'id': 100001, 'x': 95, 'y': 30, 'team': 'attacker'},  # Very close to right goal
        {'id': 100002, 'x': 90, 'y': 35, 'team': 'attacker'},  # Near right goal
        {'id': 100003, 'x': 85, 'y': 40, 'team': 'attacker'},  # Mid-range
        {'id': 100004, 'x': 80, 'y': 34, 'team': 'attacker'},  # Central
        {'id': 100005, 'x': 93, 'y': 28, 'team': 'defender'},  # Defender
        {'id': 100006, 'x': 88, 'y': 42, 'team': 'defender'},  # Defender
        {'id': 100007, 'x': 103, 'y': 34, 'team': 'keeper'},   # Goalkeeper
    ]
    
    corner_position = (105, 0)  # Right corner
    
    try:
        strategy = strategy_maker.predict_strategy(players, corner_position)
        
        primary_receiver = strategy["predictions"]["primary_receiver"]
        
        print(f"\n✅ STRATEGY GENERATED SUCCESSFULLY!")
        print(f"   Primary Receiver: Player #{primary_receiver['player_id']}")
        print(f"   Score: {primary_receiver['score']:.2%}")
        
        if primary_receiver['player_id'] is not None:
            print(f"\n✅ RIGHT CORNER TEST PASSED!")
            return True
        else:
            print(f"\n❌ RIGHT CORNER TEST FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ RIGHT CORNER TEST FAILED WITH ERROR!")
        print(f"   Error: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RECEIVER SELECTION FIX VERIFICATION")
    print("="*60)
    
    left_test = test_left_corner_receivers()
    right_test = test_right_corner_receivers()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Left Corner Test:  {'✅ PASSED' if left_test else '❌ FAILED'}")
    print(f"Right Corner Test: {'✅ PASSED' if right_test else '❌ FAILED'}")
    
    if left_test and right_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("Receiver selection now works for both left and right corners!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
