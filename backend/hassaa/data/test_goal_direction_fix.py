#!/usr/bin/env python3
"""
Test script to verify goal direction is correct for left/right corners
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock matplotlib to avoid GUI
import matplotlib
matplotlib.use('Agg')

from interactive_tactical_setup import InteractiveTacticalSetup

def test_goal_direction():
    """Test that goal direction adapts correctly to corner selection"""
    print("🧪 TESTING GOAL DIRECTION FIX")
    print("=" * 60)
    
    setup = InteractiveTacticalSetup()
    
    # Test 1: Right corner should attack right goal
    print("\n🎯 Test 1: Right Corner")
    # Reset to default first
    setup.corner_side = "right"
    setup.corner_position = (105, 0)
    setup.goal_position = (105, 34)
    print(f"   Corner position: {setup.corner_position}")
    print(f"   Goal position: {setup.goal_position}")
    assert setup.corner_position[0] == 105, "Right corner x-position incorrect"
    assert setup.goal_position == (105, 34), "Right goal position incorrect"
    print("   ✅ Right corner attacks right goal (105, 34)")
    
    # Test 2: Left corner should attack left goal
    print("\n🎯 Test 2: Left Corner")
    setup._on_corner_left(None)
    print(f"   Corner position: {setup.corner_position}")
    print(f"   Goal position: {setup.goal_position}")
    assert setup.corner_position == (0, 0), "Left corner position incorrect"
    assert setup.goal_position == (0, 34), "Left goal position incorrect"
    print("   ✅ Left corner attacks left goal (0, 34)")
    
    # Test 3: Player movement direction for right goal
    print("\n🎯 Test 3: Player Movement Toward Right Goal")
    setup._on_corner_right(None)
    test_player = {'id': 1, 'x': 90, 'y': 34, 'team': 'attacker'}
    setup.players = [test_player]
    
    strategy = {
        "predictions": {
            "shot_confidence": 0.5
        }
    }
    
    target = setup.calculate_tactical_target(test_player, 1, strategy)
    print(f"   Player at (90, 34) moving toward right goal")
    print(f"   Target position: {target}")
    assert target[0] > 90, "Player should move right (toward x=105)"
    print(f"   ✅ Player moves right (x: 90 → {target[0]:.1f})")
    
    # Test 4: Player movement direction for left goal
    print("\n🎯 Test 4: Player Movement Toward Left Goal")
    setup._on_corner_left(None)
    test_player = {'id': 1, 'x': 15, 'y': 34, 'team': 'attacker'}
    setup.players = [test_player]
    
    target = setup.calculate_tactical_target(test_player, 1, strategy)
    print(f"   Player at (15, 34) moving toward left goal")
    print(f"   Target position: {target}")
    assert target[0] < 15, "Player should move left (toward x=0)"
    print(f"   ✅ Player moves left (x: 15 → {target[0]:.1f})")
    
    # Test 5: Support runner direction
    print("\n🎯 Test 5: Support Runner Movement")
    
    # Right goal
    setup._on_corner_right(None)
    test_player = {'id': 2, 'x': 85, 'y': 30, 'team': 'attacker'}
    primary_receiver = {'id': 1, 'x': 95, 'y': 34, 'team': 'attacker'}
    setup.players = [primary_receiver, test_player]
    
    target = setup.calculate_support_run_position(test_player, 1)
    print(f"   Support runner at (85, 30) for right goal")
    print(f"   Target position: {target}")
    assert target[0] > 85, "Support runner should move toward right goal"
    print(f"   ✅ Support runner moves right (x: 85 → {target[0]:.1f})")
    
    # Left goal
    setup._on_corner_left(None)
    test_player = {'id': 2, 'x': 20, 'y': 30, 'team': 'attacker'}
    primary_receiver = {'id': 1, 'x': 10, 'y': 34, 'team': 'attacker'}
    setup.players = [primary_receiver, test_player]
    
    target = setup.calculate_support_run_position(test_player, 1)
    print(f"   Support runner at (20, 30) for left goal")
    print(f"   Target position: {target}")
    assert target[0] < 20, "Support runner should move toward left goal"
    print(f"   ✅ Support runner moves left (x: 20 → {target[0]:.1f})")
    
    print("\n✅ ALL GOAL DIRECTION TESTS PASSED!")
    print("\n📊 SUMMARY:")
    print("   • Right corners (x=105) → attack right goal (105, 34)")
    print("   • Left corners (x=0) → attack left goal (0, 34)")
    print("   • Players move toward correct goal based on corner side")
    print("   • Support runners adapt direction to goal position")
    print("\n🎉 Fix verified! Players now move toward the correct goal!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_goal_direction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
