#!/usr/bin/env python3
"""
Simple test script for corner side selection (no GUI)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock matplotlib to avoid GUI
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from interactive_tactical_setup import InteractiveTacticalSetup

def test_corner_selection():
    """Test that corner side selection works correctly"""
    print("🧪 TESTING CORNER SIDE SELECTION FEATURE")
    print("=" * 60)
    
    try:
        # Create setup instance
        setup = InteractiveTacticalSetup()
        
        # Test initial state
        print(f"\n✅ Initial state:")
        print(f"   Corner side: {setup.corner_side}")
        print(f"   Corner position: {setup.corner_position}")
        print(f"   Description: {setup.get_corner_description()}")
        
        assert setup.corner_side == "right", "Default corner side should be 'right'"
        assert setup.corner_position == (105, 0), "Default corner position should be (105, 0)"
        print("   ✅ Default state correct")
        
        # Test switching to left
        print(f"\n🎯 Test 1: Switch to left side")
        setup._on_corner_left(None)
        print(f"   Corner side: {setup.corner_side}")
        print(f"   Corner position: {setup.corner_position}")
        print(f"   Description: {setup.get_corner_description()}")
        assert setup.corner_side == "left"
        assert setup.corner_position == (0, 0)
        assert setup.get_corner_description() == "Bottom-Left Corner"
        print("   ✅ Test 1 PASSED")
        
        # Test toggling to top-left
        print(f"\n🎯 Test 2: Toggle to top-left")
        setup._on_corner_left(None)
        print(f"   Corner side: {setup.corner_side}")
        print(f"   Corner position: {setup.corner_position}")
        print(f"   Description: {setup.get_corner_description()}")
        assert setup.corner_side == "left"
        assert setup.corner_position == (0, 68)
        assert setup.get_corner_description() == "Top-Left Corner"
        print("   ✅ Test 2 PASSED")
        
        # Test switching to right
        print(f"\n🎯 Test 3: Switch to right side")
        setup._on_corner_right(None)
        print(f"   Corner side: {setup.corner_side}")
        print(f"   Corner position: {setup.corner_position}")
        print(f"   Description: {setup.get_corner_description()}")
        assert setup.corner_side == "right"
        assert setup.corner_position == (105, 0)
        assert setup.get_corner_description() == "Bottom-Right Corner"
        print("   ✅ Test 3 PASSED")
        
        # Test toggling to top-right
        print(f"\n🎯 Test 4: Toggle to top-right")
        setup._on_corner_right(None)
        print(f"   Corner side: {setup.corner_side}")
        print(f"   Corner position: {setup.corner_position}")
        print(f"   Description: {setup.get_corner_description()}")
        assert setup.corner_side == "right"
        assert setup.corner_position == (105, 68)
        assert setup.get_corner_description() == "Top-Right Corner"
        print("   ✅ Test 4 PASSED")
        
        print(f"\n✅ ALL CORNER SELECTION TESTS PASSED!")
        print(f"\n📊 FEATURE SUMMARY:")
        print(f"   • 4 corner positions available:")
        print(f"     - Bottom-Right (105, 0)")
        print(f"     - Top-Right (105, 68)")
        print(f"     - Bottom-Left (0, 0)")
        print(f"     - Top-Left (0, 68)")
        print(f"   • Left/Right buttons to switch sides")
        print(f"   • Click same button to toggle top/bottom")
        print(f"   • Visual highlight on selected corner")
        print(f"   • Corner position used in strategy generation")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_corner_selection()
    sys.exit(0 if success else 1)
