#!/usr/bin/env python3
"""
Test script to verify corner side selection feature
"""

from interactive_tactical_setup import InteractiveTacticalSetup
import matplotlib.pyplot as plt

def test_corner_selection():
    """Test that corner side selection works correctly"""
    print("🧪 TESTING CORNER SIDE SELECTION FEATURE")
    print("=" * 60)
    
    try:
        # Create setup instance
        setup = InteractiveTacticalSetup()
        
        # Test initial state
        print(f"\n✅ Initial corner side: {setup.corner_side}")
        print(f"   Corner position: {setup.corner_position}")
        print(f"   Description: {setup.get_corner_description()}")
        
        assert setup.corner_side == "right", "Default corner side should be 'right'"
        assert setup.corner_position == (105, 0), "Default corner position should be (105, 0)"
        
        # Test corner side changes
        test_scenarios = [
            {
                "action": "Switch to left side",
                "method": "_on_corner_left",
                "expected_side": "left",
                "expected_position": (0, 0),
                "expected_desc": "Bottom-Left Corner"
            },
            {
                "action": "Toggle to top-left",
                "method": "_on_corner_left",
                "expected_side": "left",
                "expected_position": (0, 68),
                "expected_desc": "Top-Left Corner"
            },
            {
                "action": "Toggle back to bottom-left",
                "method": "_on_corner_left",
                "expected_side": "left",
                "expected_position": (0, 0),
                "expected_desc": "Bottom-Left Corner"
            },
            {
                "action": "Switch to right side",
                "method": "_on_corner_right",
                "expected_side": "right",
                "expected_position": (105, 0),
                "expected_desc": "Bottom-Right Corner"
            },
            {
                "action": "Toggle to top-right",
                "method": "_on_corner_right",
                "expected_side": "right",
                "expected_position": (105, 68),
                "expected_desc": "Top-Right Corner"
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🎯 Test {i}: {scenario['action']}")
            
            # Call the method
            method = getattr(setup, scenario['method'])
            method(None)
            
            # Verify results
            print(f"   Expected side: {scenario['expected_side']}")
            print(f"   Actual side: {setup.corner_side}")
            print(f"   Expected position: {scenario['expected_position']}")
            print(f"   Actual position: {setup.corner_position}")
            print(f"   Expected description: {scenario['expected_desc']}")
            print(f"   Actual description: {setup.get_corner_description()}")
            
            if (setup.corner_side == scenario['expected_side'] and 
                setup.corner_position == scenario['expected_position'] and
                setup.get_corner_description() == scenario['expected_desc']):
                print(f"   ✅ Test {i} PASSED")
            else:
                print(f"   ❌ Test {i} FAILED")
                return False
        
        print(f"\n✅ ALL CORNER SELECTION TESTS PASSED!")
        print(f"\n📊 FEATURE SUMMARY:")
        print(f"   • 4 corner positions available")
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
    import sys
    success = test_corner_selection()
    sys.exit(0 if success else 1)
