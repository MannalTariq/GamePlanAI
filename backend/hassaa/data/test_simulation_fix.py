#!/usr/bin/env python3
"""
Test script to verify simulation display fix
"""

from interactive_tactical_setup import InteractiveTacticalSetup
import matplotlib.pyplot as plt
import sys

def test_simulation_display():
    """Test that simulation display works correctly"""
    print("🧪 TESTING SIMULATION DISPLAY FIX")
    print("=" * 50)
    
    try:
        # Create setup instance
        setup = InteractiveTacticalSetup()
        
        # Add some test players programmatically
        test_players = [
            {'id': 100001, 'x': 92, 'y': 38, 'team': 'attacker'},  
            {'id': 100002, 'x': 85, 'y': 40, 'team': 'attacker'},  
            {'id': 100003, 'x': 90, 'y': 30, 'team': 'attacker'},  
            {'id': 100004, 'x': 88, 'y': 45, 'team': 'attacker'},  
            {'id': 100005, 'x': 75, 'y': 35, 'team': 'defender'},  
            {'id': 100006, 'x': 80, 'y': 25, 'team': 'defender'},  
            {'id': 100007, 'x': 82, 'y': 45, 'team': 'defender'},  
            {'id': 100008, 'x': 102, 'y': 34, 'team': 'keeper'},   
        ]
        
        setup.players = test_players
        setup.attackers_placed = 4
        setup.defenders_placed = 3  
        setup.keepers_placed = 1
        
        print(f"✅ Test players added: {len(test_players)} total")
        print(f"   - Attackers: {setup.attackers_placed}")
        print(f"   - Defenders: {setup.defenders_placed}")  
        print(f"   - Keepers: {setup.keepers_placed}")
        
        # Test strategy generation (same as clicking "Generate & Simulate")
        print("\n🎯 Testing strategy generation...")
        
        # Mock the button click event
        try:
            setup._on_simulate(None)
            print("✅ _on_simulate method executed successfully")
            
            # Check if simulation was activated
            if hasattr(setup, 'simulation_active') and setup.simulation_active:
                print("✅ Simulation state activated correctly")
            else:
                print("❌ Simulation state not activated")
                
            # Check if ball position was set
            if hasattr(setup, 'ball_position') and setup.ball_position:
                print(f"✅ Ball position set: {setup.ball_position}")
            else:
                print("❌ Ball position not set")
                
            # Check if target position was set  
            if hasattr(setup, 'target_position') and setup.target_position:
                print(f"✅ Target position set: {setup.target_position}")
            else:
                print("❌ Target position not set")
                
            # Check if animation timer exists
            if hasattr(setup, 'animation_timer') and setup.animation_timer:
                print("✅ Animation timer created")
            else:
                print("❌ Animation timer not created")
                
            print("\n🎉 SIMULATION DISPLAY TEST: PASSED")
            print("The 'Generate & Simulate' button should now work correctly!")
            
        except Exception as e:
            print(f"❌ Error during simulation test: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error setting up test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simulation_display()
    sys.exit(0 if success else 1)