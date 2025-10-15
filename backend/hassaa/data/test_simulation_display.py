#!/usr/bin/env python3
"""
Test script to verify the simulation display is working correctly
Creates a minimal setup and tests the animation system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('TkAgg')  # Ensure we use a proper backend
import matplotlib.pyplot as plt

def test_simulation_display():
    """Test that the simulation displays correctly"""
    print("🧪 TESTING SIMULATION DISPLAY SYSTEM")
    print("=" * 50)
    
    try:
        from interactive_tactical_setup import InteractiveTacticalSetup
        
        print("✅ Interactive setup imported successfully")
        
        # Create setup instance
        setup = InteractiveTacticalSetup()
        print("✅ Setup instance created")
        
        # Add some test players
        test_players = [
            {"id": 100001, "x": 95, "y": 30, "team": "attacker"},
            {"id": 100002, "x": 92, "y": 35, "team": "attacker"},
            {"id": 100003, "x": 90, "y": 32, "team": "attacker"},
            {"id": 100011, "x": 88, "y": 28, "team": "defender"},
            {"id": 100012, "x": 89, "y": 38, "team": "defender"},
            {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
        ]
        
        # Manually add players to setup
        setup.players = test_players
        setup.attackers_placed = 3
        setup.defenders_placed = 2
        setup.keepers_placed = 1
        
        print(f"✅ Added {len(test_players)} test players")
        
        # Test strategy generation and simulation
        if setup.strategy_maker:
            print("🧠 Testing strategy generation...")
            
            strategy = setup.strategy_maker.predict_strategy(test_players, corner_position=(105, 0))
            print("✅ Strategy generated successfully")
            
            # Test simulation startup
            print("🎬 Testing simulation startup...")
            
            # Test that the simulation method exists and can be called
            setup.simulation_active = False
            setup.start_corner_simulation(strategy)
            
            if setup.simulation_active:
                print("✅ Simulation activated successfully")
                print(f"   Ball position: {setup.ball_position}")
                print(f"   Target position: {setup.target_position}")
                
                # Test animation timer
                if hasattr(setup, 'animation_timer') and setup.animation_timer:
                    print("✅ Animation timer created successfully")
                    
                    # Stop the animation after a short time for testing
                    setup.simulation_active = False
                    print("✅ Animation system verified")
                else:
                    print("❌ Animation timer not created")
                    return False
                    
            else:
                print("❌ Simulation not activated")
                return False
                
        else:
            print("❌ Strategy maker not available")
            return False
            
        print("\n🎉 SIMULATION DISPLAY TEST: PASSED")
        print("The 'Generate & Simulate' button should now work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ SIMULATION DISPLAY TEST: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simulation_display()
    if success:
        print("\n🚀 Ready to test with actual UI!")
        print("   Run: python interactive_tactical_setup.py")
        print("   Place some players and click 'Generate & Simulate'")
    else:
        print("\n⚠️  Issues detected. Check the error messages above.")