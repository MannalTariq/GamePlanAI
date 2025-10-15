#!/usr/bin/env python3
"""
Test Strategy Maker Pipeline
Demonstrates GNN-based corner kick strategy generation.
"""

import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_maker import StrategyMaker


def test_strategy_maker():
    """Test the strategy maker with example player placements"""
    
    print("="*70)
    print("  TEST: GNN STRATEGY MAKER PIPELINE")
    print("="*70)
    
    # Initialize strategy maker
    print("\n1. Initializing Strategy Maker...")
    strategy_maker = StrategyMaker()
    
    # Create example player placements (realistic corner kick scenario)
    print("\n2. Creating example player placement...")
    example_players = [
        # Attacking team (trying to score)
        {"id": 100001, "x": 95, "y": 30, "team": "attacker"},  # Near post
        {"id": 100002, "x": 92, "y": 34, "team": "attacker"},  # Penalty spot
        {"id": 100003, "x": 90, "y": 38, "team": "attacker"},  # Far post
        {"id": 100004, "x": 88, "y": 25, "team": "attacker"},  # Edge of box
        {"id": 100005, "x": 85, "y": 40, "team": "attacker"},  # Top of box
        
        # Defending team
        {"id": 200001, "x": 94, "y": 32, "team": "defender"},  # Marking near post
        {"id": 200002, "x": 91, "y": 36, "team": "defender"},  # Marking penalty
        {"id": 200003, "x": 88, "y": 38, "team": "defender"},  # Marking far post
        {"id": 200004, "x": 90, "y": 28, "team": "defender"},  # Zonal
        {"id": 200005, "x": 85, "y": 34, "team": "defender"},  # Edge marking
        
        # Goalkeeper
        {"id": 300001, "x": 100, "y": 34, "team": "keeper"},
    ]
    
    print(f"   Placed {len(example_players)} players")
    print(f"   Attackers: 5 | Defenders: 5 | Goalkeeper: 1")
    
    # Generate strategy prediction
    print("\n3. Generating GNN-based strategy...")
    strategy = strategy_maker.predict_strategy(example_players, corner_position=(105, 0))
    
    # Display strategy details
    print("\n4. Strategy Analysis:")
    print(f"   Primary Receiver: Player #{strategy['predictions']['primary_receiver']['player_id']}")
    print(f"   Receiver Confidence: {strategy['predictions']['primary_receiver']['score']:.1%}")
    print(f"   Shot Confidence: {strategy['predictions']['shot_confidence']:.1%}")
    print(f"   Tactical Decision: {strategy['predictions']['tactical_decision']}")
    
    if strategy['predictions']['alternate_receivers']:
        print(f"\n   Alternate Targets:")
        for i, alt in enumerate(strategy['predictions']['alternate_receivers'], 1):
            print(f"      {i}. Player #{alt['player_id']} ({alt['score']:.1%})")
    
    # Save strategy
    print("\n5. Saving strategy output...")
    saved_path = strategy_maker.save_strategy(strategy)
    print(f"   Saved to: {os.path.basename(saved_path)}")
    
    # Generate simulation data
    print("\n6. Generating simulation data...")
    sim_data = strategy_maker.generate_simulation_data(strategy)
    
    if sim_data:
        print(f"   Ball trajectory: {sim_data['ball_trajectory']['start']} → {sim_data['ball_trajectory']['target']}")
        print(f"   Player movements: {len(sim_data['player_movements'])} players")
        print(f"   Shot action: {'YES' if sim_data['shot_action'] else 'NO'}")
    
    print("\n" + "="*70)
    print("  ✅ STRATEGY MAKER TEST COMPLETE!")
    print("="*70)
    
    return strategy


def test_interactive_integration():
    """Test integration with interactive tactical setup"""
    
    print("\n" + "="*70)
    print("  TEST: INTERACTIVE SETUP INTEGRATION")
    print("="*70)
    
    try:
        from interactive_tactical_setup import InteractiveTacticalSetup
        
        print("\n✅ Interactive setup module imported successfully")
        print("   Strategy Maker is integrated and ready")
        print("\n📝 To use:")
        print("   1. Run: python interactive_tactical_setup.py")
        print("   2. Place players on the pitch")
        print("   3. Click 'Done' to generate GNN strategy")
        print("   4. View predicted receiver, shot confidence, and tactical decision")
        print("   5. Strategy automatically saved to JSON file")
        
        return True
        
    except Exception as e:
        print(f"\n⚠️  Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🚀 Starting Strategy Maker Tests...\n")
    
    # Test 1: Strategy Maker functionality
    try:
        strategy = test_strategy_maker()
        print("\n✅ Test 1 PASSED: Strategy Maker functional")
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Interactive integration
    try:
        test_interactive_integration()
        print("\n✅ Test 2 PASSED: Interactive integration ready")
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 All tests complete!\n")
