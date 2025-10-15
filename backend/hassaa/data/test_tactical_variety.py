#!/usr/bin/env python3
"""
Test script to verify tactical decision variety and player limits
"""

from interactive_tactical_setup import InteractiveTacticalSetup
import matplotlib.pyplot as plt
import sys

def test_tactical_variety():
    """Test that tactical decisions are varied and player limits work"""
    print("🧪 TESTING TACTICAL DECISION VARIETY & PLAYER LIMITS")
    print("=" * 60)
    
    try:
        # Create setup instance
        setup = InteractiveTacticalSetup()
        
        # Verify new player limits
        print(f"✅ Player limits updated:")
        print(f"   - Max Attackers: {setup.max_attackers} (should be 9)")
        print(f"   - Max Defenders: {setup.max_defenders} (should be 10)")  
        print(f"   - Max Keepers: {setup.max_keepers} (should be 1)")
        
        if setup.max_attackers != 9 or setup.max_keepers != 1:
            print("❌ Player limits not updated correctly!")
            return False
            
        # Test multiple different formations to get varied tactical decisions
        test_scenarios = [
            {
                "name": "Scenario 1: Crowded penalty area",
                "players": [
                    {'id': 1001, 'x': 95, 'y': 32, 'team': 'attacker'},  # Very close to goal
                    {'id': 1002, 'x': 92, 'y': 36, 'team': 'attacker'},  # Near post
                    {'id': 1003, 'x': 88, 'y': 40, 'team': 'attacker'},  # Further out
                    {'id': 1004, 'x': 85, 'y': 30, 'team': 'attacker'},  # Far side
                    {'id': 2001, 'x': 93, 'y': 34, 'team': 'defender'},  # Close marking
                    {'id': 2002, 'x': 89, 'y': 38, 'team': 'defender'},  # Area coverage
                    {'id': 3001, 'x': 103, 'y': 34, 'team': 'keeper'},  # Goalkeeper
                ]
            },
            {
                "name": "Scenario 2: Spread formation",
                "players": [
                    {'id': 1001, 'x': 80, 'y': 25, 'team': 'attacker'},  # Deep position
                    {'id': 1002, 'x': 90, 'y': 15, 'team': 'attacker'},  # Wide position
                    {'id': 1003, 'x': 85, 'y': 45, 'team': 'attacker'},  # Other side
                    {'id': 1004, 'x': 75, 'y': 35, 'team': 'attacker'},  # Very deep
                    {'id': 2001, 'x': 70, 'y': 30, 'team': 'defender'},  # Distant marking
                    {'id': 2002, 'x': 82, 'y': 20, 'team': 'defender'},  # Wide coverage
                    {'id': 3001, 'x': 103, 'y': 34, 'team': 'keeper'},  # Goalkeeper
                ]
            },
            {
                "name": "Scenario 3: Near post focus",
                "players": [
                    {'id': 1001, 'x': 96, 'y': 30, 'team': 'attacker'},  # Near post target
                    {'id': 1002, 'x': 94, 'y': 28, 'team': 'attacker'},  # Support near
                    {'id': 1003, 'x': 87, 'y': 42, 'team': 'attacker'},  # Far post cover
                    {'id': 1004, 'x': 82, 'y': 35, 'team': 'attacker'},  # Mid-range
                    {'id': 2001, 'x': 97, 'y': 32, 'team': 'defender'},  # Tight marking near post
                    {'id': 2002, 'x': 90, 'y': 40, 'team': 'defender'},  # Far post cover
                    {'id': 3001, 'x': 103, 'y': 34, 'team': 'keeper'},  # Goalkeeper
                ]
            }
        ]
        
        tactical_decisions = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\\n🎯 Testing {scenario['name']}...")
            
            # Set up players for this scenario
            setup.players = scenario['players']
            setup.attackers_placed = len([p for p in scenario['players'] if p['team'] == 'attacker'])
            setup.defenders_placed = len([p for p in scenario['players'] if p['team'] == 'defender'])
            setup.keepers_placed = len([p for p in scenario['players'] if p['team'] == 'keeper'])
            
            try:
                # Generate strategy (same as clicking Generate & Simulate)
                setup._on_simulate(None)
                
                # Extract tactical decision from the strategy
                if hasattr(setup, 'predicted_strategy') and setup.predicted_strategy:
                    decision = setup.predicted_strategy["predictions"]["tactical_decision"]
                    receiver_id = setup.predicted_strategy["predictions"]["primary_receiver"]["player_id"]
                    shot_conf = setup.predicted_strategy["predictions"]["shot_confidence"]
                    
                    tactical_decisions.append(decision)
                    
                    print(f"   ✅ Tactical Decision: '{decision}'")
                    print(f"   🎯 Primary Receiver: Player #{receiver_id}")
                    print(f"   ⚽ Shot Confidence: {shot_conf:.0%}")
                    
                else:
                    print(f"   ❌ No strategy generated for scenario {i}")
                    
            except Exception as e:
                print(f"   ❌ Error in scenario {i}: {e}")
                continue
                
        print(f"\\n📊 TACTICAL DECISION VARIETY TEST:")
        print(f"   Total scenarios tested: {len(test_scenarios)}")
        print(f"   Decisions generated: {len(tactical_decisions)}")
        print(f"   Unique decisions: {len(set(tactical_decisions))}")
        
        print(f"\\n🎲 All tactical decisions:")
        for i, decision in enumerate(tactical_decisions, 1):
            print(f"   {i}. {decision}")
            
        # Check for variety
        unique_decisions = len(set(tactical_decisions))
        if unique_decisions >= 2:
            print(f"\\n✅ TACTICAL VARIETY TEST: PASSED")
            print(f"   Generated {unique_decisions} different tactical decisions")
            print(f"   This shows the system is dynamic and not hardcoded!")
            return True
        else:
            print(f"\\n❌ TACTICAL VARIETY TEST: FAILED")
            print(f"   Only {unique_decisions} unique decision(s) generated")
            print(f"   Expected at least 2 different tactical decisions")
            if len(tactical_decisions) > 0:
                print(f"   All decisions were: '{tactical_decisions[0]}'")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tactical_variety()
    sys.exit(0 if success else 1)