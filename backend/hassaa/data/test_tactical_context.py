#!/usr/bin/env python3
"""
Enhanced Tactical Context Test for GNN Corner Kick Strategy
Tests the new tactical weighting system including:
- Unmarked player detection and bonuses
- Distance to goal weighting
- Tactical positioning bonuses
- Enhanced simulation with player movement
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_maker import StrategyMaker
import json
import math

def create_unmarked_player_test():
    """Test scenario with clearly unmarked player near goal"""
    print(f"\n{'='*80}")
    print(f"  TEST: UNMARKED PLAYER NEAR GOAL")
    print(f"{'='*80}")
    
    # Create formation with one unmarked attacker very close to goal
    players = [
        # Unmarked attacker in excellent position (should be selected)
        {"id": 100001, "x": 98, "y": 34, "team": "attacker"},  # 7m from goal, unmarked
        
        # Other attackers in decent positions but marked
        {"id": 100002, "x": 92, "y": 30, "team": "attacker"},  # Marked by defender
        {"id": 100003, "x": 94, "y": 38, "team": "attacker"},  # Marked by defender
        {"id": 100004, "x": 90, "y": 34, "team": "attacker"},  # Further from goal
        
        # Defenders marking most attackers (but missing the unmarked one)
        {"id": 100011, "x": 91, "y": 29, "team": "defender"},  # Marking 100002
        {"id": 100012, "x": 93, "y": 37, "team": "defender"},  # Marking 100003
        {"id": 100013, "x": 85, "y": 40, "team": "defender"},  # Away from action
        
        # Goalkeeper
        {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
    ]
    
    strategy_maker = StrategyMaker()
    strategy = strategy_maker.predict_strategy(players, corner_position=(105, 0))
    
    primary = strategy["predictions"]["primary_receiver"]
    return {
        "test_name": "Unmarked Player Near Goal",
        "expected_winner": 100001,  # Should pick the unmarked player
        "actual_winner": primary["player_id"],
        "winner_score": primary["score"],
        "passed": primary["player_id"] == 100001,
        "strategy": strategy
    }

def create_distance_priority_test():
    """Test scenario where distance to goal should be the deciding factor"""
    print(f"\n{'='*80}")
    print(f"  TEST: DISTANCE TO GOAL PRIORITY")
    print(f"{'='*80}")
    
    # All players unmarked, but different distances to goal
    players = [
        {"id": 100001, "x": 88, "y": 34, "team": "attacker"},  # 17m from goal
        {"id": 100002, "x": 95, "y": 34, "team": "attacker"},  # 10m from goal (closest)
        {"id": 100003, "x": 82, "y": 34, "team": "attacker"},  # 23m from goal
        {"id": 100004, "x": 90, "y": 34, "team": "attacker"},  # 15m from goal
        
        # Defenders far away (all attackers unmarked)
        {"id": 100011, "x": 70, "y": 20, "team": "defender"},
        {"id": 100012, "x": 75, "y": 45, "team": "defender"},
        {"id": 100013, "x": 80, "y": 50, "team": "defender"},
        
        {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
    ]
    
    strategy_maker = StrategyMaker()
    strategy = strategy_maker.predict_strategy(players, corner_position=(105, 0))
    
    primary = strategy["predictions"]["primary_receiver"]
    return {
        "test_name": "Distance to Goal Priority",
        "expected_winner": 100002,  # Closest to goal
        "actual_winner": primary["player_id"],
        "winner_score": primary["score"],
        "passed": primary["player_id"] == 100002,
        "strategy": strategy
    }

def create_tactical_positioning_test():
    """Test tactical positioning bonus (penalty area vs outside)"""
    print(f"\n{'='*80}")
    print(f"  TEST: TACTICAL POSITIONING BONUS")
    print(f"{'='*80}")
    
    players = [
        {"id": 100001, "x": 92, "y": 34, "team": "attacker"},  # Outside penalty area
        {"id": 100002, "x": 90, "y": 32, "team": "attacker"},  # In penalty area (should win)
        {"id": 100003, "x": 85, "y": 35, "team": "attacker"},  # Further out
        
        # Defenders moderately close (all partially marked)
        {"id": 100011, "x": 88, "y": 28, "team": "defender"},
        {"id": 100012, "x": 87, "y": 38, "team": "defender"},
        {"id": 100013, "x": 82, "y": 30, "team": "defender"},
        
        {"id": 100021, "x": 103, "y": 34, "team": "keeper"},
    ]
    
    strategy_maker = StrategyMaker()
    strategy = strategy_maker.predict_strategy(players, corner_position=(105, 0))
    
    primary = strategy["predictions"]["primary_receiver"]
    return {
        "test_name": "Tactical Positioning Bonus",
        "expected_winner": 100002,  # Player in penalty area
        "actual_winner": primary["player_id"],
        "winner_score": primary["score"],
        "passed": primary["player_id"] == 100002,
        "strategy": strategy
    }

def test_simulation_features(strategy):
    """Test that enhanced simulation features work"""
    print(f"\n🎬 TESTING SIMULATION FEATURES:")
    
    # Test player state initialization
    try:
        from interactive_tactical_setup import InteractiveTacticalSetup
        setup = InteractiveTacticalSetup()
        
        # Mock some players for testing
        setup.players = [
            {"id": 100001, "x": 95, "y": 30, "team": "attacker"},
            {"id": 100002, "x": 88, "y": 40, "team": "defender"},
            {"id": 100003, "x": 103, "y": 34, "team": "keeper"},
        ]
        
        # Test player state initialization
        setup.initialize_player_states(strategy)
        
        print(f"   ✅ Player states initialized: {len(setup.player_states)} players")
        
        # Test tactical target calculation
        for player in setup.players:
            target = setup.calculate_tactical_target(
                player, 
                strategy["predictions"]["primary_receiver"]["player_id"],
                strategy
            )
            print(f"   📍 Player #{player['id']} ({player['team']}): {player['x']:.0f},{player['y']:.0f} → {target[0]:.0f},{target[1]:.0f}")
        
        # Test unmarked detection
        for player in setup.players:
            if player['team'] == 'attacker':
                is_unmarked, dist = setup._check_if_unmarked_player(player)
                print(f"   🎯 Player #{player['id']}: {'UNMARKED' if is_unmarked else 'MARKED'} (closest defender: {dist:.1f}m)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Simulation test failed: {e}")
        return False

def main():
    """Run comprehensive tactical context tests"""
    print("🧪 TESTING ENHANCED TACTICAL CONTEXT SYSTEM")
    print("=" * 80)
    print("Testing new features:")
    print("1. Unmarked player detection and bonuses")
    print("2. Distance to goal weighting")
    print("3. Tactical positioning bonuses")
    print("4. Enhanced simulation with player movement")
    print("=" * 80)
    
    tests = [
        create_unmarked_player_test(),
        create_distance_priority_test(),
        create_tactical_positioning_test()
    ]
    
    # Analyze results
    print(f"\n{'='*80}")
    print(f"  TACTICAL CONTEXT TEST RESULTS")
    print(f"{'='*80}")
    
    passed_tests = 0
    for test in tests:
        status = "✅ PASS" if test["passed"] else "❌ FAIL"
        print(f"\n{status} {test['test_name']}")
        print(f"   Expected Winner: Player #{test['expected_winner']}")
        print(f"   Actual Winner: Player #{test['actual_winner']} (Score: {test['winner_score']:.3f})")
        
        if test["passed"]:
            passed_tests += 1
        else:
            print(f"   ⚠️  The tactical context weighting may need adjustment")
    
    # Test simulation features
    print(f"\n📊 SIMULATION FEATURES TEST:")
    sim_test_passed = test_simulation_features(tests[0]["strategy"])
    
    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    print(f"   Tactical Context Tests: {passed_tests}/{len(tests)} passed")
    print(f"   Simulation Features: {'✅ Working' if sim_test_passed else '❌ Issues detected'}")
    
    total_score = passed_tests + (1 if sim_test_passed else 0)
    max_score = len(tests) + 1
    
    if total_score == max_score:
        print(f"\n🎉 ALL TESTS PASS - Enhanced tactical context is working perfectly!")
        print(f"   🎯 Unmarked players get priority")
        print(f"   📏 Distance to goal affects selection")
        print(f"   🏟️ Tactical positioning matters")
        print(f"   🎬 Enhanced simulation with realistic movement")
    elif total_score >= max_score * 0.75:
        print(f"\n✅ MOSTLY WORKING - {total_score}/{max_score} features functional")
        print(f"   Minor adjustments may be needed for optimal performance")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT - {total_score}/{max_score} features working")
        print(f"   Tactical weighting system may need debugging")
    
    # Save detailed results
    with open("tactical_context_test_results.json", "w") as f:
        json.dump({
            "test_summary": {
                "tests_passed": passed_tests,
                "total_tests": len(tests),
                "simulation_working": sim_test_passed,
                "overall_score": f"{total_score}/{max_score}"
            },
            "test_details": tests
        }, f, indent=2)
    
    print(f"\n📝 Detailed results saved to: tactical_context_test_results.json")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()