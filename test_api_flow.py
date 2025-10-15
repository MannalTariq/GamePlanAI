#!/usr/bin/env python3
"""
Test script to verify the complete API data flow
"""

import requests
import json
import time

def test_api_flow():
    """Test the complete API data flow from frontend to backend"""
    
    print("===== API DATA FLOW TEST =====")
    
    # Test data (simulating frontend request)
    test_data = {
        "players": [
            {"id": 1, "xPct": 80, "yPct": 30, "role": "red", "label": "Attacker1"},
            {"id": 2, "xPct": 85, "yPct": 40, "role": "red", "label": "Attacker2"},
            {"id": 3, "xPct": 75, "yPct": 25, "role": "blue", "label": "Defender1"},
            {"id": 4, "xPct": 50, "yPct": 50, "role": "gk", "label": "Goalkeeper"}
        ],
        "cornerPosition": {"x": 95, "y": 5},
        "goalPosition": {"x": 95, "y": 50},
        "setPiece": "Corner Kick"
    }
    
    print("Sending test request to API...")
    print(f"   Players: {len(test_data['players'])}")
    print(f"   Corner: ({test_data['cornerPosition']['x']}%, {test_data['cornerPosition']['y']}%)")
    print(f"   Goal: ({test_data['goalPosition']['x']}%, {test_data['goalPosition']['y']}%)")
    
    try:
        # Send request to API
        response = requests.post(
            "http://localhost:5000/api/simulate",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\nAPI Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API Response Success!")
            
            # Verify response structure
            print(f"\nResponse Structure Verification:")
            print(f"   success: {result.get('success', 'MISSING')}")
            print(f"   setPiece: {result.get('setPiece', 'MISSING')}")
            print(f"   totalPlayers: {result.get('totalPlayers', 'MISSING')}")
            
            if 'prediction' in result:
                pred = result['prediction']
                print(f"   prediction.primaryPlayer: {pred.get('primaryPlayer', 'MISSING')}")
                print(f"   prediction.shotConfidence: {pred.get('shotConfidence', 'MISSING')}%")
                print(f"   prediction.tacticalDecision: {pred.get('tacticalDecision', 'MISSING')}")
                print(f"   prediction.successRate: {pred.get('successRate', 'MISSING')}%")
            
            if 'simulation' in result:
                sim = result['simulation']
                print(f"   simulation.ballTrajectory.points: {len(sim.get('ballTrajectory', {}).get('points', []))}")
                print(f"   simulation.playerMovements: {len(sim.get('playerMovements', []))}")
                print(f"   simulation.primaryReceiver.player_id: {sim.get('primaryReceiver', {}).get('player_id', 'MISSING')}")
            
            # Verify coordinate data integrity
            print(f"\nCoordinate Data Verification:")
            if 'simulation' in result and 'ballTrajectory' in result['simulation']:
                bt = result['simulation']['ballTrajectory']
                print(f"   Ball trajectory start: ({bt.get('start', {}).get('x', 'MISSING')}%, {bt.get('start', {}).get('y', 'MISSING')}%)")
                print(f"   Ball trajectory end: ({bt.get('end', {}).get('x', 'MISSING')}%, {bt.get('end', {}).get('y', 'MISSING')}%)")
                print(f"   Ball trajectory points: {len(bt.get('points', []))}")
            
            if 'simulation' in result and 'playerMovements' in result['simulation']:
                for i, movement in enumerate(result['simulation']['playerMovements'][:3]):
                    print(f"   Player {movement.get('playerId', 'N/A')} movement:")
                    print(f"     Start: ({movement.get('startPos', {}).get('x', 'MISSING')}%, {movement.get('startPos', {}).get('y', 'MISSING')}%)")
                    print(f"     Target: ({movement.get('targetPos', {}).get('x', 'MISSING')}%, {movement.get('targetPos', {}).get('y', 'MISSING')}%)")
            
            print(f"\nAPI Data Flow Test PASSED!")
            return True
            
        else:
            print(f"API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_api_flow()
    if success:
        print("\nAll tests passed! API data flow is working correctly.")
    else:
        print("\nTests failed! Check the logs above for details.")
