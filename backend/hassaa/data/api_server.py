#!/usr/bin/env python3
"""
Flask API Server for Corner Kick Tactical Setup
Bridges the Python backend with React frontend
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import sys
import base64
import io
import math
from datetime import datetime
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from strategy_maker import StrategyMaker

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize Strategy Maker
strategy_maker = None

def get_player_label(players, player_id):
    """Get the display label for a player by ID"""
    player = next((p for p in players if p['id'] == player_id), None)
    return player['label'] if player else f"Player {player_id}"

def calculate_ball_trajectory(corner_x, corner_y, primary_receiver, goal_x, goal_y):
    """
    Calculate ball trajectory points for frontend animation
    Returns list of points along Bezier curve from corner to receiver
    """
    
    # Get receiver position
    if primary_receiver and primary_receiver.get('position'):
        receiver_x = primary_receiver['position']['x']
        receiver_y = primary_receiver['position']['y']
    else:
        # Fallback to goal position
        receiver_x = goal_x
        receiver_y = goal_y
    
    # Calculate control point for Bezier curve (high arc)
    control_x = (corner_x + receiver_x) / 2
    control_y = (corner_y + receiver_y) / 2 - 15  # High arc
    
    # Generate trajectory points
    trajectory_points = []
    for t in [i / 30.0 for i in range(31)]:  # 31 points for smooth animation
        # Quadratic Bezier curve
        x = (1-t)**2 * corner_x + 2*(1-t)*t * control_x + t**2 * receiver_x
        y = (1-t)**2 * corner_y + 2*(1-t)*t * control_y + t**2 * receiver_y
        
        # Convert back to frontend percentage coordinates
        x_pct = (x / 105) * 100
        y_pct = (y / 68) * 100
        
        trajectory_points.append({
            'x': x_pct,
            'y': y_pct,
            'progress': t
        })
    
    return {
        'start': {'x': (corner_x / 105) * 100, 'y': (corner_y / 68) * 100},
        'end': {'x': (receiver_x / 105) * 100, 'y': (receiver_y / 68) * 100},
        'control': {'x': (control_x / 105) * 100, 'y': (control_y / 68) * 100},
        'points': trajectory_points
    }

def calculate_player_movements(players, primary_receiver, goal_x, goal_y):
    """
    Calculate player movement trajectories for animation
    """
    movements = []
    primary_receiver_id = primary_receiver.get('player_id') if primary_receiver else None
    
    for player in players:
        start_x_pct = (player['x'] / 105) * 100
        start_y_pct = (player['y'] / 68) * 100
        
        # Calculate target position based on role and primary receiver
        if player['id'] == primary_receiver_id:
            # Primary receiver moves toward goal
            target_x = player['x'] + (goal_x - player['x']) * 0.3
            target_y = player['y'] + (goal_y - player['y']) * 0.2
        elif player['team'] == 'attacker':
            # Supporting attackers create space
            if primary_receiver_id:
                # Find primary receiver position
                primary_player = next((p for p in players if p['id'] == primary_receiver_id), None)
                if primary_player:
                    dx = (player['x'] - primary_player['x']) * 0.1
                    dy = (player['y'] - primary_player['y']) * 0.1
                    target_x = player['x'] + dx
                    target_y = player['y'] + dy
                else:
                    target_x, target_y = player['x'], player['y']
            else:
                target_x, target_y = player['x'], player['y']
        elif player['team'] == 'defender':
            # Defenders move to mark nearest attackers
            attackers = [p for p in players if p['team'] == 'attacker']
            if attackers:
                nearest_attacker = min(attackers, 
                    key=lambda a: math.sqrt((a['x'] - player['x'])**2 + (a['y'] - player['y'])**2))
                dx = (nearest_attacker['x'] - player['x']) * 0.7
                dy = (nearest_attacker['y'] - player['y']) * 0.7
                target_x = player['x'] + dx
                target_y = player['y'] + dy
            else:
                target_x, target_y = player['x'], player['y']
        else:  # keeper
            # Goalkeeper adjusts position
            target_x = player['x']
            target_y = player['y'] + (goal_y - player['y']) * 0.1
        
        # Convert to percentage coordinates
        target_x_pct = (target_x / 105) * 100
        target_y_pct = (target_y / 68) * 100
        
        movements.append({
            'playerId': player['id'],
            'startPos': {'x': start_x_pct, 'y': start_y_pct},
            'targetPos': {'x': target_x_pct, 'y': target_y_pct},
            'role': player['team'],
            'movementSpeed': 0.8 if player['team'] == 'attacker' else 0.6 if player['team'] == 'defender' else 0.4
        })
    
    return movements

def generate_fallback_simulation(players, corner_x, corner_y, goal_x, goal_y):
    """
    Generate fallback simulation when Strategy Maker is not available
    """
    # Find closest attacker to goal as primary receiver
    attackers = [p for p in players if p['team'] == 'attacker']
    if not attackers:
        return jsonify({'error': 'No attackers found'}), 400
    
    primary_receiver = min(attackers, 
        key=lambda p: math.sqrt((p['x'] - goal_x)**2 + (p['y'] - goal_y)**2))
    
    # Generate fallback strategy
    fallback_strategy = {
        'predictions': {
            'primary_receiver': {
                'player_id': primary_receiver['id'],
                'score': 0.75,
                'position': {'x': primary_receiver['x'], 'y': primary_receiver['y']}
            },
            'shot_confidence': 0.6,
            'tactical_decision': 'Direct shot attempt',
            'alternate_receivers': []
        }
    }
    
    # Calculate trajectory and movements
    ball_trajectory = calculate_ball_trajectory(corner_x, corner_y, fallback_strategy['predictions']['primary_receiver'], goal_x, goal_y)
    player_movements = calculate_player_movements(players, fallback_strategy['predictions']['primary_receiver'], goal_x, goal_y)
    
    return jsonify({
        'success': True,
        'setPiece': 'Corner Kick',
        'team': 'Selected Team',
        'totalPlayers': len(players),
        'positions': players,
        'prediction': {
            'primaryPlayer': f"{get_player_label(players, primary_receiver['id'])} (75%)",
            'shotConfidence': 60,
            'tacticalDecision': 'Direct shot attempt',
            'alternatives': [],
            'successRate': 60
        },
        'simulation': {
            'ballTrajectory': ball_trajectory,
            'playerMovements': player_movements,
            'cornerPosition': {'x': corner_x, 'y': corner_y},
            'goalPosition': {'x': goal_x, 'y': goal_y},
            'primaryReceiver': fallback_strategy['predictions']['primary_receiver'],
            'shotAction': True,
            'shotTarget': {'x': goal_x, 'y': goal_y}
        }
    })

def initialize_strategy_maker():
    """Initialize the strategy maker on server startup"""
    global strategy_maker
    try:
        print("Initializing Strategy Maker...")
        strategy_maker = StrategyMaker()
        print("✅ Strategy Maker initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️  Could not initialize Strategy Maker: {e}")
        traceback.print_exc()
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'strategy_maker_ready': strategy_maker is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/optimize', methods=['POST'])
def optimize_strategy():
    """
    Main optimization endpoint
    Expects JSON body:
    {
        "team": "Team A",
        "setPiece": "Corner Kick",
        "players": [
            {"id": 1, "x": 95, "y": 30, "team": "attacker"},
            {"id": 2, "x": 90, "y": 34, "team": "defender"},
            ...
        ],
        "cornerPosition": {"x": 105, "y": 0}
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'players' not in data:
            return jsonify({'error': 'Missing players data'}), 400
        
        players = data['players']
        corner_position = data.get('cornerPosition', {'x': 105, 'y': 0})
        corner_pos_tuple = (corner_position['x'], corner_position['y'])
        
        # Convert frontend player format to backend format
        backend_players = []
        for p in players:
            backend_players.append({
                'id': p['id'],
                'x': p['x'],
                'y': p['y'],
                'team': p.get('team', 'attacker')
            })
        
        print(f"\n{'='*70}")
        print(f"  API OPTIMIZATION REQUEST")
        print(f"  Players: {len(backend_players)} | Corner: {corner_pos_tuple}")
        print(f"{'='*70}\n")
        
        # Generate strategy using GNN
        if strategy_maker:
            strategy = strategy_maker.predict_strategy(
                backend_players, 
                corner_position=corner_pos_tuple
            )
            
            # Save strategy
            strategy_path = strategy_maker.save_strategy(strategy)
            
            # Format response for frontend
            response = {
                'success': True,
                'strategy': {
                    'primaryReceiver': {
                        'playerId': strategy['predictions']['primary_receiver']['player_id'],
                        'score': strategy['predictions']['primary_receiver']['score'],
                        'position': strategy['predictions']['primary_receiver']['position']
                    },
                    'alternateReceivers': [
                        {
                            'playerId': alt['player_id'],
                            'score': alt['score'],
                            'position': alt['position']
                        }
                        for alt in strategy['predictions']['alternate_receivers'][:3]
                    ],
                    'shotConfidence': strategy['predictions']['shot_confidence'],
                    'tacticalDecision': strategy['predictions']['tactical_decision'],
                    'successRate': int(strategy['predictions']['shot_confidence'] * 100)
                },
                'players': [
                    {
                        'id': p['id'],
                        'name': f"Player {p['id']}",
                        'role': p['team'].capitalize(),
                        'position': {'x': p['x'], 'y': p['y']},
                        'isPrimary': p['id'] == strategy['predictions']['primary_receiver']['player_id'],
                        'isAlternate': p['id'] in [alt['player_id'] for alt in strategy['predictions']['alternate_receivers'][:3]]
                    }
                    for p in backend_players
                ],
                'optimizationInsights': strategy.get('debug_info', {}).get('decision_reason', 'Strategy optimized based on GNN predictions'),
                'strategyFile': os.path.basename(strategy_path),
                'timestamp': strategy['timestamp']
            }
            
            print(f"✅ Strategy generated successfully")
            print(f"   Primary: Player #{response['strategy']['primaryReceiver']['playerId']}")
            print(f"   Success Rate: {response['strategy']['successRate']}%")
            
            return jsonify(response)
        else:
            # Fallback if Strategy Maker not available
            return jsonify({
                'error': 'Strategy Maker not initialized',
                'fallback': True
            }), 503
            
    except Exception as e:
        print(f"❌ Error in optimize_strategy: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_corner():
    """
    Generate complete simulation data for frontend animation
    Returns ball trajectory, player movements, and tactical predictions
    """
    try:
        data = request.get_json()
        
        # DETAILED LOGGING FOR API DATA FLOW VERIFICATION
        print("\n🔍 ===== BACKEND API REQUEST RECEIVED =====")
        print(f"📥 Raw JSON Data: {data}")
        print(f"📋 Request Headers: {dict(request.headers)}")
        print(f"🌐 Client IP: {request.remote_addr}")
        
        # Extract data from frontend
        players = data.get('players', [])
        corner_position = data.get('cornerPosition', {'x': 95, 'y': 5})  # Frontend uses percentage
        goal_position = data.get('goalPosition', {'x': 95, 'y': 50})
        set_piece = data.get('setPiece', 'Corner Kick')
        
        print(f"📊 Extracted Data:")
        print(f"  Players Count: {len(players)}")
        print(f"  Corner Position: {corner_position}")
        print(f"  Goal Position: {goal_position}")
        print(f"  Set Piece: {set_piece}")
        
        print(f"👥 Frontend Player Data:")
        for i, player in enumerate(players):
            print(f"  {i+1}. {player.get('label', 'Unknown')} (ID: {player.get('id', 'N/A')})")
            print(f"     Role: {player.get('role', 'N/A')}")
            print(f"     Position: ({player.get('xPct', 'N/A')}%, {player.get('yPct', 'N/A')}%)")
        
        if not players:
            return jsonify({'error': 'No players provided'}), 400
        
        # Convert frontend coordinates to backend format
        # Frontend uses percentage (0-100), backend uses meters (0-105, 0-68)
        print(f"\n🔄 ===== COORDINATE CONVERSION =====")
        print(f"📐 Conversion Formula:")
        print(f"  Frontend: Percentage (0-100)")
        print(f"  Backend: Meters (0-105, 0-68)")
        print(f"  X: frontend_xPct * 105 / 100")
        print(f"  Y: frontend_yPct * 68 / 100")
        
        backend_players = []
        for i, p in enumerate(players):
            # Convert percentage to meters
            x_meters = (p['xPct'] / 100) * 105  # Frontend xPct to backend x
            y_meters = (p['yPct'] / 100) * 68   # Frontend yPct to backend y
            
            # Map frontend roles to backend roles
            role_mapping = {
                'red': 'attacker',
                'blue': 'defender', 
                'gk': 'keeper'
            }
            
            backend_player = {
                'id': p['id'],
                'x': x_meters,
                'y': y_meters,
                'team': role_mapping.get(p['role'], 'attacker'),
                'label': p.get('label', f"Player{p['id']}")
            }
            
            backend_players.append(backend_player)
            
            print(f"  {i+1}. {p.get('label', 'Unknown')}:")
            print(f"     Frontend: ({p['xPct']}%, {p['yPct']}%)")
            print(f"     Backend: ({x_meters:.2f}m, {y_meters:.2f}m)")
            print(f"     Role: {p['role']} → {backend_player['team']}")
        
        print(f"✅ Converted {len(backend_players)} players")
        
        # Convert corner and goal positions
        corner_x = (corner_position['x'] / 100) * 105
        corner_y = (corner_position['y'] / 100) * 68
        goal_x = (goal_position['x'] / 100) * 105
        goal_y = (goal_position['y'] / 100) * 68
        
        print(f"\n🎯 ===== POSITION CONVERSION =====")
        print(f"   Corner: Frontend ({corner_position['x']}%, {corner_position['y']}%) → Backend ({corner_x:.1f}m, {corner_y:.1f}m)")
        print(f"   Goal: Frontend ({goal_position['x']}%, {goal_position['y']}%) → Backend ({goal_x:.1f}m, {goal_y:.1f}m)")
        print(f"   Total Players: {len(backend_players)}")
        
        if strategy_maker:
            print(f"\n🧠 ===== STRATEGY GENERATION =====")
            print(f"📡 Calling StrategyMaker.predict_strategy()...")
            # Generate strategy using backend
            strategy = strategy_maker.predict_strategy(
                backend_players,
                corner_position=(corner_x, corner_y)
            )
            
            # Validate strategy data
            if not strategy or 'predictions' not in strategy:
                print("⚠️  Strategy Maker returned invalid strategy, using fallback")
                return generate_fallback_simulation(backend_players, corner_x, corner_y, goal_x, goal_y)
            
            # Validate primary receiver data
            primary_receiver = strategy['predictions'].get('primary_receiver')
            if not primary_receiver or not primary_receiver.get('player_id'):
                print("⚠️  Strategy Maker returned invalid primary receiver, using fallback")
                return generate_fallback_simulation(backend_players, corner_x, corner_y, goal_x, goal_y)
            
            # Generate simulation data
            sim_data = strategy_maker.generate_simulation_data(strategy)
            
            # Handle case where sim_data is None
            if sim_data is None:
                print("⚠️  Strategy Maker returned None for simulation data, using defaults")
                sim_data = {
                    'shot_action': True,
                    'shot_target': {'x': goal_x, 'y': goal_y}
                }
            
            # Calculate ball trajectory points for frontend animation
            ball_trajectory = calculate_ball_trajectory(
                corner_x, corner_y,
                strategy['predictions']['primary_receiver'],
                goal_x, goal_y
            )
            
            # Calculate player movements
            player_movements = calculate_player_movements(
                backend_players, 
                strategy['predictions']['primary_receiver'],
                goal_x, goal_y
            )
            
            # Prepare response data
            response_data = {
                'success': True,
                'setPiece': set_piece,
                'team': 'Selected Team',
                'totalPlayers': len(backend_players),
                'positions': backend_players,
                'prediction': {
                    'primaryPlayer': f"{get_player_label(backend_players, strategy['predictions']['primary_receiver']['player_id'])} ({int(strategy['predictions']['primary_receiver']['score'] * 100)}%)",
                    'shotConfidence': int(strategy['predictions']['shot_confidence'] * 100),
                    'tacticalDecision': strategy['predictions']['tactical_decision'],
                    'alternatives': [
                        {
                            'player': get_player_label(backend_players, alt['player_id']),
                            'percentage': int(alt['score'] * 100)
                        }
                        for alt in strategy['predictions'].get('alternate_receivers', [])[:3]
                    ],
                    'successRate': int(strategy['predictions']['shot_confidence'] * 100)
                },
                'simulation': {
                    'ballTrajectory': ball_trajectory,
                    'playerMovements': player_movements,
                    'cornerPosition': {'x': corner_x, 'y': corner_y},
                    'goalPosition': {'x': goal_x, 'y': goal_y},
                    'primaryReceiver': strategy['predictions']['primary_receiver'],
                    'shotAction': sim_data.get('shot_action', True),
                    'shotTarget': sim_data.get('shot_target', {'x': goal_x, 'y': goal_y})
                }
            }
            
            print(f"\n📤 ===== API RESPONSE PREPARATION =====")
            print(f"✅ Strategy Generation Complete!")
            print(f"🎯 Primary Receiver: {get_player_label(backend_players, strategy['predictions']['primary_receiver']['player_id'])}")
            print(f"📊 Shot Confidence: {int(strategy['predictions']['shot_confidence'] * 100)}%")
            print(f"🎲 Tactical Decision: {strategy['predictions']['tactical_decision']}")
            print(f"🎯 Ball Trajectory Points: {len(ball_trajectory['points'])}")
            print(f"👥 Player Movements: {len(player_movements)}")
            
            print(f"\n📋 Response Data Structure:")
            print(f"  - success: {response_data['success']}")
            print(f"  - setPiece: {response_data['setPiece']}")
            print(f"  - totalPlayers: {response_data['totalPlayers']}")
            print(f"  - prediction.primaryPlayer: {response_data['prediction']['primaryPlayer']}")
            print(f"  - prediction.shotConfidence: {response_data['prediction']['shotConfidence']}%")
            print(f"  - simulation.ballTrajectory.points: {len(response_data['simulation']['ballTrajectory']['points'])}")
            print(f"  - simulation.playerMovements: {len(response_data['simulation']['playerMovements'])}")
            
            print(f"\n🚀 Sending response to frontend...")
            print(f"==========================================\n")
            
            return jsonify(response_data)
            
        else:
            # Fallback simulation when Strategy Maker not available
            print("⚠️  Strategy Maker not available, using fallback simulation")
            return generate_fallback_simulation(backend_players, corner_x, corner_y, goal_x, goal_y)
            
    except Exception as e:
        print(f"❌ Error in simulate_corner: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies', methods=['GET'])
def list_strategies():
    """List all saved strategies"""
    try:
        strategy_files = [f for f in os.listdir('.') if f.startswith('corner_strategy_') and f.endswith('.json')]
        strategies = []
        
        for filename in sorted(strategy_files, reverse=True)[:10]:  # Last 10
            with open(filename, 'r') as f:
                strategy = json.load(f)
                strategies.append({
                    'filename': filename,
                    'timestamp': strategy.get('timestamp', ''),
                    'primaryReceiver': strategy['predictions']['primary_receiver']['player_id'],
                    'shotConfidence': strategy['predictions']['shot_confidence']
                })
        
        return jsonify({'strategies': strategies})
    except Exception as e:
        print(f"❌ Error in list_strategies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategy/<filename>', methods=['GET'])
def get_strategy(filename):
    """Get a specific strategy file"""
    try:
        if not filename.endswith('.json'):
            filename += '.json'
        
        with open(filename, 'r') as f:
            strategy = json.load(f)
        
        return jsonify(strategy)
    except FileNotFoundError:
        return jsonify({'error': 'Strategy not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  CORNER KICK TACTICAL OPTIMIZER - API SERVER")
    print("="*70 + "\n")
    
    # Initialize Strategy Maker
    if initialize_strategy_maker():
        print("🚀 Starting Flask API server on http://localhost:5000")
        print("📡 CORS enabled for React frontend")
        print("\nAvailable endpoints:")
        print("  GET  /api/health          - Health check")
        print("  POST /api/optimize        - Optimize strategy")
        print("  POST /api/simulate        - Generate simulation")
        print("  GET  /api/strategies      - List saved strategies")
        print("  GET  /api/strategy/<id>   - Get specific strategy")
        print("\n" + "="*70 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize Strategy Maker")
        print("   Server will not start")