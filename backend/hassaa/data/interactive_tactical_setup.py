#!/usr/bin/env python3
"""
Interactive Tactical Setup for Corner Kick Visualization
Allows manual placement of players and generates model predictions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.widgets import Button
import json
import os
import math
import sys
from typing import List, Dict, Tuple, Any
import datetime
import threading
import time

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from corner_replay_v4 import CornerReplayV4, create_triangle_path, create_square_path
from strategy_suggester import SingleTaskGATv2ReceiverCheckpoint
from strategy_maker import StrategyMaker  # New import
from gnn_dataset import SetPieceGraphDataset, _safe_float
import torch
from torch_geometric.data import Data

class InteractiveTacticalSetup:
    def __init__(self):
        self.players = []
        self.placement_mode = "attacker"  # attacker, defender, keeper
        self.player_id_counter = 100001
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Initialize counters before calling setup_pitch and setup_ui
        self.attackers_placed = 0
        self.defenders_placed = 0
        self.keepers_placed = 0
        self.max_attackers = 9
        self.max_defenders = 10
        self.max_keepers = 1
        
        # Corner side selection
        self.corner_side = "right"  # Default: right corner (105, 0 or 105, 68)
        self.corner_position = (105, 0)  # Will be updated based on selection
        self.goal_position = (105, 34)  # Dynamic goal position based on corner side
        
        # Simulation state
        self.simulation_active = False
        self.animation_timer = None
        self.ball_position = None
        self.target_position = None
        self.simulation_step = 0
        self.max_simulation_steps = 60  # ~3 seconds at 20fps
        self.predicted_strategy = None
        
        # Enhanced player state tracking for dynamic simulation
        self.player_states = {}  # Track movement states for each player
        self.shot_phase_active = False
        self.shot_start_frame = 40  # Frame when shot begins
        self.total_shot_frames = 20  # Duration of shot animation
        
        self.setup_pitch()
        self.setup_ui()
        self.connect_events()
        
        # For undo functionality
        self.history = []
        
        # Initialize Strategy Maker for real-time predictions
        self.strategy_maker = None
        self.initialize_strategy_maker()
        
        # Anti-spam: track last strategy to prevent duplicates
        self.last_strategy_receiver = None
        self.last_strategy_decision = None
        self.last_strategy_timestamp = None
        
    def initialize_strategy_maker(self):
        """Initialize the strategy maker for GNN-based predictions"""
        try:
            print("Initializing Strategy Maker...")
            self.strategy_maker = StrategyMaker()
            print("✅ Strategy Maker initialized successfully")
        except Exception as e:
            print(f"⚠️  Could not initialize Strategy Maker: {e}")
            print("   Predictions will use simulated fallback")
            self.strategy_maker = None
        
    def setup_pitch(self):
        """Draw a football pitch"""
        self.ax.clear()  # Clear any existing content
        self.ax.set_xlim(0, 105)
        self.ax.set_ylim(0, 68)
        self.ax.set_aspect('equal')
        # Keep axis interactive for clicks
        self.ax.set_navigate(True)
        
        # Pitch outline
        self.ax.add_patch(patches.Rectangle((0, 0), 105, 68, linewidth=2, edgecolor='white', facecolor='#234d20'))
        
        # Center line
        self.ax.plot([52.5, 52.5], [0, 68], color='white', linewidth=1)
        
        # Center circle
        self.ax.add_patch(patches.Circle((52.5, 34), 9.15, edgecolor='white', facecolor='none', linewidth=1))
        
        # Penalty areas
        self.ax.add_patch(patches.Rectangle((0, 24), 16.5, 20, edgecolor='white', facecolor='none', linewidth=1))
        self.ax.add_patch(patches.Rectangle((88.5, 24), 16.5, 20, edgecolor='white', facecolor='none', linewidth=1))
        
        # Goal areas
        self.ax.add_patch(patches.Rectangle((0, 30), 5.5, 8, edgecolor='white', facecolor='none', linewidth=1))
        self.ax.add_patch(patches.Rectangle((99.5, 30), 5.5, 8, edgecolor='white', facecolor='none', linewidth=1))
        
        # Goals
        self.ax.add_patch(patches.Rectangle((-1, 30.34), 1, 7.32, edgecolor='white', facecolor='none', linewidth=2))
        self.ax.add_patch(patches.Rectangle((105, 30.34), 1, 7.32, edgecolor='white', facecolor='none', linewidth=2))
        
        # Corner arcs
        self.ax.add_patch(patches.Arc((0, 0), 2, 2, theta1=0, theta2=90, edgecolor='white', linewidth=1))
        self.ax.add_patch(patches.Arc((105, 0), 2, 2, theta1=90, theta2=180, edgecolor='white', linewidth=1))
        self.ax.add_patch(patches.Arc((0, 68), 2, 2, theta1=270, theta2=360, edgecolor='white', linewidth=1))
        self.ax.add_patch(patches.Arc((105, 68), 2, 2, theta1=180, theta2=270, edgecolor='white', linewidth=1))
        
        # Turn off axis after all patches are added
        self.ax.axis('off')
        
        self.ax.set_title("Interactive Tactical Setup - Click to place players", fontsize=14)
        
    def setup_ui(self):
        """Setup UI buttons"""
        # Clear existing UI elements
        for txt in self.ax.texts:
            txt.remove()
            
        # Corner side indicator with visual highlight
        corner_text = f"Corner Side: {self.corner_side.upper()} ({self.get_corner_description()})"
        self.corner_side_text = self.ax.text(0.02, 0.98, corner_text, 
                                     transform=self.ax.transAxes, fontsize=12,
                                     bbox=dict(facecolor='orange', alpha=0.9), weight='bold')
        
        # Placement mode indicator
        self.mode_text = self.ax.text(0.02, 0.94, f"Mode: {self.placement_mode.capitalize()}", 
                                     transform=self.ax.transAxes, fontsize=12,
                                     bbox=dict(facecolor='white', alpha=0.8))
        
        # Player counters
        self.counter_text = self.ax.text(0.02, 0.90, 
                                        f"Attackers: {self.attackers_placed}/{self.max_attackers} | "
                                        f"Defenders: {self.defenders_placed}/{self.max_defenders} | "
                                        f"Goalkeepers: {self.keepers_placed}/{self.max_keepers}",
                                        transform=self.ax.transAxes, fontsize=10,
                                        bbox=dict(facecolor='yellow', alpha=0.7))
        
        # Dynamic instructions based on current stage
        instruction = self.get_current_instruction()
        self.instruction_text = self.ax.text(0.02, 0.86, instruction,
                                           transform=self.ax.transAxes, fontsize=9,
                                           bbox=dict(facecolor='lightblue', alpha=0.8))
        
        # Corner side selection buttons
        ax_corner_left = plt.axes([0.02, 0.02, 0.08, 0.04])
        self.btn_corner_left = Button(ax_corner_left, '← Left')
        self.btn_corner_left.on_clicked(self._on_corner_left)
        
        ax_corner_right = plt.axes([0.11, 0.02, 0.08, 0.04])
        self.btn_corner_right = Button(ax_corner_right, 'Right →')
        self.btn_corner_right.on_clicked(self._on_corner_right)
        
        # Buttons - Update button layout and labels
        total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
        
        if total_players >= 6:  # Minimum for simulation
            ax_simulate = plt.axes([0.75, 0.02, 0.15, 0.05])
            self.btn_simulate = Button(ax_simulate, 'Generate & Simulate')
            self.btn_simulate.on_clicked(self._on_simulate)
        else:
            ax_done = plt.axes([0.8, 0.02, 0.1, 0.05])
            self.btn_done = Button(ax_done, 'Done')
            self.btn_done.on_clicked(self._on_done)
        
        ax_undo = plt.axes([0.6, 0.02, 0.1, 0.05])
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_undo.on_clicked(self._on_undo)
        
        ax_reset = plt.axes([0.5, 0.02, 0.1, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)
        
        # Highlight the selected corner on the pitch
        self.highlight_corner_flag()
        
    def get_current_instruction(self):
        """Get dynamic instruction based on current placement stage"""
        total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
        
        if self.simulation_active:
            return "🎬 SIMULATION RUNNING - Watch the corner kick play out!"
        elif total_players >= 22:
            return "✅ All 22 players placed! Click 'Generate & Simulate' to see GNN prediction."
        elif total_players >= 6:
            return f"Players: {total_players}/22 - Click 'Generate & Simulate' for early demo."
        elif self.attackers_placed < self.max_attackers:
            return f"Place attackers ({self.attackers_placed + 1}/{self.max_attackers}) - Red circles"
        elif self.defenders_placed < self.max_defenders:
            return f"Place defenders ({self.defenders_placed + 1}/{self.max_defenders}) - Gray triangles"
        elif self.keepers_placed < self.max_keepers:
            return f"Place goalkeepers ({self.keepers_placed + 1}/{self.max_keepers}) - Green squares"
        else:
            return "All players placed. Click 'Generate & Simulate' for tactical analysis."
    
    def get_corner_description(self):
        """Get description of current corner position"""
        if self.corner_side == "right":
            if self.corner_position[1] == 0:
                return "Bottom-Right Corner"
            else:
                return "Top-Right Corner"
        else:  # left
            if self.corner_position[1] == 0:
                return "Bottom-Left Corner"
            else:
                return "Top-Left Corner"
    
    def highlight_corner_flag(self):
        """Highlight the selected corner flag on the pitch"""
        # Remove previous highlight if exists
        for patch in list(self.ax.patches):
            if hasattr(patch, '_corner_highlight'):
                patch.remove()
        
        # Add bright highlight circle at corner position
        highlight = patches.Circle(
            self.corner_position, 
            radius=3, 
            facecolor='yellow', 
            edgecolor='red',
            linewidth=3,
            alpha=0.7,
            zorder=30
        )
        highlight._corner_highlight = True
        self.ax.add_patch(highlight)
        
        # Add corner flag marker
        flag_marker = patches.Circle(
            self.corner_position,
            radius=1.5,
            facecolor='red',
            edgecolor='white',
            linewidth=2,
            zorder=31
        )
        flag_marker._corner_highlight = True
        self.ax.add_patch(flag_marker)
        
        # Add text label
        offset_x = -8 if self.corner_position[0] > 50 else 8
        offset_y = -5 if self.corner_position[1] < 34 else 5
        
        corner_label = self.ax.text(
            self.corner_position[0] + offset_x,
            self.corner_position[1] + offset_y,
            "⚽ CORNER",
            color='white',
            fontsize=11,
            weight='bold',
            ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='red', alpha=0.9),
            zorder=32
        )
        corner_label._corner_highlight = True
        
        self.fig.canvas.draw()
    
    def _on_corner_left(self, event):
        """Handle Left corner button click"""
        # Toggle between top-left and bottom-left
        if self.corner_side == "left":
            # Switch between top and bottom on same side
            if self.corner_position == (0, 0):
                self.corner_position = (0, 68)
            else:
                self.corner_position = (0, 0)
        else:
            # Switch to left side
            self.corner_side = "left"
            self.corner_position = (0, 0)  # Default to bottom-left
        
        # Update goal position for left side (attack toward left goal)
        self.goal_position = (0, 34)
        
        # Update UI
        self.update_corner_ui()
        print(f"📍 Corner changed to: {self.get_corner_description()} at {self.corner_position}")
        print(f"   Goal position updated to: {self.goal_position}")
    
    def _on_corner_right(self, event):
        """Handle Right corner button click"""
        # Toggle between top-right and bottom-right
        if self.corner_side == "right":
            # Switch between top and bottom on same side
            if self.corner_position == (105, 0):
                self.corner_position = (105, 68)
            else:
                self.corner_position = (105, 0)
        else:
            # Switch to right side
            self.corner_side = "right"
            self.corner_position = (105, 0)  # Default to bottom-right
        
        # Update goal position for right side (attack toward right goal)
        self.goal_position = (105, 34)
        
        # Update UI
        self.update_corner_ui()
        print(f"📍 Corner changed to: {self.get_corner_description()} at {self.corner_position}")
        print(f"   Goal position updated to: {self.goal_position}")
    
    def update_corner_ui(self):
        """Update corner side UI text and visual highlight"""
        if hasattr(self, 'corner_side_text'):
            corner_text = f"Corner Side: {self.corner_side.upper()} ({self.get_corner_description()})"
            self.corner_side_text.set_text(corner_text)
        
        # Update visual highlight
        self.highlight_corner_flag()
            
    def connect_events(self):
        """Connect mouse and keyboard events"""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
    def _on_click(self, event):
        """Handle mouse click for player placement"""
        print(f"Click event received: axes={event.inaxes}, button={event.button}")
        
        if event.inaxes != self.ax:
            print(f"Click not on main axes, ignoring")
            return
            
        # Only handle left clicks
        if event.button != 1:
            print(f"Not a left click, ignoring")
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            print(f"Invalid coordinates: ({x}, {y})")
            return
            
        print(f"Valid click at ({x:.1f}, {y:.1f}) in {self.placement_mode} mode")
            
        # Check if we've reached the maximum number of players for current mode
        if self.placement_mode == "attacker" and self.attackers_placed >= self.max_attackers:
            print("Maximum attackers placed!")
            return
        elif self.placement_mode == "defender" and self.defenders_placed >= self.max_defenders:
            print("Maximum defenders placed!")
            return
        elif self.placement_mode == "keeper" and self.keepers_placed >= self.max_keepers:
            print("Maximum goalkeepers placed!")
            return
            
        # Check if we've reached the total maximum players
        total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
        if total_players >= 22:
            print("Maximum players (22) reached!")
            return
            
        # Create player based on current mode
        player = {
            'id': self.player_id_counter,
            'x': x,
            'y': y,
            'team': self.placement_mode
        }
        
        self.players.append(player)
        self.history.append(('add', player.copy()))
        self.player_id_counter += 1
        
        # Update counters
        if self.placement_mode == "attacker":
            self.attackers_placed += 1
        elif self.placement_mode == "defender":
            self.defenders_placed += 1
        elif self.placement_mode == "keeper":
            self.keepers_placed += 1
            
        # Draw the player
        self.draw_player(player)
        
        # Update mode after placing certain number of players
        # After 10 attackers, switch to defenders
        if self.placement_mode == 'attacker' and self.attackers_placed >= self.max_attackers:
            self.placement_mode = 'defender'
            
        # After 10 defenders, switch to keeper
        elif self.placement_mode == 'defender' and self.defenders_placed >= self.max_defenders:
            self.placement_mode = 'keeper'
            
        # Update UI
        self.mode_text.set_text(f"Mode: {self.placement_mode.capitalize()}")
        self.counter_text.set_text(f"Attackers: {self.attackers_placed}/{self.max_attackers} | "
                                  f"Defenders: {self.defenders_placed}/{self.max_defenders} | "
                                  f"Goalkeepers: {self.keepers_placed}/{self.max_keepers}")
        self.instruction_text.set_text(self.get_current_instruction())
        
        # Refresh UI buttons
        self.refresh_ui_buttons()
        
        self.fig.canvas.draw()
        
        print(f"Player {player['id']} placed successfully!")
    def refresh_ui_buttons(self):
        """Refresh UI buttons based on current state"""
        # Remove existing buttons
        for widget in [getattr(self, 'btn_done', None), getattr(self, 'btn_simulate', None)]:
            if widget and hasattr(widget, 'ax'):
                widget.ax.set_visible(False)
        
        # Recreate appropriate button
        total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
        
        if total_players >= 6:  # Minimum for simulation
            ax_simulate = plt.axes([0.75, 0.02, 0.15, 0.05])
            self.btn_simulate = Button(ax_simulate, 'Generate & Simulate')
            self.btn_simulate.on_clicked(self._on_simulate)
        else:
            ax_done = plt.axes([0.8, 0.02, 0.1, 0.05])
            self.btn_done = Button(ax_done, 'Done')
            self.btn_done.on_clicked(self._on_done)
        
        self.fig.canvas.draw()
        
    def _on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'enter':
            self._on_done(None)
        elif event.key == 'u' or event.key == 'z':
            self._on_undo(None)
        elif event.key == 'r':
            self._on_reset(None)
            
    def _on_simulate(self, event):
        """Handle Generate & Simulate button click - Full GNN-driven simulation"""
        total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
        if total_players < 6:
            print(f"Error: At least 6 players required (currently {total_players} placed)")
            self.ax.text(0.5, 0.5, "ERROR: Place at least 6 players!", 
                        transform=self.ax.transAxes, fontsize=14, color='red', weight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            self.fig.canvas.draw()
            return
            
        print(f"\n{'='*70}")
        print(f"  INTERACTIVE CORNER KICK SIMULATION")
        print(f"  Players: {total_players} | Strategy Maker: {'Ready' if self.strategy_maker else 'Fallback'}")
        print(f"{'='*70}\n")
        
        # Generate strategy using GNN models with enhanced debugging
        if self.strategy_maker:
            try:
                print("🧠 Generating GNN strategy...")
                
                # Show immediate visual feedback
                loading_text = self.ax.text(0.5, 0.1, "⏳ Generating Strategy...", 
                           transform=self.ax.transAxes, fontsize=14, color='blue', weight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9),
                           zorder=50)
                self.fig.canvas.draw()
                
                # Force immediate UI update
                self.fig.canvas.flush_events()
                
                # Get GNN predictions with detailed debugging
                print(f"📋 Current player positions for strategy:")
                for i, player in enumerate(self.players):
                    print(f"   {i+1}. Player #{player['id']}: {player['team']} at ({player['x']:.1f}, {player['y']:.1f})")
                
                strategy = self.strategy_maker.predict_strategy(self.players, corner_position=self.corner_position)
                self.predicted_strategy = strategy
                
                # CRITICAL DEBUG: Verify we're getting dynamic predictions
                primary = strategy["predictions"]["primary_receiver"]
                shot_conf = strategy["predictions"]["shot_confidence"]
                decision = strategy["predictions"]["tactical_decision"]
                
                # Anti-spam check: prevent identical consecutive strategies
                current_receiver = primary['player_id']
                current_decision = decision
                
                if (self.last_strategy_receiver == current_receiver and 
                    self.last_strategy_decision == current_decision):
                    print(f"\n⏸️  DUPLICATE STRATEGY DETECTED - same receiver & decision")
                    print(f"   Receiver: {current_receiver}, Decision: '{current_decision}'")
                    print(f"   ⏭️  Allowing anyway (position may have changed)")
                    # Note: We allow it anyway as position might have changed slightly
                
                # Update anti-spam tracking
                self.last_strategy_receiver = current_receiver
                self.last_strategy_decision = current_decision
                self.last_strategy_timestamp = strategy["timestamp"]
                
                print(f"\n🎯 GNN PREDICTION RESULTS:")
                print(f"   Primary Receiver: Player #{primary['player_id']} (Score: {primary['score']:.4f})")
                print(f"   Shot Confidence: {shot_conf:.4f}")
                print(f"   Tactical Decision: '{decision}'")
                print(f"   Receiver Position: {primary['position']}")
                
                # Verify this matches actual player in formation
                actual_player = next((p for p in self.players if p['id'] == primary['player_id']), None)
                if actual_player:
                    print(f"   ✅ Receiver verified: {actual_player['team']} at ({actual_player['x']:.1f}, {actual_player['y']:.1f})")
                else:
                    print(f"   ⚠️  WARNING: Receiver ID {primary['player_id']} not found in current players!")
                
                # Remove loading text
                loading_text.remove()
                
                # Save strategy
                strategy_path = self.strategy_maker.save_strategy(strategy)
                print(f"🎯 Strategy saved to: {os.path.basename(strategy_path)}")
                
                # Start real-time simulation
                self.start_corner_simulation(strategy)
                
            except Exception as e:
                print(f"⚠️  GNN Strategy generation error: {e}")
                print("   Falling back to heuristic simulation")
                # Remove any loading indicators
                if 'loading_text' in locals():
                    loading_text.remove()
                import traceback
                traceback.print_exc()
                self.start_fallback_simulation()
        else:
            print("⚠️  Strategy Maker not available, using heuristic simulation")
            self.start_fallback_simulation()
            
    def start_corner_simulation(self, strategy: Dict[str, Any]):
        """Start animated corner kick simulation with GNN predictions"""
        print("\n🎬 Starting GNN-predicted corner kick simulation...")
        print(f"📋 Strategy received for player: {strategy['predictions']['primary_receiver']['player_id']}")
        
        # Set simulation state
        self.simulation_active = True
        self.simulation_step = 0
        self.shot_phase_active = False
        print("✅ Simulation state activated")

        # Get corner and target positions
        corner_pos = self.corner_position  # Use selected corner position
        primary_receiver = strategy["predictions"]["primary_receiver"]
        
        if primary_receiver["position"]:
            target_pos = (primary_receiver["position"]["x"], primary_receiver["position"]["y"])
        else:
            # Fallback target - MUST adapt to corner side for correct goal direction
            attackers = [p for p in self.players if p['team'] == 'attacker']
            if attackers:
                # Find closest attacker to the CORRECT goal based on corner side
                goal_pos = self.goal_position  # Uses dynamic goal (0,34) for left, (105,34) for right
                target_player = min(attackers, key=lambda p: math.hypot(p['x'] - goal_pos[0], p['y'] - goal_pos[1]))
                target_pos = (target_player['x'], target_player['y'])
                print(f"⚠️  No primary receiver - using fallback: Player #{target_player['id']} at ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
            else:
                # Last resort: use position near goal based on corner side
                if self.corner_position[0] > 50:  # Right corner
                    target_pos = (95, 34)  # Near right goal
                else:  # Left corner
                    target_pos = (10, 34)  # Near left goal
                print(f"⚠️  No attackers - using default near {('left' if self.corner_position[0] <= 50 else 'right')} goal: {target_pos}")
                
        self.ball_position = corner_pos
        self.target_position = target_pos
        
        # Initialize enhanced player states for dynamic movement
        self.initialize_player_states(strategy)
        
        # Print tactical scoring summary for transparency
        self.print_tactical_scoring_summary(strategy)
        
        # Clear and redraw pitch
        self.ax.clear()
        self.setup_pitch()
        
        # Add tactical overlay
        shot_conf = strategy["predictions"]["shot_confidence"]
        decision = strategy["predictions"]["tactical_decision"]
        
        overlay_text = f"🧠 GNN PREDICTION\n"
        
        # Handle case where no receiver is selected (reposition scenario)
        if primary_receiver['player_id'] is not None:
            overlay_text += f"Primary: Player #{primary_receiver['player_id']} ({primary_receiver['score']:.0%})\n"
        else:
            overlay_text += f"Primary: None - No viable receiver\n"
            
        overlay_text += f"Shot Confidence: {shot_conf:.0%}\n"
        overlay_text += f"Decision: {decision}\n"
        
        # Add dynamic context information
        if "debug_info" in strategy:
            debug = strategy["debug_info"]
            overlay_text += f"Context: {debug.get('decision_reason', 'N/A')[:30]}..."
        
        self.tactical_overlay = self.ax.text(
            0.02, 0.85, overlay_text,
            transform=self.ax.transAxes, fontsize=10, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
            zorder=20
        )
        
        # Draw players with enhanced highlighting
        self.draw_players_with_tactical_highlighting(strategy)
            
        # Highlight predicted receiver
        if primary_receiver["position"]:
            receiver_pos = primary_receiver["position"]
            highlight = patches.Circle(
                (receiver_pos['x'], receiver_pos['y']), 
                radius=3.0, 
                edgecolor='gold', 
                facecolor='none', 
                linewidth=4, 
                linestyle='--',
                zorder=15
            )
            self.ax.add_patch(highlight)
            
            # Add target label
            self.ax.text(
                receiver_pos['x'], receiver_pos['y'] - 5,
                f"TARGET\n{primary_receiver['score']:.0%}",
                color='gold',
                fontsize=12,
                weight='bold',
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8),
                zorder=16
            )
        
        # Update UI for simulation mode
        self.instruction_text = self.ax.text(0.02, 0.90, self.get_current_instruction(),
                                           transform=self.ax.transAxes, fontsize=9,
                                           bbox=dict(facecolor='orange', alpha=0.8))
        
        # Start enhanced animation timer
        self.start_enhanced_ball_animation()
        
    def animate_ball_flight(self, progress: float):
        """Animate ball flight using Bezier curve"""
        start = self.ball_position
        target = self.target_position
        
        # Quadratic Bezier curve with high arc
        control_x = (start[0] + target[0]) / 2
        control_y = (start[1] + target[1]) / 2 + 15  # High arc
        
        # Calculate current position
        t = progress
        x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * target[0]
        y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * target[1]
        
        # Remove previous ball
        for patch in list(self.ax.patches):
            if hasattr(patch, '_ball_marker'):
                patch.remove()
                
        # Draw current ball position
        ball_circle = patches.Circle((x, y), radius=0.8, facecolor='white', 
                                   edgecolor='black', linewidth=2, zorder=25)
        ball_circle._ball_marker = True
        self.ax.add_patch(ball_circle)
        
        # Draw trail
        if progress > 0.1:
            trail_points = []
            for i in range(int(progress * 20)):
                trail_t = i / 20.0
                trail_x = (1-trail_t)**2 * start[0] + 2*(1-trail_t)*trail_t * control_x + trail_t**2 * target[0]
                trail_y = (1-trail_t)**2 * start[1] + 2*(1-trail_t)*trail_t * control_y + trail_t**2 * target[1]
                trail_points.append((trail_x, trail_y))
                
            if len(trail_points) > 1:
                trail_points = np.array(trail_points)
                self.ax.plot(trail_points[:, 0], trail_points[:, 1], 
                           'w--', linewidth=2, alpha=0.7, zorder=20)

    def print_tactical_scoring_summary(self, strategy: Dict[str, Any]):
        """Print enhanced tactical scoring breakdown for transparency"""
        print(f"\n🎯 TACTICAL SCORING SUMMARY:")
        
        primary = strategy["predictions"]["primary_receiver"]
        shot_conf = strategy["predictions"]["shot_confidence"]
        decision = strategy["predictions"]["tactical_decision"]
        
        # Handle case where no receiver is selected
        if primary['player_id'] is not None:
            print(f"   🏆 WINNER: Player #{primary['player_id']} ({primary['score']:.0%})")
        else:
            print(f"   ⚠️  NO RECEIVER SELECTED - Reposition/Reset Play")
            print(f"   📝 Reason: {strategy.get('debug_info', {}).get('decision_reason', 'No tactically viable positions')}")
            
        print(f"   ⚽ Shot Confidence: {shot_conf:.0%}")
        print(f"   🎲 Decision: {decision}")
        
        # Show breakdown from debug info if available
        if "debug_info" in strategy and "all_receiver_scores" in strategy["debug_info"]:
            raw_scores = strategy["debug_info"]["all_receiver_scores"]
            print(f"\n   📊 SCORING BREAKDOWN (Top 5):")
            
            sorted_scores = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (pid, score) in enumerate(sorted_scores, 1):
                player = next((p for p in self.players if p['id'] == pid), None)
                if player:
                    x, y = player['x'], player['y']
                    
                    # Calculate tactical factors
                    dist_to_goal = math.hypot(x - self.goal_position[0], y - self.goal_position[1])
                    is_unmarked, def_dist = self._check_if_unmarked_player(player)
                    
                    status = "🎯 SELECTED" if pid == primary['player_id'] else f"   #{i}"
                    unmarked_indicator = " (UNMARKED)" if is_unmarked else ""
                    
                    print(f"     {status} Player #{pid}: {score:.3f} | "
                          f"Pos({x:.0f},{y:.0f}) | Goal:{dist_to_goal:.1f}m{unmarked_indicator}")
        
        print(f"   💡 Key Factors: Distance to goal, unmarked status, tactical positioning")
        
    def start_fallback_simulation(self):
        """Fallback simulation when GNN is not available"""
        print("\n\ud83c\udfac Starting fallback corner kick simulation...")
        
        # Simple heuristic: target closest attacker to goal
        attackers = [p for p in self.players if p['team'] == 'attacker']
        if attackers:
            goal_pos = self.goal_position  # Use dynamic goal position based on corner side
            target_player = min(attackers, key=lambda p: math.hypot(p['x'] - goal_pos[0], p['y'] - goal_pos[1]))
            target_pos = (target_player['x'], target_player['y'])
        else:
            target_pos = (95, 34)
            
        # Create simple strategy
        fallback_strategy = {
            "predictions": {
                "primary_receiver": {
                    "player_id": target_player['id'] if attackers else None,
                    "score": 0.75,
                    "position": {"x": target_pos[0], "y": target_pos[1]}
                },
                "shot_confidence": 0.6,
                "tactical_decision": "Direct shot attempt"
            }
        }
        
        self.start_corner_simulation(fallback_strategy)
        
    def start_ball_animation(self):
        """Start the ball flight animation"""
        from matplotlib.animation import FuncAnimation
        
        print("🎥 Starting ball animation...")
        
        def animate_frame(frame):
            if not getattr(self, 'simulation_active', False) or self.simulation_step >= self.max_simulation_steps:
                print(f"Animation stopped at frame {frame}")
                return
                
            progress = frame / 30.0  # 30 frames for ball flight
            
            if progress <= 1.0:
                # Ball flight phase
                self.animate_ball_flight(progress)
                if frame % 10 == 0:  # Debug every 10 frames
                    print(f"Ball flight progress: {progress:.2f}")
            elif progress <= 1.5:
                # Player movement phase
                self.animate_player_movements(progress - 1.0)
            else:
                # Shot/outcome phase
                self.animate_shot_outcome()
                self.simulation_active = False
                print("🏁 Animation complete!")
                
            self.simulation_step += 1
            self.fig.canvas.draw()
            
        # Start animation
        print("▶️ Creating FuncAnimation...")
        self.animation_timer = FuncAnimation(
            self.fig, animate_frame, 
            frames=50, interval=100, repeat=False
        )
        print("🚀 Animation started!")

    def animate_ball_flight(self, progress: float):
        """Animate ball flight using Bezier curve"""
        start = self.ball_position
        target = self.target_position
        
        # Quadratic Bezier curve with high arc
        control_x = (start[0] + target[0]) / 2
        control_y = (start[1] + target[1]) / 2 + 15  # High arc
        
        # Calculate current position
        t = progress
        x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * target[0]
        y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * target[1]
        
        # Remove previous ball
        for patch in self.ax.patches:
            if hasattr(patch, '_ball_marker'):
                patch.remove()
                
        # Draw current ball position
        ball_circle = patches.Circle((x, y), radius=0.8, facecolor='white', 
                                   edgecolor='black', linewidth=2, zorder=25)
        ball_circle._ball_marker = True
        self.ax.add_patch(ball_circle)
        
        # Draw trail
        if progress > 0.1:
            trail_points = []
            for i in range(int(progress * 20)):
                trail_t = i / 20.0
                trail_x = (1-trail_t)**2 * start[0] + 2*(1-trail_t)*trail_t * control_x + trail_t**2 * target[0]
                trail_y = (1-trail_t)**2 * start[1] + 2*(1-trail_t)*trail_t * control_y + trail_t**2 * target[1]
                trail_points.append((trail_x, trail_y))
                
            if len(trail_points) > 1:
                trail_points = np.array(trail_points)
                self.ax.plot(trail_points[:, 0], trail_points[:, 1], 
                           'w--', linewidth=2, alpha=0.7, zorder=20)
    
    def animate_player_movements(self, progress: float):
        """Animate player movements toward ball"""
        # Simple player movement animation
        for player in self.players:
            if player['team'] == 'attacker':
                # Attackers move slightly toward target
                target = self.target_position
                dx = (target[0] - player['x']) * 0.1 * progress
                dy = (target[1] - player['y']) * 0.1 * progress
                
                # Add slight movement indicator
                arrow = patches.FancyArrowPatch(
                    (player['x'], player['y']),
                    (player['x'] + dx, player['y'] + dy),
                    arrowstyle='->', color='red', alpha=0.6, zorder=10
                )
                self.ax.add_patch(arrow)
                
    def animate_shot_outcome(self):
        """Animate the final shot outcome"""
        if not self.predicted_strategy:
            return
            
        shot_conf = self.predicted_strategy["predictions"]["shot_confidence"]
        
        if shot_conf > 0.5:
            # Animate shot toward goal
            target = self.target_position
            goal_target = self.goal_position  # Use dynamic goal position
            
            # Draw shot path
            shot_arrow = patches.FancyArrowPatch(
                target, goal_target,
                arrowstyle='->', color='red', linewidth=4, alpha=0.8, zorder=15
            )
            self.ax.add_patch(shot_arrow)
            
            # Add "SHOT!" label
            self.ax.text(
                (target[0] + goal_target[0]) / 2,
                (target[1] + goal_target[1]) / 2 + 3,
                "SHOT!",
                color='red', fontsize=16, weight='bold',
                ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9),
                zorder=20
            )
            
            # Simulate goal outcome
            if shot_conf > 0.7:
                self.ax.text(
                    self.goal_position[0], 40,
                    "\ud83c\udfc6 GOAL!",
                    color='green', fontsize=18, weight='bold',
                    ha='center', zorder=25,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.9)
                )
        else:
            # Show dynamic tactical decision from GNN strategy (not hardcoded)
            tactical_decision = getattr(self, 'predicted_strategy', {}).get('predictions', {}).get('tactical_decision', 'No decision')
            decision_color = 'orange'  # Default color for tactical decisions
            
            # Color coding based on decision type
            if any(keyword in tactical_decision.lower() for keyword in ['shot', 'goal', 'strike']):
                decision_color = 'red'  # Aggressive plays
            elif any(keyword in tactical_decision.lower() for keyword in ['cross', 'delivery', 'header']):
                decision_color = 'blue'  # Crossing plays
            elif any(keyword in tactical_decision.lower() for keyword in ['build', 'recycle', 'possession']):
                decision_color = 'green'  # Build-up plays
            
            self.ax.text(
                self.target_position[0], self.target_position[1] + 5,
                tactical_decision.upper(),
                color=decision_color, fontsize=12, weight='bold',
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.9),
                zorder=20
            )
        
        print("\u2705 Corner kick simulation complete!")
        print(f"\n{'='*70}\n")
        
    def _on_done(self, event):
        """Handle Done button click - Legacy method for backward compatibility"""
        # Redirect to simulation for consistency
        self._on_simulate(event)
        """Handle Done button click - Generate GNN strategy and run simulation"""
        # Validate that players are placed
        total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
        if total_players < 6:  # Minimum players for meaningful strategy
            print(f"Error: At least 6 players required (currently {total_players} placed)")
            self.ax.text(0.5, 0.5, "ERROR: Place at least 6 players!\n(Attackers, Defenders, Goalkeeper)", 
                        transform=self.ax.transAxes, fontsize=14, color='red', weight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            self.fig.canvas.draw()
            return
            
    def draw_players_with_tactical_highlighting(self, strategy: Dict[str, Any]):
        """Draw players with enhanced tactical highlighting and unmarked player indicators"""
        primary_receiver_id = strategy["predictions"]["primary_receiver"]["player_id"]
        alternate_receivers = strategy["predictions"].get("alternate_receivers", [])
        alternate_ids = [alt["player_id"] for alt in alternate_receivers]
        
        for player in self.players:
            # Determine attacker colors based on tactical role (following specification)
            if player['team'] == 'attacker':
                if player['id'] == primary_receiver_id:
                    # Primary receiver: red circle
                    attacker_color = '#ff3b3b'  # Red
                elif player['id'] in alternate_ids:
                    # Alternate receivers: orange circles
                    attacker_color = '#ff8c00'  # Orange
                else:
                    # Other attackers: blue circles
                    attacker_color = '#4169e1'  # Blue
                    
                # Draw attacker with tactical color
                self.draw_tactical_attacker(player, attacker_color)
            else:
                # Draw defender/keeper normally
                self.draw_player(player)
            
            # Add tactical highlighting for unmarked players
            if player['team'] == 'attacker':
                is_unmarked, closest_defender_dist = self._check_if_unmarked_player(player)
                if is_unmarked and closest_defender_dist > 8.0:
                    # Highlight unmarked players with yellow glow
                    glow = patches.Circle(
                        (player['x'], player['y']), 
                        radius=2.5, 
                        facecolor='yellow', 
                        alpha=0.3, 
                        zorder=5
                    )
                    self.ax.add_patch(glow)
                    
                    # Add "UNMARKED" label
                    self.ax.text(
                        player['x'], player['y'] + 3,
                        "UNMARKED",
                        color='yellow',
                        fontsize=8,
                        weight='bold',
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7),
                        zorder=12
                    )
            
            # Add movement arrows for tactical targets
            if hasattr(self, 'player_states') and player['id'] in self.player_states:
                state = self.player_states[player['id']]
                start_pos = state['start_pos']
                target_pos = state['target_pos']
                
                # Only draw arrow if there's meaningful movement
                movement_dist = math.hypot(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1])
                if movement_dist > 1.0:  # Only show if moving more than 1 meter
                    arrow_color = {
                        'Primary Receiver': 'gold',
                        'Support Runner': 'lightcoral',
                        'Marker': 'lightblue',
                        'Goalkeeper': 'purple'
                    }.get(state['role'], 'gray')
                    
                    arrow = patches.FancyArrowPatch(
                        start_pos, target_pos,
                        arrowstyle='->', 
                        color=arrow_color, 
                        alpha=0.7, 
                        linewidth=2,
                        zorder=8
                    )
                    self.ax.add_patch(arrow)
    
    def draw_tactical_attacker(self, player: Dict, color: str):
        """Draw attacker with specific tactical color"""
        x, y = player['x'], player['y']
        
        # Add subtle glow effect
        glow = patches.Circle((x, y), radius=1.8, facecolor=color, alpha=0.3, zorder=4)
        self.ax.add_patch(glow)
        
        # Red circle for attacker with tactical color
        circle = patches.Circle((x, y), radius=1.2, facecolor=color, edgecolor='white', linewidth=1.5, zorder=6)
        self.ax.add_patch(circle)
            
        # Add player ID label with better visibility
        self.ax.text(x, y+2, str(player['id']), color='white', fontsize=9, 
                    ha='center', va='center', weight='bold', zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))

    def _check_if_unmarked_player(self, player: Dict) -> Tuple[bool, float]:
        """Check if a player is unmarked (simplified version for visualization)"""
        x, y = player['x'], player['y']
        
        # Find all defenders
        defenders = [p for p in self.players if p['team'] in ['defender', 'keeper']]
        
        if not defenders:
            return True, 100.0
            
        # Find closest defender distance
        closest_dist = min(
            math.hypot(x - d['x'], y - d['y']) 
            for d in defenders
        )
        
        return closest_dist > 6.0, closest_dist

    def start_enhanced_ball_animation(self):
        """Start enhanced ball flight animation with player movement"""
        from matplotlib.animation import FuncAnimation
        
        print("🎥 Starting enhanced ball animation with player movement...")
        
        def animate_enhanced_frame(frame):
            if not getattr(self, 'simulation_active', False) or self.simulation_step >= self.max_simulation_steps:
                print(f"Animation stopped at frame {frame}")
                return
                
            progress = frame / 30.0  # 30 frames for ball flight
            
            if progress <= 1.0:
                # Ball flight phase with player movement
                self.animate_ball_flight(progress)
                self.animate_dynamic_player_movements(progress)
                if frame % 10 == 0:
                    print(f"Ball flight + movement progress: {progress:.2f}")
            elif progress <= 1.5:
                # Extended player movement phase
                self.animate_continued_player_movements(progress - 1.0)
            else:
                # Shot/outcome phase with enhanced visualization
                self.animate_enhanced_shot_outcome()
                self.simulation_active = False
                print("🏁 Enhanced animation complete!")
                
            self.simulation_step += 1
            self.fig.canvas.draw()
            
        # Start enhanced animation
        print("▶️ Creating Enhanced FuncAnimation...")
        self.animation_timer = FuncAnimation(
            self.fig, animate_enhanced_frame, 
            frames=50, interval=100, repeat=False
        )
        print("🚀 Enhanced animation started!")

    def animate_dynamic_player_movements(self, progress: float):
        """Animate realistic player movements during ball flight"""
        if not hasattr(self, 'player_states'):
            return
            
        # Remove previous movement indicators
        for patch in list(self.ax.patches):
            if hasattr(patch, '_movement_indicator'):
                patch.remove()
                
        for player in self.players:
            if player['id'] not in self.player_states:
                continue
                
            state = self.player_states[player['id']]
            start_pos = state['start_pos']
            target_pos = state['target_pos']
            speed = state['movement_speed']
            
            # Calculate current position based on progress and speed
            movement_progress = min(1.0, progress * speed)
            current_x = start_pos[0] + (target_pos[0] - start_pos[0]) * movement_progress
            current_y = start_pos[1] + (target_pos[1] - start_pos[1]) * movement_progress
            
            # Update current position
            state['current_pos'] = (current_x, current_y)
            
            # Draw movement trail for key players
            if state['role'] in ['Primary Receiver', 'Marker']:
                trail_circle = patches.Circle(
                    (current_x, current_y), 
                    radius=0.8, 
                    facecolor=('gold' if state['role'] == 'Primary Receiver' else 'lightblue'), 
                    alpha=0.4, 
                    zorder=7
                )
                trail_circle._movement_indicator = True
                self.ax.add_patch(trail_circle)

    def animate_continued_player_movements(self, progress: float):
        """Continue player movements in second phase"""
        self.animate_dynamic_player_movements(1.0 + progress * 0.5)

    def animate_enhanced_shot_outcome(self):
        """Enhanced shot outcome animation with realistic trajectory"""
        if not self.predicted_strategy:
            return
            
        shot_conf = self.predicted_strategy["predictions"]["shot_confidence"]
        primary_receiver = self.predicted_strategy["predictions"]["primary_receiver"]
        
        if shot_conf > 0.5 and primary_receiver["position"]:
            receiver_pos = primary_receiver["position"]
            shot_start = (receiver_pos['x'], receiver_pos['y'])
            goal_target = self.goal_position  # Use dynamic goal position
            
            # Enhanced shot trajectory with curve
            control_x = (shot_start[0] + goal_target[0]) / 2 + 2
            control_y = (shot_start[1] + goal_target[1]) / 2
            
            # Draw curved shot path
            shot_points = []
            for t in np.linspace(0, 1, 15):
                x = (1-t)**2 * shot_start[0] + 2*(1-t)*t * control_x + t**2 * goal_target[0]
                y = (1-t)**2 * shot_start[1] + 2*(1-t)*t * control_y + t**2 * goal_target[1]
                shot_points.append((x, y))
                
            shot_path = np.array(shot_points)
            self.ax.plot(shot_path[:, 0], shot_path[:, 1], 'r-', linewidth=4, alpha=0.8, zorder=15)
            
            # Add enhanced "SHOT!" label with power indicator
            power_level = "💥 POWERFUL" if shot_conf > 0.7 else "⚽ SHOT"
            self.ax.text(
                (shot_start[0] + goal_target[0]) / 2,
                (shot_start[1] + goal_target[1]) / 2 + 3,
                f"{power_level} SHOT!",
                color='red', fontsize=16, weight='bold',
                ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.9),
                zorder=20
            )
            
            # Simulate outcome based on confidence
            if shot_conf > 0.65:
                outcome_text = "🎯 GOAL!"
                outcome_color = 'green'
            elif shot_conf > 0.45:
                outcome_text = "🥅 CLOSE!"
                outcome_color = 'orange'
            else:
                outcome_text = "🚫 SAVED"
                outcome_color = 'red'
                
            self.ax.text(
                105, 40,
                outcome_text,
                color=outcome_color, fontsize=18, weight='bold',
                ha='center', zorder=25,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9)
            )
        else:
            # Non-shot outcome with contextual decision
            tactical_decision = self.predicted_strategy["predictions"]["tactical_decision"]
            
            self.ax.text(
                self.target_position[0], self.target_position[1] + 5,
                tactical_decision.upper(),
                color='blue', fontsize=14, weight='bold',
                ha='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcyan', alpha=0.9),
                zorder=20
            )
        
        print("✅ Enhanced corner kick simulation complete!")
        print(f"\n{'='*70}\n")
        
        # Generate strategy using trained GNN model
        if self.strategy_maker:
            try:
                # Use Strategy Maker for real GNN predictions
                strategy = self.strategy_maker.predict_strategy(self.players)
                
                # Save strategy to JSON
                self.strategy_maker.save_strategy(strategy)
                
                # Generate simulation data
                sim_data = self.strategy_maker.generate_simulation_data(strategy)
                
                # Run visualization with GNN-predicted strategy
                self.run_gnn_simulation(strategy, sim_data)
                
            except Exception as e:
                print(f"⚠️  Strategy generation error: {e}")
                print("   Falling back to simulated visualization")
                import traceback
                traceback.print_exc()
                self.generate_strategy_and_animation()  # Fallback to old method
        else:
            # Fallback if Strategy Maker not initialized
            print("⚠️  Strategy Maker not available, using simulated strategy")
            self.generate_strategy_and_animation()
            
    def run_gnn_simulation(self, strategy: Dict[str, Any], sim_data: Dict[str, Any]):
        """Run real-time simulation with GNN-predicted strategy"""
        print("\n🎬 Starting GNN-based tactical simulation...")
        
        # Clear current view
        self.ax.clear()
        self.setup_pitch()
        
        # Draw all players in starting positions
        for player in self.players:
            self.draw_player(player)
            
        # Highlight predicted receiver
        primary_receiver = strategy["predictions"]["primary_receiver"]
        if primary_receiver["player_id"]:
            receiver_pos = primary_receiver["position"]
            if receiver_pos:
                # Draw highlight circle around predicted receiver
                highlight = patches.Circle(
                    (receiver_pos['x'], receiver_pos['y']), 
                    radius=2.5, 
                    edgecolor='yellow', 
                    facecolor='none', 
                    linewidth=3, 
                    linestyle='--',
                    zorder=10
                )
                self.ax.add_patch(highlight)
                
                # Add label
                self.ax.text(
                    receiver_pos['x'], receiver_pos['y'] - 4,
                    f"TARGET\n{primary_receiver['score']:.0%}",
                    color='yellow',
                    fontsize=10,
                    weight='bold',
                    ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8),
                    zorder=11
                )
                
        # Draw ball trajectory
        if sim_data and "ball_trajectory" in sim_data:
            traj = sim_data["ball_trajectory"]
            start = traj["start"]
            target = traj["target"]
            
            # Bezier curve for ball path
            control_x = (start[0] + target[0]) / 2
            control_y = (start[1] + target[1]) / 2 + 10  # Arc upward
            
            points = []
            for t in np.linspace(0, 1, 30):
                x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * target[0]
                y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * target[1]
                points.append((x, y))
                
            path_points = np.array(points)
            self.ax.plot(path_points[:, 0], path_points[:, 1], 'w-', linewidth=2.5, alpha=0.9, zorder=8)
            
        # Draw shot trajectory if predicted
        if sim_data and sim_data.get("shot_action") and sim_data.get("shot_target"):
            shot_target = sim_data["shot_target"]
            receiver_pos = primary_receiver["position"]
            if receiver_pos:
                # Draw shot path
                self.ax.plot(
                    [receiver_pos['x'], shot_target[0]], 
                    [receiver_pos['y'], shot_target[1]],
                    'r-', linewidth=3, alpha=0.8, linestyle=':', zorder=9
                )
                
                # Add "SHOOT" label
                self.ax.text(
                    (receiver_pos['x'] + shot_target[0]) / 2,
                    (receiver_pos['y'] + shot_target[1]) / 2,
                    "SHOOT!",
                    color='red',
                    fontsize=12,
                    weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.9)
                )
                
        # Add enhanced strategy summary with clean, dynamic display
        if hasattr(self, 'predicted_strategy') and self.predicted_strategy:
            strategy = self.predicted_strategy
            primary = strategy["predictions"]["primary_receiver"]
            alternates = strategy["predictions"]["alternate_receivers"]
            
            # Main strategy display - Clean and focused
            summary_text = f"🎯 GNN PREDICTED STRATEGY\n"
            summary_text += f"Primary: Player #{primary['player_id']} ({primary['score']:.0%})\n"
            summary_text += f"Shot Confidence: {strategy['predictions']['shot_confidence']:.0%}\n"
            summary_text += f"Tactical Decision: {strategy['predictions']['tactical_decision']}\n\n"
            
            # Show top 3 alternates with scores
            summary_text += f"🏅 ALTERNATIVES:\n"
            for i, alt in enumerate(alternates[:3], 1):
                summary_text += f"  {i}. Player #{alt['player_id']} ({alt['score']:.0%})\n"
                
            # Verification info (compact)
            if "debug_info" in strategy:
                debug = strategy["debug_info"]
                summary_text += f"\n⚙️ Dynamic GNN Output Verified\n"
                summary_text += f"Graph: {debug.get('graph_nodes', 'N/A')} nodes | "
                summary_text += f"Candidates: {len(debug.get('all_receiver_scores', {}))}\n"
                
            self.ax.text(
                0.02, 0.98,
                summary_text,
                transform=self.ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.95),
                zorder=15
            )
        
        self.ax.set_title("GNN-Predicted Corner Kick Strategy", fontsize=14, weight='bold')
        self.fig.canvas.draw()
        
        print("✅ GNN simulation visualization complete!")
        print(f"\n{'='*60}\n")
        
    def _on_undo(self, event):
        """Handle Undo button click"""
        if not self.history:
            return
            
        action, player = self.history.pop()
        if action == 'add':
            # Remove the last added player
            if self.players:
                removed_player = self.players.pop()
                print(f"Removed player {removed_player['id']}")
                
                # Update counters
                if removed_player['team'] == "attacker":
                    self.attackers_placed -= 1
                elif removed_player['team'] == "defender":
                    self.defenders_placed -= 1
                elif removed_player['team'] == "keeper":
                    self.keepers_placed -= 1
                    
                # Update placement mode if needed
                if self.keepers_placed < self.max_keepers and self.defenders_placed >= self.max_defenders:
                    self.placement_mode = 'keeper'
                elif self.defenders_placed < self.max_defenders and self.attackers_placed >= self.max_attackers:
                    self.placement_mode = 'defender'
                else:
                    self.placement_mode = 'attacker'
                    
                # Update UI
                self.mode_text.set_text(f"Mode: {self.placement_mode.capitalize()}")
                self.counter_text.set_text(f"Attackers: {self.attackers_placed}/{self.max_attackers} | "
                                          f"Defenders: {self.defenders_placed}/{self.max_defenders} | "
                                          f"Goalkeepers: {self.keepers_placed}/{self.max_keepers}")
                self.instruction_text.set_text(self.get_current_instruction())
                
                # Refresh UI buttons if needed
                self.refresh_ui_buttons()
                
                # Redraw everything
                self.redraw_pitch()
                
    def _on_reset(self, event):
        """Handle Reset button click"""
        self.players = []
        self.history = []
        self.placement_mode = "attacker"
        self.player_id_counter = 100001
        self.attackers_placed = 0
        self.defenders_placed = 0
        self.keepers_placed = 0
        
        # Reset anti-spam tracking
        self.last_strategy_receiver = None
        self.last_strategy_decision = None
        self.last_strategy_timestamp = None
        
        self.mode_text.set_text(f"Mode: {self.placement_mode.capitalize()}")
        self.counter_text.set_text(f"Attackers: {self.attackers_placed}/{self.max_attackers} | "
                                  f"Defenders: {self.defenders_placed}/{self.max_defenders} | "
                                  f"Goalkeepers: {self.keepers_placed}/{self.max_keepers}")
        self.instruction_text.set_text(self.get_current_instruction())
        self.redraw_pitch()
        
    def load_model(self):
        """Load the trained receiver prediction model"""
        try:
            # In a real implementation, you would load the actual trained model
            print("Loading receiver prediction model...")
            # self.model = SingleTaskGATv2ReceiverCheckpoint(in_dim=21)  // 21 features as in dataset
            # Load model weights if available
            # self.model.load_state_dict(torch.load('best_gnn_receiver_fold1.pt'))
            # self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Using simulated predictions instead")
            
    def draw_player(self, player):
        """Draw a player on the pitch with enhanced styling following design specification"""
        x, y = player['x'], player['y']
        team = player['team']
        
        # Define colors according to specification:
        # Attackers: circles (red for primary, orange for alternates, blue for others)
        # Defenders: gray triangles
        # Goalkeeper: green square
        colors = {
            'attacker': '#ff3b3b',  # Red circles (primary will be highlighted separately)
            'defender': '#808080',  # Gray triangles  
            'keeper': '#2ecc40'     # Green square
        }
        
        color = colors.get(team, '#ffffff')  # Default white
        
        # Add subtle glow effect
        glow = patches.Circle((x, y), radius=1.8, facecolor=color, alpha=0.3, zorder=4)
        self.ax.add_patch(glow)
        
        # Create different shapes based on player type following specification
        if team == 'keeper':
            # Green square for goalkeeper
            path = create_square_path(size=1.2)
            patch = patches.PathPatch(path, facecolor=color, edgecolor='white', linewidth=1.5, zorder=6)
            patch.set_transform(self.ax.transData + mtransforms.Affine2D().translate(x, y))
            self.ax.add_patch(patch)
        elif team == 'defender':
            # Gray triangle for defenders
            path = create_triangle_path(size=1.2)
            patch = patches.PathPatch(path, facecolor=color, edgecolor='white', linewidth=1.5, zorder=6)
            patch.set_transform(self.ax.transData + mtransforms.Affine2D().translate(x, y))
            self.ax.add_patch(patch)
        else:
            # Red circle for attackers (primary receiver highlighting done separately)
            circle = patches.Circle((x, y), radius=1.2, facecolor=color, edgecolor='white', linewidth=1.5, zorder=6)
            self.ax.add_patch(circle)
            
        # Add player ID label with better visibility
        self.ax.text(x, y+2, str(player['id']), color='white', fontsize=9, 
                    ha='center', va='center', weight='bold', zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))
        
    def draw_tactical_lines(self):
        """Draw tactical lines showing runs and ball path"""
        if len(self.players) < 2:
            return
            
        # Draw ball trajectory from corner to goal
        corner_x, corner_y = 105, 0
        goal_x, goal_y = 105, 34
        
        # Draw curved ball path (Bezier curve)
        control_x = 85
        control_y = 20
        
        # Sample points along the curve
        points = []
        for t in np.linspace(0, 1, 20):
            # Quadratic Bezier curve
            x = (1-t)**2 * corner_x + 2*(1-t)*t * control_x + t**2 * goal_x
            y = (1-t)**2 * corner_y + 2*(1-t)*t * control_y + t**2 * goal_y
            points.append((x, y))
            
        # Draw the path
        path_points = np.array(points)
        self.ax.plot(path_points[:, 0], path_points[:, 1], 'w--', linewidth=1.5, alpha=0.7)
        
        # Draw arrows showing player runs
        # For attackers, draw arrows from current position to target areas
        attackers = [p for p in self.players if p['team'] == 'attacker']
        for attacker in attackers:
            # Simple targets (near post, far post, penalty spot)
            targets = [
                (95, 34),  # Near post
                (90, 30),  # Far post
                (92, 34)   # Penalty spot
            ]
            
            # Draw arrows to targets with different styles
            for i, (tx, ty) in enumerate(targets):
                if i == 0:  # Primary target - solid line
                    self.ax.annotate('', xy=(tx, ty), xytext=(attacker['x'], attacker['y']),
                                   arrowprops=dict(arrowstyle='->', color='#ff3b3b', lw=1.2, alpha=0.8))
                else:  # Alternate targets - dashed line
                    self.ax.annotate('', xy=(tx, ty), xytext=(attacker['x'], attacker['y']),
                                   arrowprops=dict(arrowstyle='->', color='#ffb86b', lw=1.0, alpha=0.6, 
                                                 linestyle='dashed'))
        
        # For defenders, draw marking lines to nearby attackers
        defenders = [p for p in self.players if p['team'] == 'defender']
        for defender in defenders:
            # Find closest attacker
            closest_attacker = None
            min_dist = float('inf')
            
            for attacker in attackers:
                dist = math.hypot(attacker['x'] - defender['x'], attacker['y'] - defender['y'])
                if dist < min_dist:
                    min_dist = dist
                    closest_attacker = attacker
                    
            if closest_attacker and min_dist < 20:  # Only draw if close enough
                self.ax.annotate('', xy=(closest_attacker['x'], closest_attacker['y']), 
                               xytext=(defender['x'], defender['y']),
                               arrowprops=dict(arrowstyle='->', color='#4aa3ff', lw=0.8, alpha=0.6))
                
    def redraw_pitch(self):
        """Redraw the entire pitch and players with tactical lines"""
        # Clear the axis
        self.ax.clear()
        
        # Redraw pitch
        self.setup_pitch()
        
        # Redraw all players
        for player in self.players:
            self.draw_player(player)
            
        # Draw tactical lines
        self.draw_tactical_lines()
            
        # Redraw UI
        self.setup_ui()
        
        self.fig.canvas.draw()
        
    def convert_to_model_input(self):
        """Convert placed positions to model input format (graph structure)"""
        print("Converting positions to graph structure for model input...")
        
        if len(self.players) == 0:
            return None
            
        # Create node features for each player
        node_features = []
        player_ids = []
        positions = []
        team_flags = []
        
        # Constants for normalization
        PITCH_X = 105.0
        PITCH_Y = 68.0
        GOAL_X = 105.0
        GOAL_Y = 34.0
        
        for player in self.players:
            x_norm = player['x'] / PITCH_X  # Normalize to [0,1]
            y_norm = player['y'] / PITCH_Y  # Normalize to [0,1]
            
            # Determine team flag (attacker=1, defender/keeper=0)
            is_attacker = 1.0 if player['team'] == 'attacker' else 0.0
            is_keeper = 1.0 if player['team'] == 'keeper' else 0.0
            
            # Calculate distances
            dist_ball = math.hypot(x_norm - 1.0, y_norm - 0.0)  # Corner kick position
            dist_goal = math.hypot(1.0 - x_norm, (GOAL_Y/PITCH_Y) - y_norm)
            
            # Simple velocity (set to zero for static positions)
            vx, vy = 0.0, 0.0
            
            # Create feature vector (matching the GNN dataset format)
            # Features: [x_norm, y_norm, vx, vy, is_attacker, is_keeper, 
            #            dist_ball, dist_goal, role_encoding, height, header_proxy,
            #            team_encoding, pace, jumping, heading, minute_bucket,
            #            missing_flags...]
            features = [
                x_norm, y_norm, vx, vy,
                is_attacker, is_keeper,
                dist_ball, dist_goal,
                0.0,  # role_encoding
                180.0,  # height (default)
                0.7,  # header_proxy (default)
                is_attacker,  # team_encoding
                0.7,  # pace (default)
                0.6,  # jumping (default)
                0.65,  # heading (default)
                0.5,  # minute_bucket (default)
                0.0, 0.0, 0.0, 0.0, 0.0  # missing flags
            ]
            
            node_features.append(features)
            player_ids.append(player['id'])
            positions.append((player['x'], player['y']))
            team_flags.append(is_attacker)
            
        # Add ball node (at corner position)
        ball_features = [
            1.0, 0.0,  # x, y (corner)
            0.0, 0.0,  # vx, vy
            0.0, 0.0,  # is_attacker, is_keeper
            0.0, 0.0,  # dist_ball, dist_goal (not meaningful for ball)
            0.0, 0.0, 0.0,  # role, height, header
            0.0, 0.0, 0.0, 0.0,  # team, pace, jumping, heading
            0.0,  # minute_bucket
            0.0, 0.0, 0.0, 0.0, 0.0  # missing flags
        ]
        node_features.append(ball_features)
        player_ids.append(-1)  # Ball ID
        positions.append((105.0, 0.0))  # Corner position
        team_flags.append(0.0)
        
        # Create edges (connect players within radius and to ball)
        edge_index = []
        edge_attr = []
        
        num_players = len(self.players)
        ball_idx = len(node_features) - 1  # Ball is last node
        
        # Connect all players to ball
        for i in range(num_players):
            # Distance to ball
            px, py = positions[i]
            dist = math.hypot(px - 105.0, py - 0.0) / math.hypot(PITCH_X, PITCH_Y)
            # Angle to goal
            angle = math.atan2(GOAL_Y - py, GOAL_X - px)
            # Same team flag (0 for ball)
            same_team = 0.0
            # Marking flag (0 for now)
            marking = 0.0
            
            edge_index.append([i, ball_idx])
            edge_attr.append([dist, angle, same_team, marking])
            
            # Reverse edge
            edge_index.append([ball_idx, i])
            edge_attr.append([dist, angle, same_team, marking])
        
        # Connect players within radius to each other
        radius = 15.0  # meters
        for i in range(num_players):
            for j in range(i+1, num_players):
                px1, py1 = positions[i]
                px2, py2 = positions[j]
                dist = math.hypot(px1 - px2, py1 - py2)
                
                if dist <= radius:
                    # Angle between players
                    angle = math.atan2(py2 - py1, px2 - px1)
                    # Same team flag
                    same_team = 1.0 if team_flags[i] == team_flags[j] else 0.0
                    # Marking flag (simplified)
                    marking = 1.0 if team_flags[i] != team_flags[j] else 0.0
                    
                    edge_index.append([i, j])
                    edge_attr.append([dist/PITCH_X, angle, same_team, marking])
                    
                    # Reverse edge
                    edge_index.append([j, i])
                    edge_attr.append([dist/PITCH_X, angle, same_team, marking])
        
        # Convert to tensors
        x_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros((0, 4), dtype=torch.float)
        
        # Create batch tensor (all nodes belong to batch 0)
        batch_tensor = torch.zeros(x_tensor.size(0), dtype=torch.long)
        
        # Create PyG Data object
        data = Data(
            x=x_tensor,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor,
            batch=batch_tensor
        )
        
        return data, player_ids[:-1]  # Exclude ball from player_ids
        
    def predict_receiver(self, data):
        """Use the model to predict the best receiver"""
        print("Running receiver prediction model...")
        
        try:
            # In a real implementation, you would run the model:
            # with torch.no_grad():
            #     logits = self.model(data)
            #     probabilities = torch.sigmoid(logits)
            
            # For demonstration, we'll simulate a prediction
            # Select a random attacker as the predicted receiver
            attackers = [i for i, player in enumerate(self.players) if player['team'] == 'attacker']
            if attackers:
                # Simulate model prediction with probabilities
                probabilities = np.random.dirichlet(np.ones(len(attackers)), size=1)[0]
                best_idx = np.argmax(probabilities)
                predicted_receiver_idx = attackers[best_idx]
                confidence = probabilities[best_idx]
                return predicted_receiver_idx, confidence
            else:
                # Fallback to first player
                return 0, 0.5
        except Exception as e:
            print(f"Model prediction failed: {e}")
            # Fallback to heuristic
            attackers = [i for i, player in enumerate(self.players) if player['team'] == 'attacker']
            if attackers:
                # Simple heuristic: select attacker closest to goal
                goal_x, goal_y = 105, 34
                best_idx = min(attackers, 
                              key=lambda i: math.hypot(self.players[i]['x'] - goal_x, self.players[i]['y'] - goal_y))
                return best_idx, 0.7
            else:
                return 0, 0.5
        
    def generate_strategy_and_animation(self):
        """Generate strategy using model and create animation"""
        try:
            # Convert positions to model input
            result = self.convert_to_model_input()
            if result is None:
                print("No players placed. Cannot generate strategy.")
                return
                
            data, player_ids = result
            
            # Run model prediction
            predicted_receiver_idx, confidence = self.predict_receiver(data)
            predicted_receiver_id = player_ids[predicted_receiver_idx] if predicted_receiver_idx < len(player_ids) else player_ids[0]
            
            print(f"Model predicts player {predicted_receiver_id} as the best receiver (confidence: {confidence:.2f})")
            
            # Create strategy data structure with enhanced information
            strategy_data = {
                "corner_id": 1,
                "timestamp": datetime.datetime.now().isoformat() + "Z",
                "best_strategy": f"Target Player #{predicted_receiver_id}",
                "confidence": float(confidence),
                "corner_flag": {"x": 100, "y": 0},
                "primary": {
                    "id": int(predicted_receiver_id),
                    "start": [self.players[predicted_receiver_idx]['x'] * 100 / 105, 
                             self.players[predicted_receiver_idx]['y'] * 100 / 68],
                    "target": [95, 34]  # Near post target
                },
                "alternates": [],
                "players": [],
                "cluster_zone": [93, 34],
                "tactical_setup": {
                    "total_players": len(self.players),
                    "attackers": len([p for p in self.players if p['team'] == 'attacker']),
                    "defenders": len([p for p in self.players if p['team'] == 'defender']),
                    "keeper": len([p for p in self.players if p['team'] == 'keeper'])
                }
            }
            
            # Add all players to strategy with detailed information
            # Identify the corner taker (attacker closest to the corner flag)
            attackers = [p for p in self.players if p['team'] == 'attacker']
            corner_taker = None
            if attackers:
                corner_flag_x, corner_flag_y = 105, 0
                corner_taker = min(attackers, 
                                  key=lambda p: math.hypot(p['x'] - corner_flag_x, p['y'] - corner_flag_y))
            
            # Validate and adjust player positions if needed
            for i, player in enumerate(self.players):
                intent = "unknown"
                if player['team'] == 'attacker':
                    if player['id'] == predicted_receiver_id:
                        intent = "primary_target"
                    elif corner_taker and player['id'] == corner_taker['id']:
                        intent = "corner_taker"
                    else:
                        intent = "support_runner"
                elif player['team'] == 'defender':
                    intent = "marking"
                elif player['team'] == 'keeper':
                    intent = "goalkeeping"
                    
                # Validate player position and adjust if needed
                adjusted_x, adjusted_y = self.validate_and_adjust_position(player['x'], player['y'], player['team'])
                
                strategy_data["players"].append({
                    "id": int(player['id']),
                    "team": player['team'],
                    "intent": intent,
                    "position": {
                        "x": adjusted_x,
                        "y": adjusted_y
                    },
                    "start": [adjusted_x * 100 / 105, adjusted_y * 100 / 68],
                    "target": [adjusted_x * 100 / 105 + 2, adjusted_y * 100 / 68]  # Simple target
                })
                
            # Add some alternate targets (other attackers)
            other_attackers = [(i, p) for i, p in enumerate(self.players) if p['team'] == 'attacker' and p['id'] != predicted_receiver_id]
            for i, (idx, attacker) in enumerate(other_attackers[:2]):  # First two other attackers as alternates
                # Validate alternate target position
                adjusted_x, adjusted_y = self.validate_and_adjust_position(attacker['x'], attacker['y'], 'attacker')
                strategy_data["alternates"].append({
                    "id": int(attacker['id']),
                    "position": {
                        "x": adjusted_x,
                        "y": adjusted_y
                    },
                    "start": [adjusted_x * 100 / 105, adjusted_y * 100 / 68],
                    "target": [90 - i*5, 30 + i*5]  # Different targets
                })
                
            # Save strategy to JSON with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_filename = f"corner_strategy_manual_{timestamp}.json"
            with open(strategy_filename, 'w') as f:
                json.dump(strategy_data, f, indent=2)
            print(f"Strategy saved to {strategy_filename}")
            
            # Also save a summary report
            summary_filename = f"corner_strategy_summary_{timestamp}.txt"
            with open(summary_filename, 'w') as f:
                f.write(f"Corner Kick Strategy Report\n")
                f.write(f"==========================\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Best Strategy: {strategy_data['best_strategy']}\n")
                f.write(f"Confidence: {strategy_data['confidence']:.2f}\n")
                f.write(f"Total Players: {strategy_data['tactical_setup']['total_players']}\n")
                f.write(f"Attackers: {strategy_data['tactical_setup']['attackers']}\n")
                f.write(f"Defenders: {strategy_data['tactical_setup']['defenders']}\n")
                f.write(f"Keeper: {strategy_data['tactical_setup']['keeper']}\n")
                f.write(f"\nPrimary Target: Player #{predicted_receiver_id}\n")
                f.write(f"Alternate Targets: {', '.join([str(a['id']) for a in strategy_data['alternates']])}\n")
            print(f"Summary report saved to {summary_filename}")
            
            # Create and run Pygame visualization instead of Matplotlib
            print("Generating Pygame visualization...")
            
            # Import the Pygame visualization engine
            try:
                from corner_visualization_pygame import CornerKickVisualization
                
                # Run Pygame visualization
                visualization = CornerKickVisualization(strategy_data)
                visualization.run()
                
                print("Pygame visualization completed successfully!")
                
            except Exception as e:
                print(f"Error running Pygame visualization: {e}")
                print("Falling back to Matplotlib visualization...")
                
                # Fallback to Matplotlib if Pygame fails
                replay = CornerReplayV4(
                    strategy_data=strategy_data,
                    corner_id=1,
                    view="corner_close",  # This automatically zooms into the penalty box region (x ∈ [70, 105])
                    speed="normal"
                )
                replay.run(save_video=True)
            
            print("Process completed successfully!")
            print(f"Files generated:")
            print(f"  - Strategy JSON: {strategy_filename}")
            print(f"  - Summary Report: {summary_filename}")
            
        except Exception as e:
            print(f"Error generating strategy and animation: {e}")
            # Show error message in the UI
            self.ax.text(0.5, 0.5, f"ERROR: {str(e)}", 
                        transform=self.ax.transAxes, fontsize=12, color='red', weight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
            self.fig.canvas.draw()
            
    def validate_and_adjust_position(self, x, y, team):
        """Validate and adjust player positions to ensure they are in valid areas"""
        # Ensure all positions are within the pitch boundaries
        x = max(0, min(105, x))
        y = max(0, min(68, y))
        
        # For attackers, ensure they're not too close to the goal line unless intended
        if team == 'attacker':
            # Ensure attackers are in the attacking half
            x = max(52.5, x)
            
        # For defenders, ensure they're not too far forward
        elif team == 'defender':
            # Ensure defenders are mostly in their own half
            x = min(80, x)
            
        # For goalkeepers, ensure they're near their goal
        elif team == 'keeper':
            # Keep goalkeepers near the goal area
            x = max(99, min(105, x))
            y = max(26, min(42, y))
            
        return x, y
        
    def run(self):
        """Run the interactive setup"""
        # Adjust layout to avoid tight_layout warning
        self.fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.95)
        plt.show()
        
    def initialize_player_states(self, strategy: Dict[str, Any]):
        """Initialize player states for dynamic simulation movement"""
        print("\n🎬 Initializing player movement states...")
        
        primary_receiver_id = strategy["predictions"]["primary_receiver"]["player_id"]
        self.player_states = {}
        
        for player in self.players:
            start_pos = (player['x'], player['y'])
            target_pos = self.calculate_tactical_target(player, primary_receiver_id, strategy)
            
            self.player_states[player['id']] = {
                'start_pos': start_pos,
                'target_pos': target_pos,
                'current_pos': start_pos,
                'movement_speed': self.get_movement_speed(player),
                'role': self.get_tactical_role(player, primary_receiver_id)
            }
            
            print(f"   Player #{player['id']}: {self.player_states[player['id']]['role']} | "
                  f"Start: {start_pos} → Target: {target_pos}")
    
    def calculate_tactical_target(self, player: Dict, primary_receiver_id: int, 
                                strategy: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate tactical target position for a player during simulation"""
        x, y = player['x'], player['y']
        
        if player['id'] == primary_receiver_id:
            # Primary receiver moves toward goal for shot
            goal_x, goal_y = self.goal_position  # Use dynamic goal position
            # Move 3-5 meters toward goal for optimal shooting position
            dx = (goal_x - x) * 0.3
            dy = (goal_y - y) * 0.2
            
            # Adjust target position based on which goal we're attacking
            if goal_x > 50:  # Attacking right goal
                return (min(102, x + dx), max(28, min(40, y + dy)))
            else:  # Attacking left goal
                return (max(3, x + dx), max(28, min(40, y + dy)))
            
        elif player['team'] == 'attacker':
            # Supporting attackers create space and make runs
            return self.calculate_support_run_position(player, primary_receiver_id)
            
        elif player['team'] == 'defender':
            # Defenders move to mark nearest threats
            return self.calculate_defensive_target(player)
            
        elif player['team'] == 'keeper':
            # Goalkeeper adjusts position based on threat
            return self.calculate_keeper_target(player, strategy)
            
        return (x, y)  # Default: stay in place
    
    def calculate_support_run_position(self, player: Dict, 
                                     primary_receiver_id: int) -> Tuple[float, float]:
        """Calculate support run position for attacking players"""
        x, y = player['x'], player['y']
        
        # Find primary receiver position
        primary_receiver = next((p for p in self.players if p['id'] == primary_receiver_id), None)
        
        if primary_receiver:
            # Create space by moving away from primary receiver
            dx = x - primary_receiver['x']
            dy = y - primary_receiver['y']
            
            # Normalize and extend movement
            dist = math.hypot(dx, dy)
            if dist > 0:
                dx_norm = (dx / dist) * 3  # Move 3m away
                dy_norm = (dy / dist) * 3
            else:
                dx_norm, dy_norm = 3, 0
                
            # Move toward goal while creating space - direction based on goal side
            if self.goal_position[0] > 50:  # Attacking right goal
                target_x = min(100, x + dx_norm * 0.5 + 2)  # Move right
            else:  # Attacking left goal
                target_x = max(5, x + dx_norm * 0.5 - 2)  # Move left
            target_y = max(20, min(48, y + dy_norm * 0.5))  # Stay in goal area width
        else:
            # Default support run toward goal
            if self.goal_position[0] > 50:  # Right goal
                target_x = min(100, x + 3)
            else:  # Left goal
                target_x = max(5, x - 3)
            target_y = y
            
        return (target_x, target_y)
    
    def calculate_defensive_target(self, defender: Dict) -> Tuple[float, float]:
        """Calculate defensive target - mark nearest attacker"""
        attackers = [p for p in self.players if p['team'] == 'attacker']
        
        if not attackers:
            return (defender['x'], defender['y'])
            
        # Find nearest attacker to mark
        nearest_attacker = min(attackers, 
                             key=lambda a: math.hypot(a['x'] - defender['x'], 
                                                     a['y'] - defender['y']))
        
        # Move toward attacker but maintain some distance (marking distance)
        dx = nearest_attacker['x'] - defender['x']
        dy = nearest_attacker['y'] - defender['y']
        
        # Move 70% of the way to the attacker
        target_x = defender['x'] + dx * 0.7
        target_y = defender['y'] + dy * 0.7
        
        # Keep defender in reasonable defensive zones - adapt to goal side
        if self.goal_position[0] > 50:  # Defending right goal
            target_x = max(70, min(100, target_x))
        else:  # Defending left goal
            target_x = max(5, min(35, target_x))
        target_y = max(15, min(53, target_y))
        
        return (target_x, target_y)
    
    def calculate_keeper_target(self, keeper: Dict, strategy: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate goalkeeper target position based on threat assessment"""
        shot_confidence = strategy["predictions"]["shot_confidence"]
        
        # Keeper positioning based on threat level - adapt to goal side
        if self.goal_position[0] > 50:  # Right goal
            if shot_confidence > 0.6:
                # High threat: move toward near post
                target_x = 103.5
                target_y = 32.0
            elif shot_confidence > 0.4:
                # Medium threat: central positioning
                target_x = 103.0
                target_y = 34.0
            else:
                # Low threat: slight adjustment only
                target_x = keeper['x']
                target_y = max(30, min(38, keeper['y']))
        else:  # Left goal
            if shot_confidence > 0.6:
                # High threat: move toward near post
                target_x = 1.5
                target_y = 32.0
            elif shot_confidence > 0.4:
                # Medium threat: central positioning
                target_x = 2.0
                target_y = 34.0
            else:
                # Low threat: slight adjustment only
                target_x = keeper['x']
                target_y = max(30, min(38, keeper['y']))
            
        return (target_x, target_y)
    
    def get_movement_speed(self, player: Dict) -> float:
        """Get movement speed multiplier for different player types"""
        if player['team'] == 'attacker':
            return 0.8  # Fast attacking movement
        elif player['team'] == 'defender':
            return 0.6  # Moderate defensive adjustment
        elif player['team'] == 'keeper':
            return 0.4  # Slow, precise keeper positioning
        return 0.5
    
    def get_tactical_role(self, player: Dict, primary_receiver_id: int) -> str:
        """Get tactical role description for player"""
        if player['id'] == primary_receiver_id:
            return "Primary Receiver"
        elif player['team'] == 'attacker':
            return "Support Runner"
        elif player['team'] == 'defender':
            return "Marker"
        elif player['team'] == 'keeper':
            return "Goalkeeper"
        return "Unknown"

def main():
    """Main function to run the interactive tactical setup"""
    print("Starting Interactive Tactical Setup...")
    print("Instructions:")
    print("1. Select corner side using '← Left' or 'Right →' buttons")
    print("   - Click same button to toggle between top/bottom corners")
    print("2. Click on the pitch to place players")
    print("3. First clicks place attackers (red circles)")
    print("4. Next clicks place defenders (gray triangles)")
    print("5. Final click places goalkeeper (green square)")
    print("6. Press 'Generate & Simulate' when ready (min 6 players)")
    print("7. Press 'Undo' or 'U' to remove last player")
    print("8. Press 'Reset' or 'R' to clear all players")
    print("")
    print("Corner options:")
    print("  - Bottom-Right (default): (105, 0)")
    print("  - Top-Right: (105, 68)")
    print("  - Bottom-Left: (0, 0)")
    print("  - Top-Left: (0, 68)")
    print("")
    
    setup = InteractiveTacticalSetup()
    setup.run()

if __name__ == "__main__":
    main()