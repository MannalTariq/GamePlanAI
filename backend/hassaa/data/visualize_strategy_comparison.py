#!/usr/bin/env python3
"""
Visual Strategy Comparison Tool
Creates side-by-side comparison of different formation strategies.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os

def visualize_strategy_comparison():
    """Create visual comparison of formation strategies"""
    
    # Check if comparison file exists
    if not os.path.exists("formation_sensitivity_test.json"):
        print("❌ formation_sensitivity_test.json not found. Run test_formation_sensitivity.py first.")
        return
        
    # Load comparison data
    with open("formation_sensitivity_test.json", "r") as f:
        data = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("GNN Strategy Comparison: Formation Sensitivity Analysis", fontsize=16, weight='bold')
    
    scenarios = ["compact", "spread", "near_post"]
    scenario_names = ["Compact Formation", "Spread Formation", "Near Post Formation"]
    
    for i, (scenario, name) in enumerate(zip(scenarios, scenario_names)):
        ax = axes[i]
        
        # Get scenario data
        scenario_data = data["scenarios"][scenario]
        formation = scenario_data["formation"]
        strategy = scenario_data["strategy"]
        
        # Draw pitch
        ax.set_xlim(75, 105)
        ax.set_ylim(20, 50)
        ax.set_aspect('equal')
        ax.add_patch(patches.Rectangle((75, 20), 30, 30, linewidth=1, edgecolor='white', facecolor='#234d20'))
        
        # Goal
        ax.add_patch(patches.Rectangle((105, 30.34), 1, 7.32, edgecolor='white', facecolor='none', linewidth=2))
        
        # Penalty area
        ax.add_patch(patches.Rectangle((88.5, 24), 16.5, 20, edgecolor='white', facecolor='none', linewidth=1))
        
        # Goal area
        ax.add_patch(patches.Rectangle((99.5, 30), 5.5, 8, edgecolor='white', facecolor='none', linewidth=1))
        
        # Corner arc
        ax.add_patch(patches.Arc((105, 20), 2, 2, theta1=90, theta2=180, edgecolor='white', linewidth=1))
        
        # Draw players
        primary_receiver_id = strategy["predictions"]["primary_receiver"]["player_id"]
        
        for player in formation:
            x, y = player["x"], player["y"]
            team = player["team"]
            player_id = player["id"]
            
            # Choose color and shape
            if team == "attacker":
                color = '#ff3b3b'
                marker = 'o'
                size = 200
            elif team == "defender":
                color = '#4aa3ff'
                marker = '^'
                size = 150
            else:  # keeper
                color = '#9b59b6'
                marker = 's'
                size = 150
            
            # Highlight primary receiver
            if player_id == primary_receiver_id:
                ax.scatter(x, y, c='gold', marker='*', s=400, edgecolors='black', linewidth=2, zorder=10)
                ax.text(x, y-2, f"TARGET\n{strategy['predictions']['primary_receiver']['score']:.3f}", 
                       ha='center', va='top', fontsize=8, weight='bold', color='gold')
            else:
                ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='white', linewidth=1, zorder=5)
            
            # Add player ID
            ax.text(x, y+1.5, str(player_id), ha='center', va='bottom', fontsize=8, color='white', weight='bold')
        
        # Add strategy info
        shot_conf = strategy["predictions"]["shot_confidence"]
        decision = strategy["predictions"]["tactical_decision"]
        
        info_text = f"Shot Confidence: {shot_conf:.4f}\nDecision: {decision}"
        ax.text(77, 48, info_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(name, fontsize=12, weight='bold')
        ax.axis('off')
    
    # Add summary
    analysis = data["analysis"]
    summary_text = f"""Analysis Summary:
    
Receiver Score Range: {analysis['receiver_score_range'][0]:.4f} - {analysis['receiver_score_range'][1]:.4f}
Shot Confidence Range: {analysis['shot_confidence_range'][0]:.4f} - {analysis['shot_confidence_range'][1]:.4f}
Unique Decisions: {len(analysis['unique_decisions'])}
Models Responsive: {'✅ Yes' if analysis['models_responsive'] else '⚠️ Limited'}

Legend:
🔴 Attackers  🔷 Defenders  🟣 Goalkeepers  ⭐ Primary Target"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig("strategy_comparison_visual.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Visual comparison saved as: strategy_comparison_visual.png")

if __name__ == "__main__":
    visualize_strategy_comparison()