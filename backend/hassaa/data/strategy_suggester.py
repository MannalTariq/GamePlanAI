#!/usr/bin/env python3
"""
Strategy Suggester for Optimal Corner Strategies
Combines trained shot and receiver GNN models to recommend optimal corner strategies.
"""

import os
import sys
import json
import math
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from torch_geometric.nn.aggr import AttentionalAggregation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from gnn_dataset import SetPieceGraphDataset, _safe_float

# Constants
PITCH_X = 100.0
PITCH_Y = 100.0
GOAL_X = 100.0
GOAL_Y = 50.0

def draw_pitch(ax):
    """Draw a football pitch on the given axes"""
    # Pitch outline (scaled to 105m x 68m)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.add_patch(patches.Rectangle((0,0), 105, 68, linewidth=2, edgecolor='black', facecolor='green', alpha=0.1))
    # Goal boxes
    ax.add_patch(patches.Rectangle((0, 24), 6, 20, linewidth=1.5, edgecolor='white', facecolor='none'))
    ax.add_patch(patches.Rectangle((99, 24), 6, 20, linewidth=1.5, edgecolor='white', facecolor='none'))
    # Halfway line
    ax.plot([52.5, 52.5], [0, 68], color='white', linestyle='--')
    ax.set_aspect('equal')
    ax.axis('off')

def visualize_strategy(strategy_data, corner_id):
    """Visualize the corner strategy on a football pitch"""
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax)

    # Plot all player positions
    for player_id, (x, y) in strategy_data['player_positions'].items():
        # Convert from normalized coordinates (0-100) to pitch coordinates (0-105, 0-68)
        pitch_x = x * 1.05  # 100 -> 105
        pitch_y = y * 0.68  # 100 -> 68
        ax.scatter(pitch_x, pitch_y, color='blue', s=60, alpha=0.6)
        ax.text(pitch_x+0.5, pitch_y+0.5, str(player_id), fontsize=8, color='white')

    # Highlight primary target
    px, py = strategy_data['primary_position']
    pitch_px = px * 1.05
    pitch_py = py * 0.68
    ax.scatter(pitch_px, pitch_py, color='red', s=100, edgecolor='black', label='Primary Target')

    # Highlight alternate targets
    for alt in strategy_data['alternate_positions']:
        alt_x, alt_y = alt
        pitch_alt_x = alt_x * 1.05
        pitch_alt_y = alt_y * 0.68
        ax.scatter(pitch_alt_x, pitch_alt_y, color='orange', s=80, alpha=0.8, label='Alternate Target')

    # Draw cluster zone
    cx, cy = strategy_data['cluster_zone']
    pitch_cx = cx * 1.05
    pitch_cy = cy * 0.68
    ax.add_patch(patches.Circle((pitch_cx, pitch_cy), radius=5, color='yellow', alpha=0.2, label='Cluster Zone'))

    # Draw ball trajectory (from corner to primary target)
    # Corner position (assuming right corner for this visualization)
    ball_x = 105  # Right corner
    ball_y = 0   # Bottom corner
    ax.plot([ball_x, pitch_px], [ball_y, pitch_py], color='white', linewidth=2.5, linestyle='--', label='Predicted Ball Path')

    # Annotate strategy
    plt.title(f"Corner Strategy: {strategy_data['best_strategy']} (Conf: {strategy_data['confidence']:.2f})", fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"corner_strategy_{corner_id}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as {output_path}")
    
    # Show the plot
    plt.show()

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads, edge_dim=edge_dim, add_self_loops=True, dropout=dropout)
        self.ln = LayerNorm(out_dim * heads, affine=True)
        self.proj = None
        if in_dim != out_dim * heads:
            self.proj = nn.Linear(in_dim, out_dim * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h = self.gat(x, edge_index, edge_attr)
        if self.proj is not None:
            x = self.proj(x)
        x = x + h
        x = self.ln(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class GATv2EncoderCheckpoint(nn.Module):
    """GATv2 Encoder that matches the checkpoint architecture (edge_dim=4)"""
    def __init__(self, in_dim: int, hidden: int = 128, heads1: int = 4, heads2: int = 4, dropout: float = 0.3):
        super().__init__()
        # Use edge_dim=4 to match checkpoint
        self.block1 = ResidualBlock(in_dim, hidden, heads1, edge_dim=4, dropout=dropout)
        self.block2 = ResidualBlock(hidden * heads1, hidden, heads2, edge_dim=4, dropout=dropout)
        # final heads=1, keep dim hidden
        self.conv3 = GATv2Conv(hidden * heads2, 64, heads=1, edge_dim=4, add_self_loops=True, dropout=dropout)
        self.ln3 = LayerNorm(64, affine=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.block1(x, edge_index, edge_attr)
        x = self.block2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

class ShotHead(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        gate_nn = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.pool = AttentionalAggregation(gate_nn)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x, batch):
        hg = self.pool(x, batch)
        logit = self.mlp(hg)
        return logit.view(-1, 1)

class ReceiverHead(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class SingleTaskGATv2ShotCheckpoint(nn.Module):
    """Shot model that matches the checkpoint architecture"""
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = GATv2EncoderCheckpoint(in_dim, hidden=hidden, heads1=heads, heads2=heads, dropout=dropout)
        self.shot_head = ShotHead(64)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.encoder(x, edge_index, edge_attr)
        shot_logit = self.shot_head(h, batch)
        return shot_logit

class SingleTaskGATv2ReceiverCheckpoint(nn.Module):
    """Receiver model that matches the checkpoint architecture"""
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = GATv2EncoderCheckpoint(in_dim, hidden=hidden, heads1=heads, heads2=heads, dropout=dropout)
        self.receiver_head = ReceiverHead(64)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.encoder(x, edge_index, edge_attr)
        receiver_logit = self.receiver_head(h)
        return receiver_logit

class ResidualGATv2Receiver(torch.nn.Module):
    """Residual GATv2 model for receiver prediction (matches the architecture in best_fixed_receiver_fold1.pt)"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.25):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)
        self.res_fc = torch.nn.Linear(in_channels, hidden_channels * heads)
        self.dropout = torch.nn.Dropout(dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 1️⃣ Normalize Node Features
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        
        res = self.res_fc(x)
        x = self.gat1(x, edge_index)
        x = torch.nn.functional.elu(self.bn1(x + res))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = torch.nn.functional.elu(self.bn2(x))
        x = self.dropout(x)
        return self.lin(x)

class CornerStrategySuggester:
    def __init__(self, root: str, device: str = "cpu"):
        """
        Initialize the strategy suggester with trained models.
        
        Args:
            root: Path to data directory containing trained model checkpoints
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.root = root
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load trained models
        self.receiver_model = self._load_receiver_model()
        self.shot_model = self._load_shot_model()
        
        # Set models to evaluation mode
        self.receiver_model.eval()
        self.shot_model.eval()
        
        print("✅ Strategy Suggester initialized successfully")

    def _load_receiver_model(self):
        """Load the trained receiver prediction model."""
        # Look for the best receiver model checkpoint
        checkpoint_paths = [
            os.path.join(self.root, "best_fixed_receiver_fold1.pt"),
            os.path.join(self.root, "best_gnn_receiver_fold1.pt"),
            os.path.join(self.root, "best_improved_receiver_fold1.pt")
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError("No trained receiver model checkpoint found")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Check the architecture based on state dict keys
        state_dict_keys = list(checkpoint["model_state"].keys())
        
        if any("gat1" in key for key in state_dict_keys) and any("gat2" in key for key in state_dict_keys):
            # This is the ResidualGATv2Receiver model
            print("Loading ResidualGATv2Receiver model architecture")
            model = ResidualGATv2Receiver(
                in_channels=checkpoint["in_dim"],
                hidden_channels=checkpoint["hidden"],
                out_channels=1,
                heads=checkpoint["heads"],
                dropout=checkpoint["dropout"]
            ).to(self.device)
            
            # Load state dict
            model.load_state_dict(checkpoint["model_state"])
            print(f"✅ Receiver model loaded from {checkpoint_path}")
        else:
            # This is the SingleTaskGATv2Receiver model with checkpoint architecture
            print("Loading SingleTaskGATv2Receiver model architecture (checkpoint version)")
            model = SingleTaskGATv2ReceiverCheckpoint(
                in_dim=checkpoint["in_dim"],
                hidden=checkpoint["hidden"],
                heads=checkpoint["heads"],
                dropout=checkpoint["dropout"]
            ).to(self.device)
            
            # Load state dict
            model.load_state_dict(checkpoint["model_state"])
            print(f"✅ Receiver model loaded from {checkpoint_path}")
        
        return model

    def _load_shot_model(self):
        """Load the trained shot prediction model."""
        # Look for the best shot model checkpoint
        checkpoint_paths = [
            os.path.join(self.root, "best_gnn_shot_fold1.pt"),
            os.path.join(self.root, "best_gnn_multitask_fold1.pt")
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError("No trained shot model checkpoint found")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with checkpoint parameters (matching the checkpoint architecture)
        model = SingleTaskGATv2ShotCheckpoint(
            in_dim=checkpoint["in_dim"],
            hidden=checkpoint["hidden"],
            heads=checkpoint["heads"],
            dropout=checkpoint["dropout"]
        ).to(self.device)
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Shot model loaded from {checkpoint_path}")
        
        return model

    def _build_corner_graph(self, corner_data: pd.Series, players_data: pd.DataFrame) -> Data:
        """
        Build a graph representation for a corner event.
        
        Args:
            corner_data: DataFrame row containing corner event data
            players_data: DataFrame containing player positions for this corner
            
        Returns:
            PyG Data object representing the corner graph
        """
        corner_id = corner_data.get("corner_id")
        ev_players = players_data[players_data["corner_id"] == corner_id]
        
        if len(ev_players) == 0:
            raise ValueError(f"No players found for corner_id {corner_id}")
        
        # Ball node (last index)
        ball_x = _safe_float(corner_data.get("contact_location_x"), 
                            _safe_float(corner_data.get("header_location_x"), 
                                       _safe_float(corner_data.get("x"), 0.0)))
        ball_y = _safe_float(corner_data.get("contact_location_y"), 
                            _safe_float(corner_data.get("header_location_y"), 
                                       _safe_float(corner_data.get("y"), 50.0)))
        
        # Normalize to [0,1]
        ball_xn = np.clip(ball_x / PITCH_X, 0.0, 1.0)
        ball_yn = np.clip(ball_y / PITCH_Y, 0.0, 1.0)
        
        # Player nodes
        node_feats: List[List[float]] = []
        player_ids: List[int] = []
        positions: List[Tuple[float, float]] = []
        
        for _, prow in ev_players.iterrows():
            px = _safe_float(prow.get("x"))
            py = _safe_float(prow.get("y"))
            vx = _safe_float(prow.get("velocity_x"), default=None)
            vy = _safe_float(prow.get("velocity_y"), default=None)
            if vx is None or vy is None:
                vm = _safe_float(prow.get("velocity_magnitude"))
                vx, vy = vm, 0.0
            is_attacker = int(_safe_float(prow.get("is_attacking_team"), 0.0))
            is_goalkeeper = int(_safe_float(prow.get("is_goalkeeper"), 0.0))
            height_cm = _safe_float(prow.get("height_cm"))
            header_proxy = _safe_float(prow.get("header_ability_proxy"))
            team_encoding = _safe_float(prow.get("team_encoding"), is_attacker)
            role_encoding = _safe_float(prow.get("role_encoding"))
            pace = _safe_float(prow.get("pace"))
            jumping = _safe_float(prow.get("jumping"))
            heading = _safe_float(prow.get("heading"))
            minute_bucket = _safe_float(prow.get("minute_bucket"))
            
            # Normalize coordinates [0,1]
            xn = np.clip(px / PITCH_X, 0.0, 1.0)
            yn = np.clip(py / PITCH_Y, 0.0, 1.0)
            
            # Distances normalized by pitch diagonal
            dist_ball = math.hypot(px - ball_x, py - ball_y) / math.hypot(PITCH_X, PITCH_Y)
            dist_goal = math.hypot(PITCH_X - px, 50.0 - py) / math.hypot(PITCH_X, PITCH_Y)
            
            # Missing value handling - mark missing flags
            height_missing = 1.0 if np.isnan(height_cm) or height_cm == 0.0 else 0.0
            header_missing = 1.0 if np.isnan(header_proxy) or header_proxy == 0.0 else 0.0
            pace_missing = 1.0 if np.isnan(pace) or pace == 0.0 else 0.0
            jumping_missing = 1.0 if np.isnan(jumping) or jumping == 0.0 else 0.0
            heading_missing = 1.0 if np.isnan(heading) or heading == 0.0 else 0.0
            
            feat = [
                xn, yn, vx, vy,
                float(is_attacker), float(is_goalkeeper),
                dist_ball, dist_goal,
                role_encoding, height_cm, header_proxy,
                team_encoding, pace, jumping, heading,
                minute_bucket,
                height_missing, header_missing, pace_missing, jumping_missing, heading_missing
            ]
            node_feats.append(feat)
            player_ids.append(int(_safe_float(prow.get("player_id"), -1)))
            positions.append((px, py))
        
        # Ball node features appended to end
        delivery_type = str(corner_data.get("delivery_type")) if "delivery_type" in corner_data else "unknown"
        # crude speed proxy as mean of |vx| over players
        avg_speed = float(np.mean([abs(f[2]) + abs(f[3]) for f in node_feats]) / 2.0) if node_feats else 0.0
        
        # Delivery type encoding (inswing/outswing)
        delivery_type_encoding = 1.0 if delivery_type.lower() == "inswing" else 0.0
        
        ball_feat = [
            ball_xn, ball_yn, avg_speed, delivery_type_encoding,
            0.0, 0.0,  # is_attacker, is_goalkeeper
            0.0, 0.0,  # dist_ball, dist_goal (not meaningful for ball)
            0.0, 0.0, 0.0,  # role, height, header
            0.0, 0.0, 0.0, 0.0,  # team, pace, jumping, heading
            0.0,  # minute_bucket
            0.0, 0.0, 0.0, 0.0, 0.0  # missing flags
        ]
        node_feats.append(ball_feat)
        
        # Edges: within radius and to ball
        num_players = len(player_ids)
        ball_idx = num_players  # last
        edge_index: List[Tuple[int, int]] = []
        edge_attr: List[List[float]] = []  # Add edge attributes
        
        # Connect to ball with edge attributes (4 features to match checkpoint)
        for i in range(num_players):
            px, py = positions[i]
            dist = math.hypot(px - ball_x, py - ball_y)
            angle_goal = math.atan2(GOAL_Y - py, GOAL_X - px)
            same_team = 0.0  # ball
            marking_flag = 0.0
            edge_index.append((i, ball_idx))
            edge_index.append((ball_idx, i))
            # Only use 4 edge features to match checkpoint
            edge_attr.append([dist, angle_goal, same_team, marking_flag])
            edge_attr.append([dist, angle_goal, same_team, marking_flag])
        
        # Player-player edges by radius (15m as in training)
        R = 15.0
        for i in range(num_players):
            xi, yi = positions[i]
            for j in range(i + 1, num_players):
                xj, yj = positions[j]
                d = math.hypot(xi - xj, yi - yj)
                if d <= R:
                    ang_i = math.atan2(GOAL_Y - yi, GOAL_X - xi)
                    ang_j = math.atan2(GOAL_Y - yj, GOAL_X - xj)
                    same_team = 1.0 if 1.0 == 1.0 else 0.0  # Simplified - assuming all players are on same team for this context
                    marking_flag = 0.0
                    edge_index.append((i, j))
                    edge_index.append((j, i))
                    # Only use 4 edge features to match checkpoint
                    edge_attr.append([d, ang_i, same_team, marking_flag])
                    edge_attr.append([d, ang_j, same_team, marking_flag])
        
        x = torch.tensor(node_feats, dtype=torch.float)
        if len(edge_index) > 0:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_t = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index_t = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_t = torch.zeros((0, 4), dtype=torch.float)  # Use 4 features to match checkpoint
        
        # Create graph data
        data = Data(
            x=x,
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            num_nodes=x.size(0),
        )
        
        # Attach metadata
        data.corner_id = int(_safe_float(corner_id, -1))
        data.ball_index = int(ball_idx)
        data.player_positions = positions  # Store positions for strategy analysis
        data.player_ids = player_ids  # Store player IDs
        
        return data

    def _get_delivery_type(self, player_x: float, player_y: float, goal_x: float = GOAL_X, goal_y: float = GOAL_Y) -> str:
        """
        Determine delivery type based on player position relative to goal.
        
        Args:
            player_x: Player's x-coordinate
            player_y: Player's y-coordinate
            goal_x: Goal's x-coordinate (default 100)
            goal_y: Goal's y-coordinate (default 50)
            
        Returns:
            Delivery type description
        """
        # Calculate angle to goal center
        angle_to_goal = math.degrees(math.atan2(goal_y - player_y, goal_x - player_x))
        distance_to_goal = math.hypot(goal_x - player_x, goal_y - player_y)
        
        # Determine delivery type
        if player_x < goal_x / 2:
            delivery = "Near Post"
        elif player_x > goal_x / 2:
            delivery = "Far Post"
        else:
            delivery = "Center"
            
        # Add distance modifier
        if distance_to_goal < 8.0:
            delivery += " Short"
        elif distance_to_goal > 20.0:
            delivery += " Long"
            
        # Add angle modifier
        if abs(angle_to_goal) < 30:
            delivery += " Cut-back"
            
        return delivery

    def suggest_strategy(self, corner_id: int, corner_data_path: str = None, 
                        players_data_path: str = None, visualize: bool = True) -> Dict[str, Any]:
        """
        Suggest optimal corner strategy for a given corner event.
        
        Args:
            corner_id: ID of the corner event
            corner_data_path: Path to corner data CSV (optional, uses default if not provided)
            players_data_path: Path to player positions CSV (optional, uses default if not provided)
            visualize: Whether to generate visualization (default True)
            
        Returns:
            Dictionary containing strategy recommendation
        """
        # Load data if paths not provided
        if corner_data_path is None:
            corner_data_path = os.path.join(self.root, "processed_csv", "corner_data.csv")
        if players_data_path is None:
            players_data_path = os.path.join(self.root, "processed_csv", "player_positions.csv")
            
        corner_df = pd.read_csv(corner_data_path)
        players_df = pd.read_csv(players_data_path)
        
        # Find the corner event
        corner_event = corner_df[corner_df["corner_id"] == corner_id]
        if len(corner_event) == 0:
            raise ValueError(f"Corner event with ID {corner_id} not found")
        
        corner_row = corner_event.iloc[0]
        
        # Build graph for this corner
        graph = self._build_corner_graph(corner_row, players_df)
        graph = graph.to(self.device)
        
        # Get receiver probabilities
        with torch.no_grad():
            receiver_logits = self.receiver_model(graph)
            receiver_probs = torch.sigmoid(receiver_logits).squeeze()
            
        # Get player indices (excluding ball node)
        num_players = len(graph.player_ids)
        player_indices = list(range(num_players))
        player_probs = receiver_probs[player_indices].cpu().numpy()
        
        # Rank players by probability
        ranked_players = sorted(
            [(i, prob, graph.player_positions[i]) for i, prob in enumerate(player_probs)],
            key=lambda x: x[1], reverse=True
        )
        
        # Get top players
        primary_target = ranked_players[0] if len(ranked_players) > 0 else None
        alternate_targets = ranked_players[1:4] if len(ranked_players) > 1 else []
        
        # Calculate cluster zone (average location of top 3 targets)
        top_targets = ranked_players[:3]
        if top_targets:
            avg_x = np.mean([pos[0] for _, _, pos in top_targets])
            avg_y = np.mean([pos[1] for _, _, pos in top_targets])
            cluster_zone = (avg_x, avg_y)
        else:
            cluster_zone = (0, 0)
        
        # Estimate shot success for each target
        strategy_scores = []
        
        for player_idx, receiver_prob, position in [primary_target] + alternate_targets:
            if player_idx is None:
                continue
                
            # Create a mock graph for shot prediction (simplified approach)
            # In a real implementation, we would combine player features with corner context
            player_x, player_y = position
            
            # Estimate shot probability (simplified)
            with torch.no_grad():
                # For this simplified version, we'll use the same graph but focus on shot prediction
                shot_logits = self.shot_model(graph)
                shot_prob = torch.sigmoid(shot_logits).item()
            
            # Calculate joint strategy score
            strategy_score = receiver_prob * shot_prob
            strategy_scores.append((player_idx, receiver_prob, shot_prob, strategy_score, position))
        
        # Sort by strategy score
        strategy_scores.sort(key=lambda x: x[3], reverse=True)
        
        # Generate strategy recommendations
        recommendations = []
        
        for player_idx, receiver_prob, shot_prob, strategy_score, position in strategy_scores[:3]:
            player_id = graph.player_ids[player_idx]
            player_x, player_y = position
            
            # Determine delivery type
            delivery_type = self._get_delivery_type(player_x, player_y)
            
            # Create recommendation description
            recommendation = f"{delivery_type} to Player #{player_id}"
            recommendations.append(recommendation)
        
        # Prepare result
        result = {
            "corner_id": int(corner_id),
            "best_strategy": recommendations[0] if recommendations else "No strategy available",
            "alternates": recommendations[1:] if len(recommendations) > 1 else [],
            "confidence": strategy_scores[0][3] if strategy_scores else 0.0,
            "details": {
                "primary_target": {
                    "player_id": graph.player_ids[primary_target[0]] if primary_target else None,
                    "receiver_probability": float(primary_target[1]) if primary_target else 0.0,
                    "position": primary_target[2] if primary_target else None
                } if primary_target else None,
                "alternate_targets": [
                    {
                        "player_id": graph.player_ids[player_idx],
                        "receiver_probability": float(prob),
                        "position": position
                    }
                    for player_idx, prob, position in alternate_targets
                ],
                "cluster_zone": {
                    "x": float(cluster_zone[0]),
                    "y": float(cluster_zone[1])
                }
            }
        }
        
        # 4️⃣ Generate visual data and visualize if requested
        if visualize:
            # Create visualization data
            player_positions_dict = {}
            for i, player_id in enumerate(graph.player_ids):
                if i < len(graph.player_positions):  # Exclude ball node
                    px, py = graph.player_positions[i]
                    player_positions_dict[player_id] = (px, py)
            
            strategy_data = {
                "player_positions": player_positions_dict,
                "primary_position": primary_target[2] if primary_target else (0, 0),
                "alternate_positions": [pos for _, _, pos in alternate_targets],
                "cluster_zone": cluster_zone,
                "best_strategy": recommendations[0] if recommendations else "No strategy available",
                "confidence": strategy_scores[0][3] if strategy_scores else 0.0
            }
            
            # 5️⃣ Visualize and save the strategy
            visualize_strategy(strategy_data, corner_id)
            
            # Generate enhanced animated replay if available
            try:
                from corner_replay_v2 import CornerReplayV2
                print("\n🎬 Generating enhanced animated corner replay...")
                replay = CornerReplayV2(strategy_data=result, speed="normal")
                # Note: In a full implementation, we would run the animation here
                # For now, we just show that it can be created
                print("✅ Enhanced animated replay ready for visualization")
            except ImportError:
                print("💡 Enhanced animated replay not available (corner_replay_v2.py not found)")
        
        return result

    def suggest_strategies_batch(self, corner_ids: List[int], 
                               corner_data_path: str = None,
                               players_data_path: str = None) -> List[Dict[str, Any]]:
        """
        Suggest strategies for multiple corner events.
        
        Args:
            corner_ids: List of corner event IDs
            corner_data_path: Path to corner data CSV
            players_data_path: Path to player positions CSV
            
        Returns:
            List of strategy recommendations
        """
        results = []
        for corner_id in corner_ids:
            try:
                strategy = self.suggest_strategy(corner_id, corner_data_path, players_data_path)
                results.append(strategy)
            except Exception as e:
                print(f"Warning: Could not generate strategy for corner {corner_id}: {e}")
                results.append({
                    "corner_id": int(corner_id),
                    "error": str(e)
                })
        
        return results

def main():
    """Main function to demonstrate strategy suggestion."""
    print("Corner Strategy Suggester")
    print("========================")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Initialize strategy suggester
        suggester = CornerStrategySuggester(current_dir, device="cpu")
        
        # Load sample corner data to get a valid corner ID
        corner_data_path = os.path.join(current_dir, "processed_csv", "corner_data.csv")
        if os.path.exists(corner_data_path):
            corner_df = pd.read_csv(corner_data_path)
            # Use the first corner ID as an example
            sample_corner_id = int(corner_df.iloc[0]["corner_id"])
            
            print(f"\nGenerating strategy for corner ID: {sample_corner_id}")
            
            # Suggest strategy
            strategy = suggester.suggest_strategy(sample_corner_id)
            
            # Display results
            print("\n🎯 Recommended Corner Strategy:")
            print(f"Corner ID: {strategy['corner_id']}")
            print(f"Best Strategy: {strategy['best_strategy']}")
            print(f"Confidence: {strategy['confidence']:.3f}")
            
            if strategy['alternates']:
                print("\nAlternate Strategies:")
                for i, alt in enumerate(strategy['alternates'], 1):
                    print(f"  {i}. {alt}")
            
            if strategy.get('details'):
                details = strategy['details']
                if details.get('primary_target'):
                    target = details['primary_target']
                    print(f"\nPrimary Target: Player #{target['player_id']}")
                    print(f"  Receiver Probability: {target['receiver_probability']:.3f}")
                    if target.get('position'):
                        pos = target['position']
                        print(f"  Position: ({pos[0]:.1f}, {pos[1]:.1f})")
                
                if details.get('alternate_targets'):
                    print("\nAlternate Targets:")
                    for target in details['alternate_targets']:
                        print(f"  Player #{target['player_id']}: {target['receiver_probability']:.3f}")
                
                if details.get('cluster_zone'):
                    zone = details['cluster_zone']
                    print(f"\nCluster Zone: ({zone['x']:.1f}, {zone['y']:.1f})")
            
            # Save to file
            output_path = os.path.join(current_dir, f"corner_strategy_{sample_corner_id}.json")
            with open(output_path, 'w') as f:
                json.dump(strategy, f, indent=2)
            print(f"\n✅ Strategy saved to {output_path}")
            
        else:
            print("❌ Corner data not found. Please ensure processed data exists.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)