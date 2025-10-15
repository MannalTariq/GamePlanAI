import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import pandas as pd
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, f1_score
from model import TacticalAI

def calculate_distance_angle(pos1, pos2):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    distance = np.sqrt(dx*dx + dy*dy)
    angle = np.arctan2(dy, dx)
    return distance, angle

def load_processed_data():
    print("Loading processed data...")
    data_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(data_dir, 'processed_csv')
    corner_data = pd.read_csv(os.path.join(processed_dir, 'corner_data.csv'))
    # freekick_data currently unused in training
    # freekick_data = pd.read_csv(os.path.join(processed_dir, 'freekick_data.csv'))
    player_positions = pd.read_csv(os.path.join(processed_dir, 'player_positions.csv'))
    marking_assignments = pd.read_csv(os.path.join(processed_dir, 'marking_assignments.csv'))
    # graph_features = pd.read_csv(os.path.join(processed_dir, 'graph_features.csv'))
    return {
        'corner_data': corner_data,
        'player_positions': player_positions,
        'marking_assignments': marking_assignments
    }

def _extract_node_features(player_row: pd.Series) -> list:
    features = []
    # Position
    features.append(float(player_row['x']) if 'x' in player_row and pd.notna(player_row['x']) else 0.0)
    features.append(float(player_row['y']) if 'y' in player_row and pd.notna(player_row['y']) else 0.0)
    # Velocity components if present; otherwise try magnitude split or zeros
    vx = float(player_row['velocity_x']) if 'velocity_x' in player_row and pd.notna(player_row.get('velocity_x')) else None
    vy = float(player_row['velocity_y']) if 'velocity_y' in player_row and pd.notna(player_row.get('velocity_y')) else None
    if vx is None or vy is None:
        vm = float(player_row['velocity_magnitude']) if 'velocity_magnitude' in player_row and pd.notna(player_row.get('velocity_magnitude')) else 0.0
        # Without direction components, put magnitude in vx and zero vy
        vx = vm
        vy = 0.0
    features.append(vx)
    features.append(vy)
    # Team encoding / attacker flag
    team_encoding = None
    if 'team_encoding' in player_row:
        team_encoding = float(player_row['team_encoding']) if pd.notna(player_row['team_encoding']) else 0.0
    elif 'is_attacking_team' in player_row:
        team_encoding = float(player_row['is_attacking_team']) if pd.notna(player_row['is_attacking_team']) else 0.0
    elif 'is_attacker' in player_row:
        team_encoding = float(player_row['is_attacker']) if pd.notna(player_row['is_attacker']) else 0.0
    else:
        team_encoding = 0.0
    features.append(team_encoding)
    # Explicit attacker indicator (last feature) required by model/eval; mirror team flag if specific col not present
    if 'is_attacker' in player_row:
        attacker_flag = float(player_row['is_attacker']) if pd.notna(player_row['is_attacker']) else 0.0
    else:
        attacker_flag = float(player_row.get('is_attacking_team', team_encoding)) if pd.notna(player_row.get('is_attacking_team', team_encoding)) else 0.0
    features.append(attacker_flag)
    return features

def create_graph_data(data_dict):
    print("\nStarting graph data creation...")
    start_time = time.time()
    graphs = []

    corner_data = data_dict['corner_data']
    player_positions = data_dict['player_positions']
    marking_assignments = data_dict['marking_assignments']

    for _, corner in corner_data.iterrows():
        event_id = corner['corner_id']
        corner_players = player_positions[player_positions['corner_id'] == event_id]
        if len(corner_players) == 0:
            continue

        node_features = []
        player_to_idx = {}
        for idx, (_, player) in enumerate(corner_players.iterrows()):
            features = _extract_node_features(player)
            node_features.append(features)
            player_to_idx[player['player_id']] = idx
        if not node_features:
            continue

        edge_index = []
        edge_attr = []
        event_markings = marking_assignments[marking_assignments['corner_id'] == event_id]
        for _, marking in event_markings.iterrows():
            defender_id = marking.get('defender_id')
            attacker_id = marking.get('attacker_id')
            if defender_id in player_to_idx and attacker_id in player_to_idx:
                def_pos = corner_players[corner_players['player_id'] == defender_id][['x','y']].values[0]
                att_pos = corner_players[corner_players['player_id'] == attacker_id][['x','y']].values[0]
                dist, angle = calculate_distance_angle(def_pos, att_pos)
                edge_index.append([player_to_idx[defender_id], player_to_idx[attacker_id]])
                edge_attr.append([dist, angle])
        if not edge_index:
            # Keep graphs even if no edges; model supports add_self_loops
            edge_index = []
            edge_attr = []

        x_tensor = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if len(edge_attr) > 0 else torch.zeros((0, 2), dtype=torch.float)

        # Receiver labels: closest player to header location if available
        receiver_labels = torch.zeros(len(node_features), dtype=torch.float)
        if 'header_location_x' in corner and 'header_location_y' in corner and pd.notna(corner['header_location_x']) and pd.notna(corner['header_location_y']):
            target_location = [corner['header_location_x'], corner['header_location_y']]
            closest_player_idx = None
            min_distance = float('inf')
            for idx_p, player in enumerate(corner_players.itertuples()):
                player_pos = [getattr(player, 'x'), getattr(player, 'y')]
                dist = calculate_distance_angle(player_pos, target_location)[0]
                if dist < min_distance:
                    min_distance = dist
                    closest_player_idx = idx_p
            if closest_player_idx is not None:
                receiver_labels[closest_player_idx] = 1.0

        # Shot/goal label if available
        is_goal = float(corner['is_goal']) if 'is_goal' in corner and pd.notna(corner['is_goal']) else 0.0
        is_clearance = float(corner['is_clearance']) if 'is_clearance' in corner and pd.notna(corner['is_clearance']) else 0.0
        y = torch.tensor([[is_goal, is_clearance]], dtype=torch.float)

        graph = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            receiver_labels=receiver_labels
        )
        graphs.append(graph)

    print(f"Created {len(graphs)} graphs in {time.time() - start_time:.2f}s")
    return graphs

def create_data_loaders(graphs, batch_size=32, val_split=0.2):
    if len(graphs) < 2:
        raise ValueError("Not enough graphs to split. Need at least 2.")
    num_val = max(1, int(len(graphs) * val_split))
    num_train = len(graphs) - num_val
    train_graphs = graphs[:num_train]
    val_graphs = graphs[num_train:]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    return train_loader, val_loader

def compute_shot_likelihood(shot_probs):
    return torch.sigmoid(shot_probs).mean().item() * 100  # percent

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_val_acc = 0
    best_metrics = {}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_receiver_preds = []
        all_receiver_labels = []
        all_shot_probs = []
        all_shot_labels = []

        for batch in train_loader:
            optimizer.zero_grad()
            receiver_probs, shot_probs, *_ = model(batch)
            receiver_labels = batch.receiver_labels.unsqueeze(1)
            shot_labels = batch.y[:, 0:1]  # is_goal

            # Receiver loss & shot loss
            receiver_loss = criterion(receiver_probs, receiver_labels)
            shot_loss = F.binary_cross_entropy(shot_probs, shot_labels)

            loss = receiver_loss + shot_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Metrics for this batch
            all_receiver_preds.extend(torch.sigmoid(receiver_probs).round().detach().cpu().numpy())
            all_receiver_labels.extend(receiver_labels.detach().cpu().numpy())
            all_shot_probs.extend(torch.sigmoid(shot_probs).detach().cpu().numpy())
            all_shot_labels.extend(shot_labels.detach().cpu().numpy())

        # Training metrics
        train_acc = accuracy_score(all_receiver_labels, all_receiver_preds)
        train_f1 = f1_score(all_receiver_labels, all_receiver_preds)
        avg_shot_likelihood = np.mean(all_shot_probs) * 100
        train_shot_pred = (np.array(all_shot_probs) > 0.5).astype(int)
        train_shot_acc = accuracy_score(all_shot_labels, train_shot_pred)
        train_shot_f1 = f1_score(all_shot_labels, train_shot_pred)

        # Validation
        model.eval()
        val_loss = 0
        all_val_receiver_preds = []
        all_val_receiver_labels = []
        all_val_shot_probs = []
        all_val_shot_labels = []
        with torch.no_grad():
            for batch in val_loader:
                receiver_probs, shot_probs, *_ = model(batch)
                receiver_labels = batch.receiver_labels.unsqueeze(1)
                shot_labels = batch.y[:, 0:1]
                receiver_loss = criterion(receiver_probs, receiver_labels)
                shot_loss = F.binary_cross_entropy(shot_probs, shot_labels)
                val_loss += (receiver_loss + shot_loss).item()
                all_val_receiver_preds.extend(torch.sigmoid(receiver_probs).round().detach().cpu().numpy())
                all_val_receiver_labels.extend(receiver_labels.detach().cpu().numpy())
                all_val_shot_probs.extend(torch.sigmoid(shot_probs).detach().cpu().numpy())
                all_val_shot_labels.extend(shot_labels.detach().cpu().numpy())

        val_acc = accuracy_score(all_val_receiver_labels, all_val_receiver_preds)
        val_f1 = f1_score(all_val_receiver_labels, all_val_receiver_preds)
        avg_val_shot_likelihood = np.mean(all_val_shot_probs) * 100
        val_shot_pred = (np.array(all_val_shot_probs) > 0.5).astype(int)
        val_shot_acc = accuracy_score(all_val_shot_labels, val_shot_pred)
        val_shot_f1 = f1_score(all_val_shot_labels, val_shot_pred)

        if epoch == num_epochs - 1:
            print(f"Receiver Accuracy: {val_acc:.4f}")
            print(f"Receiver F1 Score: {val_f1:.4f}")
            print(f"Shot Prediction Accuracy: {val_shot_acc:.4f}")
            print(f"Shot Prediction F1 Score: {val_shot_f1:.4f}")

        # Save best metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                "train_acc": train_acc, "train_f1": train_f1,
                "val_acc": val_acc, "val_f1": val_f1,
                "avg_shot_likelihood": avg_shot_likelihood,
                "avg_val_shot_likelihood": avg_val_shot_likelihood,
                "train_shot_acc": train_shot_acc, "train_shot_f1": train_shot_f1,
                "val_shot_acc": val_shot_acc, "val_shot_f1": val_shot_f1,
            }

    torch.save(model.state_dict(), 'tactical_ai_model.pt')
    # Clear summary of validation metrics
    if best_metrics:
        print("\n=== Validation Summary ===")
        print(f"Receiver - Acc: {best_metrics.get('val_acc', 0):.4f}, F1: {best_metrics.get('val_f1', 0):.4f}")
        print(f"Shot     - Acc: {best_metrics.get('val_shot_acc', 0):.4f}, F1: {best_metrics.get('val_shot_f1', 0):.4f}")

def main():
    print("Starting training process...")
    data_dict = load_processed_data()
    graphs = create_graph_data(data_dict)
    if not graphs:
        print("No valid graphs were created. Exiting.")
        return
    train_loader, val_loader = create_data_loaders(graphs, batch_size=16)
    num_features = graphs[0].x.size(1)
    model = TacticalAI(num_features=num_features, hidden_channels=128, num_heads=8, latent_dim=64)
    print("\nStarting model training...")
    train_model(model, train_loader, val_loader, num_epochs=150)
    torch.save(model.state_dict(), 'tactical_ai_model.pt')

if __name__ == "__main__":
    main()
