import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import pyarrow.csv as csv
from functools import partial
import time
import os

def load_data():
    """Load all required CSV files using pyarrow for faster reading."""
    print("\n=== Step 1: Loading Data ===")
    start_time = time.time()
    # Resolve paths relative to project root
    data_dir = os.path.dirname(__file__)
    csv_dir = os.path.abspath(os.path.join(data_dir, '..', 'csv'))
    try:
        print("Attempting to load data with pyarrow...")
        # Load event data with pyarrow
        corner_events = pd.read_csv(os.path.join(csv_dir, 'corner_kick_events.csv'), engine='pyarrow')
        print("✓ Loaded corner_kick_events.csv")
        corner_outcomes = pd.read_csv(os.path.join(csv_dir, 'corner_outcomes.csv'), engine='pyarrow')
        print("✓ Loaded corner_outcomes.csv")
        freekick_events = pd.read_csv(os.path.join(csv_dir, 'free_kick_events.csv'), engine='pyarrow')
        print("✓ Loaded free_kick_events.csv")
        freekick_outcomes = pd.read_csv(os.path.join(csv_dir, 'freekick_outcomes.csv'), engine='pyarrow')
        print("✓ Loaded freekick_outcomes.csv")
        matches = pd.read_csv(os.path.join(csv_dir, 'matches.csv'), engine='pyarrow')
        print("✓ Loaded matches.csv")
        players = pd.read_csv(os.path.join(csv_dir, 'players.csv'), engine='pyarrow')
        print("✓ Loaded players.csv")
        player_positions = pd.read_csv(os.path.join(csv_dir, 'player_positions.csv'), engine='pyarrow')
        print("✓ Loaded player_positions.csv")
        marking_assignments = pd.read_csv(os.path.join(csv_dir, 'marking_assignments.csv'), engine='pyarrow')
        print("✓ Loaded marking_assignments.csv")
        
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        return {
            'corner_events': corner_events,
            'corner_outcomes': corner_outcomes,
            'freekick_events': freekick_events,
            'freekick_outcomes': freekick_outcomes,
            'matches': matches,
            'players': players,
            'player_positions': player_positions,
            'marking_assignments': marking_assignments
        }
    except Exception as e:
        print(f"Error with pyarrow, falling back to pandas: {e}")
        print("Attempting to load data with pandas...")
        # Fallback to pandas if pyarrow fails
        corner_events = pd.read_csv(os.path.join(csv_dir, 'corner_kick_events.csv'))
        print("✓ Loaded corner_kick_events.csv")
        corner_outcomes = pd.read_csv(os.path.join(csv_dir, 'corner_outcomes.csv'))
        print("✓ Loaded corner_outcomes.csv")
        freekick_events = pd.read_csv(os.path.join(csv_dir, 'free_kick_events.csv'))
        print("✓ Loaded free_kick_events.csv")
        freekick_outcomes = pd.read_csv(os.path.join(csv_dir, 'freekick_outcomes.csv'))
        print("✓ Loaded freekick_outcomes.csv")
        matches = pd.read_csv(os.path.join(csv_dir, 'matches.csv'))
        print("✓ Loaded matches.csv")
        players = pd.read_csv(os.path.join(csv_dir, 'players.csv'))
        print("✓ Loaded players.csv")
        player_positions = pd.read_csv(os.path.join(csv_dir, 'player_positions.csv'))
        print("✓ Loaded player_positions.csv")
        marking_assignments = pd.read_csv(os.path.join(csv_dir, 'marking_assignments.csv'))
        print("✓ Loaded marking_assignments.csv")
        
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        return {
            'corner_events': corner_events,
            'corner_outcomes': corner_outcomes,
            'freekick_events': freekick_events,
            'freekick_outcomes': freekick_outcomes,
            'matches': matches,
            'players': players,
            'player_positions': player_positions,
            'marking_assignments': marking_assignments
        }

def normalize_coordinates(df, x_col, y_col):
    """Normalize x and y coordinates to 0-100 scale using numpy for speed."""
    if x_col in df.columns and y_col in df.columns:
        df[x_col] = np.clip(df[x_col].values, 0, 100)
        df[y_col] = np.clip(df[y_col].values, 0, 100)
    return df

def merge_corner_data(corner_events, corner_outcomes):
    """Merge corner kick events with outcomes."""
    print("\n=== Step 2: Merging Corner Data ===")
    start_time = time.time()
    merged_data = pd.merge(corner_events, corner_outcomes, on='corner_id', how='left')
    print(f"✓ Merged {len(corner_events)} corner events with outcomes")
    print(f"Corner data merging completed in {time.time() - start_time:.2f} seconds")
    return merged_data

def merge_freekick_data(freekick_events, freekick_outcomes):
    """Merge free kick events with outcomes."""
    print("\n=== Step 3: Merging Free Kick Data ===")
    start_time = time.time()
    merged_data = pd.merge(freekick_events, freekick_outcomes, on='freekick_id', how='left')
    print(f"✓ Merged {len(freekick_events)} free kick events with outcomes")
    print(f"Free kick data merging completed in {time.time() - start_time:.2f} seconds")
    return merged_data

def engineer_features(player_positions, corner_events, freekick_events):
    """Engineer additional features for player positions."""
    print("\n=== Step 4: Feature Engineering ===")
    start_time = time.time()
    
    # Calculate velocity magnitude if velocity columns exist
    if all(col in player_positions.columns for col in ['velocity_x', 'velocity_y']):
        print("Calculating velocity magnitude...")
        player_positions['velocity_magnitude'] = np.sqrt(
            player_positions['velocity_x'].values**2 + 
            player_positions['velocity_y'].values**2
        )
        print("✓ Velocity magnitude calculated")
    
    # Create team mapping using numpy arrays for speed
    print("Creating team mapping...")
    team_mapping = {}
    
    # Add corner kick teams
    for _, event in corner_events.iterrows():
        team_mapping[('corner', event['corner_id'])] = event['team_taking']
    print("✓ Corner kick team mapping created")
    
    # Add free kick teams
    for _, event in freekick_events.iterrows():
        team_mapping[('freekick', event['freekick_id'])] = event['team_taking']
    print("✓ Free kick team mapping created")
    
    print("Adding team information to player positions...")
    # Add team information to player positions
    def get_team_taking(row):
        if pd.notna(row['corner_id']):
            return team_mapping.get(('corner', row['corner_id']))
        elif pd.notna(row['freekick_id']):
            return team_mapping.get(('freekick', row['freekick_id']))
        return None
    
    player_positions['team_taking'] = player_positions.apply(get_team_taking, axis=1)
    print("✓ Team information added to player positions")
    
    # Add team flag if team column exists
    if 'team' in player_positions.columns:
        print("Adding team flag...")
        player_positions['is_attacking_team'] = (player_positions['team'] == player_positions['team_taking']).astype(int)
        print("✓ Team flag added")
    else:
        print("Warning: 'team' column not found in player_positions")
    
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds")
    return player_positions

def create_graph_data_batch(events_batch, player_positions, marking_assignments):
    """Create graph data structures for a batch of events."""
    graphs = []
    for event_data in events_batch:
        # Filter player positions for this event
        if 'corner_id' in event_data:
            event_players = player_positions[
                (player_positions['corner_id'] == event_data['corner_id']) |
                (player_positions['freekick_id'].isna())
            ]
            event_markings = marking_assignments[
                marking_assignments['corner_id'] == event_data['corner_id']
            ]
        else:
            event_players = player_positions[
                (player_positions['freekick_id'] == event_data['freekick_id']) |
                (player_positions['corner_id'].isna())
            ]
            event_markings = marking_assignments[
                marking_assignments['freekick_id'] == event_data['freekick_id']
            ]
        
        # Create node features
        node_features = []
        for _, player in event_players.iterrows():
            features = [
                player['x'] if 'x' in player else 0,
                player['y'] if 'y' in player else 0,
                player['velocity_magnitude'] if 'velocity_magnitude' in player else 0,
                player['is_attacking_team'] if 'is_attacking_team' in player else 0
            ]
            node_features.append(features)
        
        # Create edge indices and attributes
        edge_index = []
        edge_attr = []
        
        # Add marking relationships
        for _, marking in event_markings.iterrows():
            if pd.notna(marking['defender_id']) and pd.notna(marking['attacker_id']):
                defender_idx = event_players[event_players['player_id'] == marking['defender_id']].index
                attacker_idx = event_players[event_players['player_id'] == marking['attacker_id']].index
                
                if len(defender_idx) > 0 and len(attacker_idx) > 0:
                    edge_index.append([defender_idx[0], attacker_idx[0]])
                    edge_attr.append([1 if marking['marking_type'] == 'man-marking' else 0])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.zeros((0, 1), dtype=torch.float)
        
        # Create target tensors
        y_receiver = torch.zeros(len(event_players), dtype=torch.float)
        y_shot = torch.tensor([event_data['is_shot']], dtype=torch.float)
        
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                         y_receiver=y_receiver, y_shot=y_shot))
    
    return graphs

def save_processed_data(corner_data, freekick_data, player_positions, marking_assignments, graph_data):
    """Save processed data in CSV format."""
    print("\n=== Step 7: Saving Processed Data ===")
    start_time = time.time()
    
    # Create processed_csv directory under data/
    data_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(data_dir, 'processed_csv')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    try:
        # Save corner data
        print("Saving corner data...")
        corner_data.to_csv(os.path.join(processed_dir, 'corner_data.csv'), index=False)
        print("✓ Corner data saved")
        
        # Save freekick data
        print("Saving freekick data...")
        freekick_data.to_csv(os.path.join(processed_dir, 'freekick_data.csv'), index=False)
        print("✓ Freekick data saved")
        
        # Save player positions
        print("Saving player positions...")
        player_positions.to_csv(os.path.join(processed_dir, 'player_positions.csv'), index=False)
        print("✓ Player positions saved")
        
        # Save marking assignments
        print("Saving marking assignments...")
        marking_assignments.to_csv(os.path.join(processed_dir, 'marking_assignments.csv'), index=False)
        print("✓ Marking assignments saved")
        
        # Save graph data features
        print("Saving graph data features...")
        graph_features = []
        for i, graph in enumerate(graph_data):
            features = {
                'graph_id': i,
                'num_nodes': graph.x.size(0),
                'num_edges': graph.edge_index.size(1) if graph.edge_index.size(1) > 0 else 0,
                'is_shot': graph.y_shot.item()
            }
            graph_features.append(features)
        
        pd.DataFrame(graph_features).to_csv(os.path.join(processed_dir, 'graph_features.csv'), index=False)
        print("✓ Graph features saved")
        
        print(f"Data saving completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        raise

def preprocess_pipeline():
    """Main preprocessing pipeline with parallel processing."""
    total_start_time = time.time()
    print("\n=== Starting Preprocessing Pipeline ===")
    
    # Load data
    data = load_data()
    
    # Merge event data with outcomes
    corner_data = merge_corner_data(data['corner_events'], data['corner_outcomes'])
    freekick_data = merge_freekick_data(data['freekick_events'], data['freekick_outcomes'])
    
    # Normalize coordinates
    print("\n=== Step 5: Normalizing Coordinates ===")
    start_time = time.time()
    corner_data = normalize_coordinates(corner_data, 'header_location_x', 'header_location_y')
    freekick_data = normalize_coordinates(freekick_data, 'contact_location_x', 'contact_location_y')
    print(f"✓ Coordinates normalized in {time.time() - start_time:.2f} seconds")
    
    # Engineer features
    data['player_positions'] = engineer_features(
        data['player_positions'],
        data['corner_events'],
        data['freekick_events']
    )
    
    # Create graph data in parallel
    print("\n=== Step 6: Creating Graph Data ===")
    start_time = time.time()
    print("Preparing events for processing...")
    all_events = []
    for _, event in corner_data.iterrows():
        all_events.append(event)
    for _, event in freekick_data.iterrows():
        all_events.append(event)
    
    # Process in smaller batches to manage memory
    batch_size = 100
    graph_data = []
    for i in range(0, len(all_events), batch_size):
        batch = all_events[i:i + batch_size]
        batch_graphs = create_graph_data_batch(batch, data['player_positions'], data['marking_assignments'])
        graph_data.extend(batch_graphs)
        print(f"Processed batch {i//batch_size + 1}/{(len(all_events) + batch_size - 1)//batch_size}")
    
    print(f"✓ Created {len(graph_data)} graph structures")
    print(f"Graph data creation completed in {time.time() - start_time:.2f} seconds")
    
    # Save processed data
    save_processed_data(corner_data, freekick_data, data['player_positions'], 
                       data['marking_assignments'], graph_data)
    
    print(f"\nTotal preprocessing completed in {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    preprocess_pipeline() 