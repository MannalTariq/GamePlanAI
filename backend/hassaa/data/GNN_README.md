# Graph Neural Network Implementation for Set-Piece Analysis

This implementation provides Graph Neural Network models using GATv2 to evaluate whether GNNs improve upon current tabular baselines for set-piece analysis in football/soccer.

## Features

### 1. Dual Prediction Tasks
- **Shot Prediction**: Graph-level classification (does the corner lead to a shot?)
- **Receiver Prediction**: Node-level ranking/classification (which attacker receives the ball?)

### 2. Enhanced Graph Construction
- **Nodes**: All players present at event + special "ball" node
- **Node Features**: 
  - Player position (x,y), velocity, is_attacker, is_goalkeeper
  - Normalized distance to ball and goal
  - Role encoding, height, header ability proxy
  - Team encoding, player skill embeddings (pace, jumping, heading)
  - Minute bucket embedding
  - Missing value flags for imputed features
- **Ball Node Features**: 
  - Ball position, speed proxy
  - Delivery type encoding (inswing/outswing)
  - Delivery target embedding
- **Edges**: 
  - Undirected edges between players within radius R
  - Connections between each player and ball node
  - K-nearest-neighbour edges for connectivity
- **Edge Features**: 
  - Euclidean distance
  - Relative angle to goal
  - Same team flag
  - Marking assignment flag

### 3. Model Architectures

#### GATv2 Encoder
- 3 GATv2Conv layers (128→128→64 hidden dimensions)
- Multi-head attention (4 heads for first two layers, 1 for final)
- Residual connections and LayerNorm after each layer
- Dropout (0.2) between layers

#### Task-Specific Heads
- **Shot Head**: Global attention pooling + MLP (64→32→1) with sigmoid
- **Receiver Head**: Node MLP (64→32→1) with sigmoid for binary or raw scores for ranking

#### Training Variants
- **Multi-task**: Shared encoder with two heads
- **Single-task**: Separate models for each task

### 4. Advanced Training Features
- **Loss Functions**:
  - Shot: BCEWithLogitsLoss with positive weighting
  - Receiver: Combination of pairwise ranking loss and BCE
- **Imbalance Handling**: Class weights and focal loss options
- **Regularization**: Label smoothing, L2 weight decay
- **Optimization**: AdamW with learning rate scheduling
- **Early Stopping**: Based on validation metrics

### 5. Comprehensive Evaluation
- **Shot Metrics**: Accuracy, precision, recall, F1, ROC-AUC, PR-AUC
- **Receiver Metrics**: Node-level Precision/Recall/F1, Top-k accuracy, MAP, NDCG
- **Calibration**: Reliability diagrams and ECE with temperature scaling
- **Cross-validation**: GroupKFold by match_id to prevent data leakage

## Implementation Files

```
data/
├── models/
│   └── gatv2_models.py          # GATv2 model architectures
├── gnn_dataset.py               # Graph dataset construction
├── gnn_train.py                 # Training and evaluation pipeline
├── run_experiments.py           # Hyperparameter search and final evaluation
├── generate_final_report.py     # Model comparison and reporting
├── test_gnn_implementation.py   # Unit tests for implementation
└── GNN_README.md                # This file
```

## Usage

### 1. Prerequisites
Ensure you have the required packages installed:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
The implementation expects processed CSV files in `data/processed_csv/`:
- `corner_data.csv`: Corner kick events with outcomes
- `player_positions.csv`: Player position data for each event
- `marking_assignments.csv`: Player marking assignments
- `freekick_data.csv`: Free kick events (optional)

### 3. Running Tests
Validate the implementation:
```bash
cd data
python test_gnn_implementation.py
```

### 4. Training Models
Run the complete training pipeline:
```bash
cd data
python run_experiments.py
```

This will:
1. Run hyperparameter search (simplified for demo)
2. Train multi-task and single-task GATv2 models
3. Evaluate and compare with tabular baselines
4. Generate comprehensive reports

### 5. Generating Final Reports
Create detailed comparison reports:
```bash
cd data
python generate_final_report.py
```

## Key Improvements Over Baseline

### 1. Enhanced Feature Engineering
- More comprehensive player attributes
- Missing value handling with explicit flags
- Better edge features including marking assignments
- Normalized spatial features

### 2. Advanced Model Architecture
- GATv2 for edge attention
- Residual connections for gradient flow
- Multi-head attention for representation diversity
- Layer normalization for training stability

### 3. Sophisticated Training
- Ranking loss for receiver prediction
- Class weighting for imbalanced data
- Temperature scaling for calibration
- Early stopping to prevent overfitting

### 4. Comprehensive Evaluation
- Multiple metrics for both tasks
- Calibration assessment
- Cross-validation with proper grouping
- Detailed model comparison

## Hyperparameters to Explore

- `hidden_dim`: [64, 128, 256]
- `num_heads`: [2, 4, 8]
- `lr`: [1e-2, 1e-3, 3e-4]
- `dropout`: [0.1, 0.3]
- `edge_radius`: [8, 12, 18]
- `negatives_per_positive`: [3, 7, 15]

## Expected Results

The GNN models should provide:
1. Improved shot prediction F1 score
2. Better receiver ranking (Top-k accuracy, MAP, NDCG)
3. More calibrated probability estimates
4. Robust performance across different match contexts

## Debugging and Sanity Checks

The implementation includes:
- Dataset statistics printing
- Sample graph visualization
- Class ratio monitoring
- Connectivity analysis
- Gradient clipping for training stability

If training loss drops but metrics stay flat:
- Increase node features
- Add marking flags
- Sample more negative pairs for ranking
- Increase model capacity

## Performance Comparison

The final report will compare:
- Tabular baseline vs GNN multi-task
- Tabular baseline vs GNN single-task
- Multi-task vs single-task GNNs

Metrics reported:
- Shot: F1, ROC-AUC, PR-AUC
- Receiver: Top-1/3/5 accuracy, MAP, NDCG
- Calibration: ECE scores

## Next Steps

After training, the implementation will:
1. Identify the best performing model variant
2. Save model checkpoints and normalizers
3. Generate per-event predictions with confidence scores
4. Provide actionable insights for tactical decision making