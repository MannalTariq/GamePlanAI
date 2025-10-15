# Graph Neural Networks for Set-Piece Analysis

This repository contains a complete implementation of Graph Neural Network models using GATv2 to evaluate whether GNNs improve upon current tabular baselines for football/soccer set-piece analysis.

## Project Overview

The implementation addresses two key prediction tasks:
1. **Shot Prediction**: Graph-level classification (does the corner lead to a shot?)
2. **Receiver Prediction**: Node-level ranking/classification (which attacker receives the ball?)

## Repository Structure

```
.
├── data/
│   ├── models/
│   │   └── gatv2_models.py          # GATv2 model architectures
│   ├── gnn_dataset.py               # Graph dataset construction
│   ├── gnn_train.py                 # Training and evaluation pipeline
│   ├── run_experiments.py           # Hyperparameter search and training
│   ├── generate_final_report.py     # Model comparison and reporting
│   ├── test_gnn_implementation.py   # Unit tests
│   ├── validate_implementation.py   # Implementation validation
│   ├── run_complete_pipeline.py     # Complete pipeline execution
│   ├── GNN_README.md                # Detailed GNN documentation
│   └── IMPLEMENTATION_SUMMARY.md    # Implementation summary
├── csv/                             # Raw data files (not included in repo)
└── README.md                        # This file
```

## Key Features

### Enhanced Graph Construction
- **Nodes**: All players + ball node per set-piece
- **Node Features**: Position, velocity, team roles, physical attributes, skills
- **Edges**: Radius-based connections + k-NN for connectivity
- **Edge Features**: Distance, angle, team relationship, marking assignments

### Advanced Model Architectures
- **GATv2 Encoder**: 3-layer attention with residual connections
- **Multi-task Training**: Shared encoder with task-specific heads
- **Single-task Models**: Dedicated models for each prediction task

### Sophisticated Training
- **Loss Functions**: BCE + ranking loss for imbalanced data
- **Regularization**: Dropout, weight decay, gradient clipping
- **Optimization**: AdamW with learning rate scheduling
- **Evaluation**: Comprehensive metrics with proper cross-validation

## Implementation Status

✅ **Complete**: All required components have been implemented and validated

### Implemented Components
- [x] Enhanced GATv2 model architectures
- [x] Advanced graph dataset construction
- [x] Comprehensive training pipeline
- [x] Multi-task and single-task training variants
- [x] Hyperparameter search framework
- [x] Detailed evaluation and reporting
- [x] Unit tests and validation scripts
- [x] Documentation and usage instructions

## Getting Started

### Prerequisites
```bash
pip install -r data/requirements.txt
```

### Quick Validation
```bash
cd data
python validate_implementation.py
```

### Run Complete Pipeline Demo
```bash
cd data
python run_complete_pipeline.py
```

## Usage

### 1. Full Experimentation
```bash
cd data
python run_experiments.py
```

### 2. Generate Final Reports
```bash
cd data
python generate_final_report.py
```

### 3. Run Unit Tests
```bash
cd data
python test_gnn_implementation.py
```

## Expected Results

The GNN implementation should provide:
1. **Improved Shot Prediction**: Higher F1 scores and AUC metrics
2. **Better Receiver Ranking**: Higher Top-k accuracy, MAP, and NDCG
3. **More Calibrated Probabilities**: Better reliability diagrams and lower ECE
4. **Robust Generalization**: Consistent performance across match contexts

## Documentation

For detailed information about the implementation:
- [GNN_README.md](data/GNN_README.md): Comprehensive GNN documentation
- [IMPLEMENTATION_SUMMARY.md](data/IMPLEMENTATION_SUMMARY.md): Implementation summary
- Individual Python files: Inline documentation and comments

## Next Steps

1. Run hyperparameter search to optimize performance
2. Train final models with best configurations
3. Compare GNN performance with tabular baselines
4. Generate production-ready model artifacts
5. Create confidence reports for tactical decision making

## Support

For questions or issues with the implementation, please check the documentation or create an issue in the repository.