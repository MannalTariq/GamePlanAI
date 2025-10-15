# TacticalAI: Football Set-Piece Analysis

An AI-powered system for analyzing and optimizing football set-pieces (corner kicks and free kicks) using Graph Neural Networks and Variational Autoencoders.

## Features

- Predicts key receivers and shot probability for corner and free kicks
- Suggests defensive formation adjustments to reduce threats
- Provides explainable visualizations
- Invariant to team/color/mirroring

## Project Structure

```
.
├── preprocessing.py    # Data preprocessing pipeline
├── eda.py             # Exploratory data analysis
├── model.py           # Neural network architecture
├── train.py           # Training script
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Data

The system uses the following data sources:

- Corner kick events and outcomes
- Free kick events and outcomes
- Player positions and movements
- Marking assignments
- Match information
- Player metadata

## Model Architecture

The system uses a hybrid architecture combining:

1. Graph Attention Network (GATv2)
   - Processes player positions and movements
   - Predicts receiver probabilities
   - Estimates shot probability

2. Variational Autoencoder (VAE)
   - Learns latent representations of formations
   - Generates optimized defensive setups

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tactical-ai.git
cd tactical-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run exploratory data analysis:
```bash
python eda.py
```

2. Train the model:
```bash
python train.py
```

3. Train the new GATv2 multi-task model (receiver ranking + shot):
```bash
python -m data.gnn_train
```

The training script will:
- Load and preprocess the data
- Train the model
- Generate visualizations
- Save the trained model

## Output

The system generates several visualizations and artifacts:
- Delivery target heatmaps
- Marking type analysis
- Player feature correlations
- Training history
- Original vs. optimized formations
- GNN artifacts under `data/`: `sample_graph.json`, `best_gnn_fold*.pt`, `val_preds_fold*.json`, `gnn_report.json`

## Model Performance

The model is evaluated on:
- Receiver prediction accuracy
- Shot probability prediction
- Formation optimization effectiveness

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 