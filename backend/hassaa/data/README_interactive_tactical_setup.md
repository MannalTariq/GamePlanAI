# Interactive Tactical Setup for Corner Kick Visualization

This tool allows you to manually place players on a football pitch and generate corner kick strategies using machine learning models.

## Features

1. **Interactive Player Placement**:
   - Click on the pitch to place players
   - First clicks place attackers (red circles)
   - After 5 attackers, clicks place defenders (blue triangles)
   - After 4 defenders, next click places goalkeeper (purple square)

2. **Model Integration**:
   - Converts manual positions to graph structure for GNN model input
   - Uses trained receiver prediction model to identify best target
   - Generates complete corner kick strategy

3. **Enhanced Visualization**:
   - Styled player icons with glow effects
   - Tactical lines showing runs and ball path
   - Professional corner kick animation

4. **Output Generation**:
   - JSON strategy file with detailed player information
   - Summary report with key metrics
   - MP4 animation of the corner kick

## Usage

Run the interactive setup:

```bash
cd data
python interactive_tactical_setup.py
```

### Controls

- **Mouse Click**: Place players on the pitch
- **Done Button** or **Enter**: Generate strategy and animation
- **Undo Button** or **U/Z**: Remove last placed player
- **Reset Button** or **R**: Clear all players

## Output Files

The tool generates several output files:

1. `corner_strategy_manual_YYYYMMDD_HHMMSS.json` - Detailed strategy in JSON format
2. `corner_strategy_summary_YYYYMMDD_HHMMSS.txt` - Human-readable summary
3. `corner_animation_1_v4.mp4` - Animated corner kick visualization

## Technical Details

### Player Positioning

Players are placed in three categories:
- **Attackers** (red circles): Primary targets for the corner kick
- **Defenders** (blue triangles): Opposition players marking attackers
- **Goalkeeper** (purple square): Goalkeeper positioning

### Model Input Conversion

The manual positions are converted to a graph structure compatible with the GNN model:
- Node features include position, team, distance metrics
- Edges represent spatial relationships between players
- Ball node is added at the corner flag position

### Visualization Enhancements

The visualization includes:
- Styled player icons with team-specific colors and shapes
- Subtle glow effects for better visibility
- Tactical lines showing player runs and ball trajectory
- Professional camera angles focused on the penalty area

## Requirements

- Python 3.7+
- Matplotlib
- PyTorch
- PyTorch Geometric
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```