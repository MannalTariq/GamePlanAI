# Quick Start Guide - Interactive Tactical Setup

## 🚀 Getting Started

### Run the Application
```bash
cd c:\Users\DELL\Desktop\hassaa\data
python interactive_tactical_setup.py
```

## 🎮 Step-by-Step Usage

### Step 1: Select Corner Side
📍 **Choose which corner to take the kick from**

- **Bottom-Left**: ← Left button (default when first clicked)
- **Bottom-Right**: Right → button (application default)
- **Top-Left**: ← Left button (click twice)
- **Top-Right**: Right → button (click twice)

**Visual Feedback**:
- Orange indicator at top shows: `Corner Side: RIGHT (Bottom-Right Corner)`
- Yellow highlight circle appears at selected corner
- Red flag marker shows exact corner position

### Step 2: Place Players
🏃 **Click on the pitch to position players**

**Placement Order**:
1. **Attackers** (up to 9) - Red circles
   - Click anywhere on pitch
   - Place in penalty area, near post, far post
   
2. **Defenders** (up to 10) - Gray triangles
   - Placement starts automatically after 9 attackers
   - Position to mark attackers
   
3. **Goalkeeper** (1 only) - Green square
   - Placement starts after 10 defenders
   - Position near goal line

**Tips**:
- Minimum 6 players needed for simulation
- Use `Undo` button to remove last player
- Use `Reset` button to clear all and start over

### Step 3: Generate Strategy
🧠 **Let the GNN analyze your formation**

**Click "Generate & Simulate" button**

The system will:
1. Analyze all player positions
2. Calculate distances to goal
3. Detect unmarked players
4. Generate GNN predictions
5. Determine tactical decision

**What You'll See**:
```
🎯 GNN PREDICTION RESULTS:
   Primary Receiver: Player #100001 (Score: 160%)
   Shot Confidence: 50%
   Tactical Decision: 'Powerful header attempt'
```

### Step 4: Watch Simulation
🎬 **Enjoy the animated corner kick**

**Animation Sequence**:
1. **Ball Flight** (3 seconds)
   - Ball flies from selected corner to target
   - Bezier curve creates realistic arc
   - White trail shows ball path

2. **Player Movement** (2 seconds)
   - Attackers run toward target
   - Defenders track attackers
   - Goalkeeper adjusts position

3. **Shot Outcome** (1 second)
   - Primary receiver attempts action
   - Tactical decision displayed
   - Goal/save result shown

### Step 5: Experiment!
🔄 **Try different scenarios**

**Test Different Corners**:
- Bottom corners → Low crosses
- Top corners → High deliveries
- Left side → Right-to-left attack
- Right side → Left-to-right attack

**Test Different Formations**:
- Crowded box → "Powerful header attempt"
- Spread formation → "Targeted delivery"
- Deep positions → "Build-up play"
- Wide positions → "Cross for incoming runner"

## 🎯 Visual Guide

### Screen Layout
```
┌────────────────────────────────────────────────────────┐
│ Corner: RIGHT (Bottom-Right)  [Orange Box]            │ Top
│ Mode: Attacker                [White Box]             │
│ Attackers: 4/9 | Defenders: 3/10 | GK: 1/1  [Yellow] │
│ Instructions...               [Light Blue]            │
├────────────────────────────────────────────────────────┤
│                                                        │
│              [FOOTBALL PITCH WITH PLAYERS]             │
│                                                        │
│         ⚽ [Yellow highlight at corner]                │
│                                                        │
├────────────────────────────────────────────────────────┤
│ [← Left] [Right →]    [Reset] [Undo] [Generate & Sim] │ Bottom
└────────────────────────────────────────────────────────┘
```

### Player Color Guide
| Symbol | Type | Color | Description |
|--------|------|-------|-------------|
| ⭕ | Attacker (Primary) | Red | Selected receiver |
| ⭕ | Attacker (Alternate) | Orange | Backup option |
| ⭕ | Attacker (Other) | Blue | Support player |
| 🔺 | Defender | Gray | Marking opponent |
| ⬜ | Goalkeeper | Green | Goal protection |

### Corner Position Guide
```
     Top-Left              Top-Right
        (0,68)               (105,68)
          ┌─────────────────────┐
          │                     │
          │   [FOOTBALL PITCH]  │
          │                     │
          └─────────────────────┘
     Bottom-Left          Bottom-Right
        (0,0)                (105,0)
                           ⚽ DEFAULT
```

## 💡 Tips & Tricks

### For Best Results:
1. **Start with corner selection** - Set the corner before placing players
2. **Mix formations** - Don't always crowd the penalty area
3. **Test unmarked positions** - Place attackers away from defenders
4. **Vary distances** - Mix near/far post positioning
5. **Observe patterns** - Note which formations trigger which decisions

### Common Mistakes:
❌ Placing all players in same spot  
❌ Not leaving space for runs  
❌ Ignoring goalkeeper positioning  
❌ Forgetting to select corner side  
❌ Trying to simulate with < 6 players  

### Pro Tips:
✅ Place one attacker very close to goal (< 10m) for header  
✅ Leave 1-2 attackers deep for short corner option  
✅ Position defenders to track specific attackers  
✅ Test all 4 corners with same formation  
✅ Screenshot strategies for comparison  

## 🎲 Sample Scenarios

### Scenario 1: Near Post Attack
```
Corner: Bottom-Right (105, 0)
Formation:
- 3 attackers clustered near post (6m area)
- 2 attackers far post
- 4 attackers mid-range
Expected: "Powerful header attempt" or "Near post delivery"
```

### Scenario 2: Spread Formation
```
Corner: Top-Left (0, 68)
Formation:
- Attackers evenly spread across box
- No clustering
- Good spacing (5m+ between players)
Expected: "Targeted delivery to best positioned player"
```

### Scenario 3: Deep Build-Up
```
Corner: Bottom-Left (0, 0)
Formation:
- Most attackers 20m+ from goal
- Only 1-2 in penalty area
- Wide positioning
Expected: "Build-up play from corner" or "Short corner variation"
```

## 📊 Understanding Decisions

### Decision Logic

**High Confidence Decisions**:
- Direct shot on first touch (Penalty area + high confidence)
- Powerful header attempt (Very close to goal)

**Medium Confidence Decisions**:
- Controlled header to teammate
- Far post header
- Quick shot after control

**Build-Up Decisions**:
- Build-up play from corner
- Recycle possession
- Short corner variation

**Crossing Decisions**:
- Targeted delivery
- Cross for incoming runner
- Whipped cross to back post

## 🐛 Troubleshooting

### Animation Not Showing?
- Ensure minimum 6 players placed
- Check console for error messages
- Try clicking "Generate & Simulate" again

### Wrong Corner Selected?
- Click corner button to toggle
- Check orange indicator at top
- Look for yellow highlight on pitch

### Can't Place More Players?
- Check if limit reached (9 attackers, 10 defenders, 1 GK)
- Use Undo to remove players
- Use Reset to start fresh

### Simulation Freezes?
- Close matplotlib window
- Restart application
- Reduce player count if issue persists

## 📞 Quick Reference

### Keyboard Shortcuts
- `U` or `Z` - Undo last player
- `R` - Reset all players
- `Enter` - Same as "Done" button

### Button Functions
- `← Left` - Select left corner (toggle top/bottom)
- `Right →` - Select right corner (toggle top/bottom)
- `Reset` - Clear all players
- `Undo` - Remove last player
- `Generate & Simulate` - Run GNN analysis and animation

### Corner Coordinates
- Bottom-Right: (105, 0) ← Default
- Top-Right: (105, 68)
- Bottom-Left: (0, 0)
- Top-Left: (0, 68)

## 🎉 Have Fun!

The system is designed to be intuitive and fun to use. Experiment with different formations, corner positions, and tactical setups to discover optimal corner kick strategies!

**Questions? Issues?**  
Check the detailed documentation:
- `CORNER_SIDE_SELECTION_FEATURE.md` - Corner selection guide
- `SESSION_IMPROVEMENTS_SUMMARY.md` - All recent improvements
- `ENHANCED_TACTICAL_CONTEXT_SUMMARY.md` - Tactical system details

Enjoy exploring corner kick tactics! ⚽
