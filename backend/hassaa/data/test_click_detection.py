#!/usr/bin/env python3
"""
Test Click Detection for Interactive Setup
Quick test to verify mouse click events are working properly.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_click_detection():
    """Simple test for click detection"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Setup simple pitch
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect('equal')
    ax.add_patch(patches.Rectangle((0, 0), 105, 68, linewidth=2, edgecolor='white', facecolor='#234d20'))
    ax.axis('off')
    ax.set_title("Click Detection Test - Click anywhere on the green area")
    
    clicks = []
    
    def on_click(event):
        if event.inaxes != ax:
            print(f"Click outside main area: {event.inaxes}")
            return
            
        if event.button != 1:
            print(f"Not left click: button={event.button}")
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            print(f"Invalid coordinates: ({x}, {y})")
            return
            
        print(f"✅ VALID CLICK at ({x:.1f}, {y:.1f})")
        clicks.append((x, y))
        
        # Draw a circle at click location
        circle = patches.Circle((x, y), radius=2, facecolor='red', edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        
        # Add click number
        ax.text(x, y+3, str(len(clicks)), color='white', ha='center', va='center', fontsize=10, weight='bold')
        
        fig.canvas.draw()
        
        if len(clicks) >= 5:
            print(f"Test complete! {len(clicks)} clicks detected.")
            print("Click locations:", clicks)
    
    # Connect event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("Starting click detection test...")
    print("Instructions:")
    print("1. Click anywhere on the green pitch area")
    print("2. You should see red circles appear where you click")
    print("3. Console should show coordinates")
    print("4. Test will complete after 5 clicks")
    
    plt.show()

if __name__ == "__main__":
    test_click_detection()