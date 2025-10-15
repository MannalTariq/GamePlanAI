#!/usr/bin/env python3
"""
Test script for the unified corner kick visualization
"""

from unified_corner_visualization import UnifiedCornerVisualization

def main():
    """Test the unified visualization"""
    print("Testing Unified Corner Kick Visualization...")
    print("Instructions:")
    print("1. Click on the pitch to place players")
    print("2. First clicks place attackers (red circles)")
    print("3. After 10 attackers, clicks place defenders (blue triangles)")
    print("4. After 10 defenders, clicks place goalkeepers (purple squares)")
    print("5. Press 'Done' to start simulation")
    print("6. Press 'Back' to adjust placements")
    print("7. Press 'Reset' to start over")
    print("8. Press ESC to quit")
    
    # Create and run visualization
    try:
        visualization = UnifiedCornerVisualization()
        visualization.run()
        print("Unified visualization completed successfully!")
    except Exception as e:
        print(f"Error running unified visualization: {e}")
        return False
        
    return True

if __name__ == "__main__":
    main()