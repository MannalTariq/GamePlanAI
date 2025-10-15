#!/usr/bin/env python3
"""
Training Monitor - Check GNN training progress
"""

import os
import time
import glob

def monitor_training_progress(root_dir):
    """Monitor the training progress by checking for generated files"""
    
    print("=== GNN TRAINING MONITOR ===")
    print(f"Monitoring directory: {root_dir}")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # Check for training artifacts
            checkpoints = glob.glob(os.path.join(root_dir, "best_gnn_*.pt"))
            reports = glob.glob(os.path.join(root_dir, "*_report.json"))
            predictions = glob.glob(os.path.join(root_dir, "val_preds*.json"))
            
            print(f"\r⏰ {time.strftime('%H:%M:%S')} | ", end="")
            print(f"Checkpoints: {len(checkpoints)} | ", end="")
            print(f"Reports: {len(reports)} | ", end="")
            print(f"Predictions: {len(predictions)}", end="", flush=True)
            
            # If we have completed training artifacts, show them
            if checkpoints or reports:
                print("\n\n📁 Generated Training Artifacts:")
                for file_path in sorted(checkpoints + reports + predictions):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    file_time = time.ctime(os.path.getmtime(file_path))
                    print(f"   {os.path.basename(file_path)} ({file_size:.1f} KB) - {file_time}")
                
                if len(checkpoints) >= 3:  # All model types trained
                    print("\n✅ TRAINING APPEARS COMPLETE!")
                    print("🎯 You should now have trained models with:")
                    print("   - Accuracy and F1 scores")
                    print("   - ROC-AUC and PR-AUC metrics")
                    print("   - Strategy suggestions capability")
                    break
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped.")
        
        if checkpoints:
            print(f"\n📊 Current training status:")
            print(f"   Checkpoints found: {len(checkpoints)}")
            print(f"   Reports found: {len(reports)}")
            print("   Training is in progress...")
        else:
            print("   No training artifacts found yet.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    monitor_training_progress(current_dir)