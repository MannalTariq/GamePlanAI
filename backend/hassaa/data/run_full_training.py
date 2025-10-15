#!/usr/bin/env python3
"""
Full GNN Training Pipeline
Runs complete training with all epochs, all folds, and generates comprehensive reports.
"""

import os
import sys
import time
import torch
import numpy as np
from gnn_train import train_and_eval
from generate_final_report import generate_final_report

def run_full_training_pipeline():
    """Run the complete GNN training pipeline with full epochs and cross-validation"""
    print("="*60)
    print("  FULL GNN TRAINING PIPELINE")
    print("="*60)
    
    start_time = time.time()
    root = os.path.dirname(os.path.abspath(__file__))
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n1. Running dataset statistics...")
    try:
        from run_experiments import print_dataset_statistics
        print_dataset_statistics(root)
    except Exception as e:
        print(f"Warning: Could not print dataset statistics: {e}")
    
    print("\n2. Starting full GNN training...")
    print("   - Training all model variants (multitask + single-task)")
    print("   - Using 200 epochs with early stopping")
    print("   - Running all 5 cross-validation folds")
    print("   - Comprehensive evaluation metrics")
    
    try:
        # Run full training with optimal hyperparameters
        train_and_eval(
            root=root,
            batch_size=16,
            lr=1e-3,
            hidden=128,
            heads=4,
            dropout=0.2,
            radius=15.0
        )
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Generating comprehensive evaluation report...")
    try:
        generate_final_report(root)
        print("✅ Report generation completed!")
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")
    
    # List generated files
    print("\n4. Training artifacts generated:")
    generated_files = []
    for file_pattern in ["*.pt", "*report*.json", "val_preds*.json"]:
        import glob
        files = glob.glob(os.path.join(root, file_pattern))
        generated_files.extend(files)
    
    if generated_files:
        for file_path in sorted(generated_files):
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   📁 {os.path.basename(file_path)} ({file_size:.1f} KB)")
    else:
        print("   ⚠️  No training artifacts found")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n5. Training pipeline completed in {duration/60:.1f} minutes")
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Check model checkpoints: best_gnn_*.pt")
    print("2. Review evaluation reports: *_report.json")
    print("3. Test strategy suggestions: python strategy_suggester.py")
    
    return True

if __name__ == "__main__":
    print("Starting Full GNN Training Pipeline...")
    print("This will take significant time (30+ minutes)")
    
    # Ask for confirmation
    try:
        response = input("\nDo you want to proceed with full training? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        sys.exit(0)
    
    success = run_full_training_pipeline()
    sys.exit(0 if success else 1)