#!/usr/bin/env python3
"""
Validation script for GNN implementation
This script checks that all components of the GNN implementation are correctly set up
"""

import os
import sys

# Add current directory to Python path to fix imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_file_exists(filepath):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "❌"
    print(f"{status} {filepath}")
    return exists

def validate_implementation():
    """Validate the GNN implementation"""
    print("=== GNN Implementation Validation ===\n")
    
    # Check required files
    required_files = [
        "models/gatv2_models.py",
        "gnn_dataset.py", 
        "gnn_train.py",
        "run_experiments.py",
        "generate_final_report.py",
        "test_gnn_implementation.py",
        "GNN_README.md"
    ]
    
    print("1. Checking required files:")
    all_files_exist = True
    for file in required_files:
        filepath = os.path.join(os.path.dirname(__file__), file)
        if not check_file_exists(filepath):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing!")
        return False
    
    print("\n✓ All required files are present\n")
    
    # Check Python imports
    print("2. Checking Python imports:")
    try:
        from models.gatv2_models import MultiTaskGATv2, SingleTaskGATv2Shot, SingleTaskGATv2Receiver
        print("✓ GATv2 models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import GATv2 models: {e}")
        return False
    
    try:
        from gnn_dataset import SetPieceGraphDataset
        print("✓ Dataset module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import dataset module: {e}")
        return False
        
    try:
        from gnn_train import train_and_eval, pairwise_ranking_loss
        print("✓ Training module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import training module: {e}")
        return False
    
    print("\n✓ All Python imports successful\n")
    
    # Check dependencies
    print("3. Checking dependencies:")
    dependencies = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn")  # Fixed import name
    ]
    
    all_deps_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"❌ {name} not available")
            all_deps_ok = False
    
    if not all_deps_ok:
        print("\n❌ Some dependencies are missing!")
        print("Please install required packages with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies available\n")
    
    # Summary
    print("=== VALIDATION COMPLETE ===")
    print("✅ Your GNN implementation is ready for use!")
    print("\nTo run experiments:")
    print("  python run_experiments.py")
    print("\nTo generate final reports:")
    print("  python generate_final_report.py")
    print("\nTo test the implementation:")
    print("  python test_gnn_implementation.py")
    
    return True

if __name__ == "__main__":
    success = validate_implementation()
    sys.exit(0 if success else 1)