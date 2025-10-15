#!/usr/bin/env python3
"""
Complete pipeline execution script
This script demonstrates how to run the entire GNN implementation pipeline
"""

import os
import sys
import time
import json

def print_section_header(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_prerequisites():
    """Check that all prerequisites are met"""
    print_section_header("CHECKING PREREQUISITES")
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        "torch", "torch_geometric", "numpy", "pandas", 
        "sklearn", "matplotlib", "seaborn"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    # Check data files
    required_files = [
        "processed_csv/corner_data.csv",
        "processed_csv/player_positions.csv",
        "processed_csv/marking_assignments.csv"
    ]
    
    missing_files = []
    for file in required_files:
        filepath = os.path.join(os.path.dirname(__file__), file)
        if not os.path.exists(filepath):
            print(f"❌ {file} missing")
            missing_files.append(file)
        else:
            print(f"✓ {file} found")
    
    if missing_files:
        print(f"\nMissing data files: {', '.join(missing_files)}")
        print("Please run data preprocessing first")
        return False
    
    print("\n✓ All prerequisites met")
    return True

def run_model_validation():
    """Run model validation"""
    print_section_header("VALIDATING MODEL IMPLEMENTATION")
    
    try:
        from validate_implementation import validate_implementation
        success = validate_implementation()
        return success
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def run_unit_tests():
    """Run unit tests"""
    print_section_header("RUNNING UNIT TESTS")
    
    try:
        from test_gnn_implementation import run_all_tests
        success = run_all_tests()
        return success
    except Exception as e:
        print(f"❌ Unit tests failed: {e}")
        return False

def demonstrate_model_creation():
    """Demonstrate model creation and basic functionality"""
    print_section_header("DEMONSTRATING MODEL CREATION")
    
    try:
        import torch
        from models.gatv2_models import MultiTaskGATv2, SingleTaskGATv2Shot, SingleTaskGATv2Receiver
        
        # Create models
        multitask_model = MultiTaskGATv2(in_dim=21, hidden=64, heads=4, dropout=0.2, edge_dim=4)
        shot_model = SingleTaskGATv2Shot(in_dim=21, hidden=64, heads=4, dropout=0.2, edge_dim=4)
        receiver_model = SingleTaskGATv2Receiver(in_dim=21, hidden=64, heads=4, dropout=0.2, edge_dim=4)
        
        print("✓ MultiTaskGATv2 model created")
        print("✓ SingleTaskGATv2Shot model created")
        print("✓ SingleTaskGATv2Receiver model created")
        
        # Count parameters
        multitask_params = sum(p.numel() for p in multitask_model.parameters())
        shot_params = sum(p.numel() for p in shot_model.parameters())
        receiver_params = sum(p.numel() for p in receiver_model.parameters())
        
        print(f"\nModel parameter counts:")
        print(f"  MultiTaskGATv2: {multitask_params:,}")
        print(f"  SingleTaskGATv2Shot: {shot_params:,}")
        print(f"  SingleTaskGATv2Receiver: {receiver_params:,}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation demonstration failed: {e}")
        return False

def demonstrate_dataset_loading():
    """Demonstrate dataset loading"""
    print_section_header("DEMONSTRATING DATASET LOADING")
    
    try:
        from gnn_dataset import SetPieceGraphDataset
        
        # Try to load a small sample of the dataset
        root = os.path.dirname(__file__)
        dataset = SetPieceGraphDataset(
            root=root,
            is_train=True,
            radius_m=15.0,
            knn_k=4
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of graphs: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample graph nodes: {sample.num_nodes}")
            print(f"  Sample graph edges: {sample.edge_index.size(1)}")
            print(f"  Node features: {sample.x.shape[1]}")
            print(f"  Shot label: {sample.y_shot.item()}")
            print(f"  Receiver positives: {int(sample.y_receiver.sum().item())}")
        
        return True
    except FileNotFoundError:
        print("⚠ Dataset files not found - run preprocessing first")
        return True  # This is expected if data isn't preprocessed
    except Exception as e:
        print(f"❌ Dataset loading demonstration failed: {e}")
        return False

def print_execution_plan():
    """Print the complete execution plan"""
    print_section_header("COMPLETE EXECUTION PLAN")
    
    plan = [
        "1. Data preprocessing (if not already done)",
        "2. Model validation and unit tests",
        "3. Hyperparameter search",
        "4. Model training (multi-task and single-task variants)",
        "5. Model evaluation and comparison",
        "6. Final report generation",
        "7. Production model artifact creation"
    ]
    
    for step in plan:
        print(step)
    
    print("\nFor full execution:")
    print("  python run_experiments.py")
    print("  python generate_final_report.py")

def main():
    """Main execution function"""
    print("GNN Implementation Complete Pipeline Demo")
    print("========================================")
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        return False
    
    # Run validations
    if not run_model_validation():
        print("\n❌ Model validation failed. Exiting.")
        return False
    
    # Run unit tests
    if not run_unit_tests():
        print("\n❌ Unit tests failed. Exiting.")
        return False
    
    # Demonstrate key components
    demonstrations = [
        demonstrate_model_creation,
        demonstrate_dataset_loading
    ]
    
    for demo in demonstrations:
        if not demo():
            print(f"\n❌ {demo.__name__} failed. Continuing anyway.")
    
    # Print execution plan
    print_execution_plan()
    
    end_time = time.time()
    print(f"\nDemo completed in {end_time - start_time:.2f} seconds")
    
    print("\n" + "="*60)
    print("  DEMO COMPLETE - YOUR IMPLEMENTATION IS READY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run full experiments: python run_experiments.py")
    print("2. Generate reports: python generate_final_report.py")
    print("3. Check documentation: GNN_README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)