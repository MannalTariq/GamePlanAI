#!/usr/bin/env python3
"""
Restart GNN Training with Improved Configuration
This script stops the current training and restarts with fixed early stopping issues.
"""

import os
import signal
import subprocess
import time
import psutil

def kill_python_processes():
    """Kill existing Python training processes"""
    print("🛑 Stopping current training processes...")
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'gnn_train.py' in cmdline or 'run_full_training.py' in cmdline:
                    print(f"   Terminating process {proc.info['pid']}: {cmdline}")
                    proc.terminate()
                    proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass
    
    time.sleep(2)
    print("✅ Processes stopped")

def restart_training():
    """Restart training with improved configuration"""
    print("🔄 Restarting GNN training with improved configuration...")
    print("   ✅ Increased patience from 12 to 25 epochs")
    print("   ✅ Fixed receiver evaluation function")
    print("   ✅ Improved early stopping logic")
    print("   ✅ Better handling of zero metrics")
    
    # Change to data directory
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(data_dir)
    
    # Start training
    print(f"\n🚀 Starting training in: {data_dir}")
    print("   This will take 60-120 minutes for complete training...")
    
    try:
        # Use subprocess to run training in background
        process = subprocess.Popen(
            ['python', 'gnn_train.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"   Training process started with PID: {process.pid}")
        print("   Monitoring first few outputs...")
        
        # Show first few lines of output
        line_count = 0
        for line in process.stdout:
            print(f"   {line.strip()}")
            line_count += 1
            if line_count > 10:
                break
        
        print(f"\n✅ Training is running successfully!")
        print(f"   Process PID: {process.pid}")
        print("   You can monitor progress with:")
        print("   - python monitor_training.py")
        print("   - Check the terminal output")
        
        return process.pid
        
    except Exception as e:
        print(f"❌ Failed to start training: {e}")
        return None

def main():
    print("="*60)
    print("  RESTARTING GNN TRAINING WITH IMPROVEMENTS")
    print("="*60)
    
    # Kill existing processes
    kill_python_processes()
    
    # Restart training
    pid = restart_training()
    
    if pid:
        print(f"\n🎯 TRAINING RESTARTED SUCCESSFULLY!")
        print(f"   Process ID: {pid}")
        print("   Expected improvements:")
        print("   - Longer training (25 epochs patience instead of 12)")
        print("   - Better handling of evaluation metrics")
        print("   - More robust early stopping")
        print("   - Should achieve better accuracy and F1 scores")
        
        print(f"\n📊 Monitor progress with:")
        print(f"   python monitor_training.py")
    else:
        print(f"\n❌ Failed to restart training")
        print(f"   Try running manually: python gnn_train.py")

if __name__ == "__main__":
    main()