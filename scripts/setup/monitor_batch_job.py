#!/usr/bin/env python3
"""
Batch Job Monitor for Mac
========================

This script monitors the progress of batch training jobs on your Mac,
providing real-time updates on:
- Training progress
- Memory usage
- Estimated completion time
- System resources

Author: RagaSense AI Team
Date: 2024
"""

import os
import sys
import json
import time
import psutil
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchJobMonitor:
    """Monitor batch training jobs"""
    
    def __init__(self):
        self.checkpoint_dir = Path("ml/batch_training_checkpoints")
        self.log_file = Path("batch_job.log")
        self.last_checkpoint = None
        self.start_time = None
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not self.checkpoint_dir.exists():
            return None
        
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.json"))
        if not checkpoint_files:
            return None
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoint_files[-1]
    
    def get_training_progress(self):
        """Get current training progress"""
        latest_checkpoint = self.find_latest_checkpoint()
        
        if not latest_checkpoint:
            return None
        
        try:
            with open(latest_checkpoint, 'r') as f:
                checkpoint = json.load(f)
            
            return {
                'epoch': checkpoint.get('epoch', 0),
                'accuracy': checkpoint.get('accuracy', 0.0),
                'training_time': checkpoint.get('training_time', 0),
                'timestamp': checkpoint.get('timestamp', ''),
                'checkpoint_file': latest_checkpoint.name
            }
        except Exception as e:
            logger.error(f"Error reading checkpoint: {e}")
            return None
    
    def get_system_resources(self):
        """Get current system resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        # Check if training process is running
        training_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower() and 'working_improved_mve' in ' '.join(proc.cmdline()):
                    training_processes.append(proc.info)
            except:
                continue
        
        return {
            'memory': {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent
            },
            'cpu_percent': cpu_percent,
            'disk': {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': (disk.used / disk.total) * 100
            },
            'training_processes': training_processes
        }
    
    def estimate_completion_time(self, progress):
        """Estimate training completion time"""
        if not progress or progress['epoch'] == 0:
            return None
        
        # Estimate based on current progress
        epochs_per_hour = progress['epoch'] / (progress['training_time'] / 3600) if progress['training_time'] > 0 else 0
        
        if epochs_per_hour == 0:
            return None
        
        # Assume 200 epochs total (configurable)
        total_epochs = 200
        remaining_epochs = total_epochs - progress['epoch']
        remaining_hours = remaining_epochs / epochs_per_hour
        
        return {
            'epochs_per_hour': epochs_per_hour,
            'remaining_epochs': remaining_epochs,
            'remaining_hours': remaining_hours,
            'estimated_completion': datetime.now() + timedelta(hours=remaining_hours)
        }
    
    def display_status(self):
        """Display current training status"""
        print("\n" + "="*60)
        print("📊 BATCH TRAINING MONITOR")
        print("="*60)
        
        # Get progress
        progress = self.get_training_progress()
        if progress:
            print(f"🎯 Training Progress:")
            print(f"   Epoch: {progress['epoch']}/200")
            print(f"   Accuracy: {progress['accuracy']:.4f}")
            print(f"   Training Time: {progress['training_time']/3600:.1f} hours")
            print(f"   Last Checkpoint: {progress['checkpoint_file']}")
            
            # Estimate completion
            completion = self.estimate_completion_time(progress)
            if completion:
                print(f"   Epochs/Hour: {completion['epochs_per_hour']:.1f}")
                print(f"   Remaining: {completion['remaining_epochs']} epochs ({completion['remaining_hours']:.1f} hours)")
                print(f"   Estimated Completion: {completion['estimated_completion'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("⚠️ No training progress found")
        
        # Get system resources
        resources = self.get_system_resources()
        print(f"\n💻 System Resources:")
        print(f"   Memory: {resources['memory']['used_gb']:.1f}/{resources['memory']['total_gb']:.1f} GB ({resources['memory']['percent']:.1f}%)")
        print(f"   CPU: {resources['cpu_percent']:.1f}%")
        print(f"   Disk: {resources['disk']['used_gb']:.1f}/{resources['disk']['total_gb']:.1f} GB ({resources['disk']['percent']:.1f}%)")
        
        # Check training processes
        if resources['training_processes']:
            print(f"\n🚀 Training Processes:")
            for proc in resources['training_processes']:
                print(f"   PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}%, Memory {proc['memory_percent']:.1f}%")
        else:
            print(f"\n⚠️ No training processes detected")
        
        print("="*60)
    
    def monitor_continuously(self, interval=60):
        """Monitor training continuously"""
        print("🔍 Starting continuous monitoring...")
        print(f"Update interval: {interval} seconds")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                self.display_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
    
    def save_status_report(self):
        """Save status report to file"""
        progress = self.get_training_progress()
        resources = self.get_system_resources()
        completion = self.estimate_completion_time(progress) if progress else None
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'progress': progress,
            'resources': resources,
            'completion_estimate': completion
        }
        
        report_path = Path("ml/batch_training_status.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Status report saved: {report_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor batch training job")
    parser.add_argument("--continuous", action="store_true", help="Monitor continuously")
    parser.add_argument("--interval", type=int, default=60, help="Update interval in seconds")
    parser.add_argument("--save-report", action="store_true", help="Save status report to file")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = BatchJobMonitor()
    
    if args.continuous:
        # Continuous monitoring
        monitor.monitor_continuously(args.interval)
    else:
        # Single status check
        monitor.display_status()
        
        if args.save_report:
            monitor.save_status_report()
    
    return 0

if __name__ == "__main__":
    main()
