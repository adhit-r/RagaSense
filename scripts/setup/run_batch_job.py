#!/usr/bin/env python3
"""
Simple Batch Job Runner for Mac
==============================

This script provides a simple way to run long training jobs on your Mac
with automatic checkpointing, resuming, and monitoring.

Usage:
    python3 scripts/run_batch_job.py --config ml/batch_config.json
    python3 scripts/run_batch_job.py --resume --checkpoint ml/checkpoints/latest.pth

Author: RagaSense AI Team
Date: 2024
"""

import os
import sys
import json
import time
import argparse
import signal
from pathlib import Path
import logging
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_job.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleBatchRunner:
    """Simple batch job runner for Mac"""
    
    def __init__(self):
        self.running = False
        self.process = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, stopping batch job...")
        self.running = False
        if self.process:
            self.process.terminate()
        sys.exit(0)
    
    def run_training_job(self, config_path=None, resume=False, checkpoint_path=None):
        """Run training job with automatic checkpointing"""
        logger.info("🚀 Starting batch training job")
        
        # Prepare command
        cmd = [sys.executable, "scripts/working_improved_mve.py"]
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        if resume and checkpoint_path:
            cmd.extend(["--resume", checkpoint_path])
        
        # Add batch mode flag
        cmd.append("--batch-mode")
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            # Start process
            self.running = True
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor process
            while self.running and self.process.poll() is None:
                output = self.process.stdout.readline()
                if output:
                    print(output.strip())
                    logger.info(output.strip())
                
                time.sleep(0.1)
            
            # Wait for completion
            return_code = self.process.wait()
            
            if return_code == 0:
                logger.info("✅ Batch training job completed successfully")
                return True
            else:
                logger.error(f"❌ Batch training job failed with return code {return_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running batch job: {e}")
            return False
    
    def run_overnight_training(self, hours=12):
        """Run training overnight with automatic scheduling"""
        logger.info(f"🌙 Starting overnight training for {hours} hours")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=hours)
        
        logger.info(f"Training will run from {start_time} to {end_time}")
        
        # Run training
        success = self.run_training_job()
        
        if success:
            logger.info("✅ Overnight training completed successfully")
        else:
            logger.info("⚠️ Overnight training completed with issues")
        
        return success

def create_batch_config():
    """Create optimized batch training configuration"""
    config = {
        "batch_training": {
            "enabled": True,
            "checkpoint_interval": 5,  # Save every 5 epochs
            "max_epochs": 200,
            "early_stopping_patience": 30
        },
        "memory_optimization": {
            "batch_size": 8,  # Smaller batch size for stability
            "gradient_accumulation": 4,  # Simulate larger batch size
            "mixed_precision": True,
            "gradient_checkpointing": True
        },
        "monitoring": {
            "log_interval": 50,
            "save_interval": 100,
            "memory_check_interval": 200
        },
        "scheduling": {
            "auto_pause_on_low_battery": True,
            "pause_during_night": False,
            "max_continuous_hours": 24
        }
    }
    
    config_path = Path("ml/batch_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"📝 Batch configuration created: {config_path}")
    return config_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run batch training job on Mac")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file path")
    parser.add_argument("--overnight", type=int, default=0, help="Run overnight for N hours")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_batch_config()
        return 0
    
    # Create batch runner
    runner = SimpleBatchRunner()
    
    if args.overnight > 0:
        # Run overnight training
        success = runner.run_overnight_training(args.overnight)
    else:
        # Run regular training
        success = runner.run_training_job(
            config_path=args.config,
            resume=args.resume,
            checkpoint_path=args.checkpoint
        )
    
    if success:
        print("\n✅ Batch job completed successfully!")
    else:
        print("\n❌ Batch job failed. Check logs for details.")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    main()
