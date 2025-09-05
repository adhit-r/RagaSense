#!/usr/bin/env python3
"""
Mac Batch Training System for RagaSense
======================================

This script implements a robust batch training system optimized for Mac MPS
that can handle large-scale training (1000+ ragas) efficiently using:

1. Background processing
2. Checkpointing and resuming
3. Memory optimization
4. Progress monitoring
5. Automatic scheduling

Author: RagaSense AI Team
Date: 2024
"""

import os
import sys
import json
import time
import signal
import threading
import subprocess
from pathlib import Path
import logging
from datetime import datetime, timedelta
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mac_batch_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MacBatchTrainer:
    """Mac-optimized batch training system"""
    
    def __init__(self, config_path="ml/batch_training_config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # Setup device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🚀 Using device: {self.device}")
        
        # Training state
        self.training_active = False
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_start_time = None
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self):
        """Load batch training configuration"""
        default_config = {
            "model": {
                "n_classes": 1000,
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 6,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "max_epochs": 200
            },
            "data": {
                "dataset_path": "carnatic-hindustani-dataset",
                "target_ragas": 1000,
                "samples_per_raga": 10,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            },
            "training": {
                "checkpoint_interval": 10,
                "save_best_only": True,
                "early_stopping_patience": 20,
                "gradient_clip_val": 1.0,
                "weight_decay": 1e-4
            },
            "optimization": {
                "use_mixed_precision": True,
                "memory_efficient_attention": True,
                "gradient_checkpointing": True,
                "dataloader_workers": 4,
                "pin_memory": True
            },
            "monitoring": {
                "log_interval": 100,
                "save_interval": 1000,
                "memory_check_interval": 500,
                "max_memory_usage": 0.9  # 90% of available RAM
            },
            "scheduling": {
                "auto_pause_on_low_battery": True,
                "pause_during_night": False,
                "night_start": "22:00",
                "night_end": "06:00",
                "max_continuous_hours": 12
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = default_config
            self.save_config(config)
        
        return config
    
    def save_config(self, config=None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.training_active = False
        self.save_checkpoint(force=True)
        sys.exit(0)
    
    def check_system_resources(self):
        """Check if system has enough resources for training"""
        logger.info("🔍 Checking system resources...")
        
        # Check RAM
        memory = psutil.virtual_memory()
        ram_gb = memory.total / (1024**3)
        ram_available_gb = memory.available / (1024**3)
        ram_usage_percent = memory.percent / 100
        
        logger.info(f"   RAM: {ram_gb:.1f} GB total, {ram_available_gb:.1f} GB available ({ram_usage_percent:.1%} used)")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        disk_gb = disk.total / (1024**3)
        disk_available_gb = disk.free / (1024**3)
        disk_usage_percent = disk.used / disk.total
        
        logger.info(f"   Disk: {disk_gb:.1f} GB total, {disk_available_gb:.1f} GB available ({disk_usage_percent:.1%} used)")
        
        # Check battery (if available)
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = battery.percent
                battery_plugged = battery.power_plugged
                logger.info(f"   Battery: {battery_percent:.1f}% ({'plugged' if battery_plugged else 'unplugged'})")
                
                if not battery_plugged and battery_percent < 20:
                    logger.warning("⚠️ Low battery - consider plugging in for long training")
                    return False
        except:
            logger.info("   Battery: Not available")
        
        # Check if resources are sufficient
        if ram_usage_percent > 0.8:
            logger.warning("⚠️ High RAM usage - training may be slow")
        
        if disk_usage_percent > 0.9:
            logger.error("❌ Insufficient disk space")
            return False
        
        if ram_available_gb < 4:
            logger.error("❌ Insufficient RAM for training")
            return False
        
        logger.info("✅ System resources are sufficient for training")
        return True
    
    def create_optimized_model(self):
        """Create memory-optimized model for batch training"""
        from scripts.working_improved_mve import CulturalMusicTransformer
        
        model = CulturalMusicTransformer(
            n_classes=self.config['model']['n_classes'],
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_layers=self.config['model']['n_layers']
        ).to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config['optimization']['gradient_checkpointing']:
            model.gradient_checkpointing_enable()
        
        return model
    
    def create_optimized_dataloader(self, dataset, batch_size=None):
        """Create memory-optimized dataloader"""
        if batch_size is None:
            batch_size = self.config['model']['batch_size']
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['optimization']['dataloader_workers'],
            pin_memory=self.config['optimization']['pin_memory'],
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def train_epoch_batch(self, model, train_loader, optimizer, criterion, epoch):
        """Train one epoch with batch optimization"""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.config['optimization']['use_mixed_precision'] else None
        
        for batch_idx, (mel, f0, raga, tonic, tradition) in enumerate(train_loader):
            if not self.training_active:
                break
            
            mel, f0, raga, tonic, tradition = (
                mel.to(self.device, non_blocking=True),
                f0.to(self.device, non_blocking=True),
                raga.squeeze().to(self.device, non_blocking=True),
                tonic.squeeze().to(self.device, non_blocking=True),
                tradition.squeeze().to(self.device, non_blocking=True)
            )
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    raga_logits, tonic_pred, confidence = model(mel, f0, tradition)
                    loss = criterion(raga_logits, tonic_pred, confidence, raga, tonic)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['gradient_clip_val'])
                scaler.step(optimizer)
                scaler.update()
            else:
                raga_logits, tonic_pred, confidence = model(mel, f0, tradition)
                loss = criterion(raga_logits, tonic_pred, confidence, raga, tonic)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['gradient_clip_val'])
                optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(raga_logits, 1)
            correct_predictions += (predicted == raga).sum().item()
            total_predictions += raga.size(0)
            
            # Memory monitoring
            if batch_idx % self.config['monitoring']['memory_check_interval'] == 0:
                self.memory_monitor.check_memory_usage()
            
            # Logging
            if batch_idx % self.config['monitoring']['log_interval'] == 0:
                accuracy = correct_predictions / total_predictions
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")
        
        return total_loss / len(train_loader), correct_predictions / total_predictions
    
    def validate_epoch_batch(self, model, val_loader, criterion):
        """Validate one epoch with batch optimization"""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for mel, f0, raga, tonic, tradition in val_loader:
                mel, f0, raga, tonic, tradition = (
                    mel.to(self.device, non_blocking=True),
                    f0.to(self.device, non_blocking=True),
                    raga.squeeze().to(self.device, non_blocking=True),
                    tonic.squeeze().to(self.device, non_blocking=True),
                    tradition.squeeze().to(self.device, non_blocking=True)
                )
                
                raga_logits, tonic_pred, confidence = model(mel, f0, tradition)
                loss = criterion(raga_logits, tonic_pred, confidence, raga, tonic)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(raga_logits, 1)
                correct_predictions += (predicted == raga).sum().item()
                total_predictions += raga.size(0)
        
        return total_loss / len(val_loader), correct_predictions / total_predictions
    
    def save_checkpoint(self, epoch=None, accuracy=None, force=False):
        """Save training checkpoint"""
        if epoch is None:
            epoch = self.current_epoch
        
        if accuracy is None:
            accuracy = self.best_accuracy
        
        checkpoint_dir = Path("ml/batch_training_checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'accuracy': accuracy,
            'config': self.config,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint info
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save model state
        model_path = checkpoint_dir / f"model_epoch_{epoch}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'accuracy': accuracy
        }, model_path)
        
        logger.info(f"💾 Checkpoint saved: epoch {epoch}, accuracy {accuracy:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        if not checkpoint_path.exists():
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_accuracy = checkpoint['accuracy']
            
            logger.info(f"📂 Checkpoint loaded: epoch {self.current_epoch}, accuracy {self.best_accuracy:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return False
    
    def run_batch_training(self):
        """Run the complete batch training process"""
        logger.info("🚀 Starting Mac Batch Training System")
        logger.info("=" * 60)
        
        try:
            # Check system resources
            if not self.check_system_resources():
                logger.error("❌ Insufficient system resources")
                return False
            
            # Initialize training
            self.training_active = True
            self.training_start_time = time.time()
            
            # Create model and data
            self.model = self.create_optimized_model()
            # Note: You'll need to implement dataset creation based on your data
            
            # Setup optimizer and criterion
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['model']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            # Custom criterion for multi-task learning
            def multi_task_criterion(raga_logits, tonic_pred, confidence, raga_target, tonic_target):
                raga_criterion = nn.CrossEntropyLoss()
                tonic_criterion = nn.MSELoss()
                conf_criterion = nn.BCELoss()
                
                raga_loss = raga_criterion(raga_logits, raga_target)
                tonic_loss = tonic_criterion(tonic_pred.squeeze(), tonic_target)
                conf_loss = conf_criterion(confidence, torch.ones_like(confidence) * 0.9)
                
                return raga_loss + 0.1 * tonic_loss + 0.05 * conf_loss
            
            # Training loop
            for epoch in range(self.current_epoch, self.config['model']['max_epochs']):
                if not self.training_active:
                    break
                
                self.current_epoch = epoch
                
                # Train epoch
                train_loss, train_acc = self.train_epoch_batch(
                    self.model, train_loader, self.optimizer, multi_task_criterion, epoch
                )
                
                # Validate epoch
                val_loss, val_acc = self.validate_epoch_batch(
                    self.model, val_loader, multi_task_criterion
                )
                
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
                
                # Save best model
                if val_acc > self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.save_checkpoint(epoch, val_acc)
                
                # Regular checkpointing
                if epoch % self.config['training']['checkpoint_interval'] == 0:
                    self.save_checkpoint(epoch, val_acc)
                
                # Memory cleanup
                if epoch % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.info("🎉 Batch training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch training failed: {e}")
            return False
        finally:
            self.training_active = False

class MemoryMonitor:
    """Memory monitoring for batch training"""
    
    def __init__(self):
        self.max_memory_usage = 0.9  # 90% of available RAM
    
    def check_memory_usage(self):
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100
        
        if usage_percent > self.max_memory_usage:
            logger.warning(f"⚠️ High memory usage: {usage_percent:.1%}")
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

class BatchScheduler:
    """Batch job scheduler for Mac"""
    
    def __init__(self):
        self.jobs = []
    
    def schedule_training(self, start_time=None, duration_hours=12):
        """Schedule training job"""
        if start_time is None:
            start_time = datetime.now()
        
        job = {
            'start_time': start_time,
            'duration': timedelta(hours=duration_hours),
            'end_time': start_time + timedelta(hours=duration_hours),
            'status': 'scheduled'
        }
        
        self.jobs.append(job)
        logger.info(f"📅 Training scheduled: {start_time} for {duration_hours} hours")
    
    def run_scheduled_jobs(self):
        """Run scheduled training jobs"""
        current_time = datetime.now()
        
        for job in self.jobs:
            if job['status'] == 'scheduled' and current_time >= job['start_time']:
                if current_time <= job['end_time']:
                    logger.info(f"🚀 Starting scheduled training job")
                    job['status'] = 'running'
                    
                    # Run training
                    trainer = MacBatchTrainer()
                    success = trainer.run_batch_training()
                    
                    job['status'] = 'completed' if success else 'failed'
                else:
                    job['status'] = 'expired'

def main():
    """Main function for batch training"""
    print("🚀 Mac Batch Training System for RagaSense")
    print("=" * 50)
    
    # Create batch trainer
    trainer = MacBatchTrainer()
    
    # Run training
    success = trainer.run_batch_training()
    
    if success:
        print("\n✅ Batch training completed successfully!")
        print("📁 Check the 'ml/batch_training_checkpoints' directory for results")
    else:
        print("\n❌ Batch training failed. Check logs for details.")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    main()
