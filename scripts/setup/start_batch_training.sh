#!/bin/bash
# Mac Batch Training Script for RagaSense
# ======================================
# This script starts a batch training job optimized for Mac MPS

echo "🚀 Starting Mac Batch Training for RagaSense"
echo "============================================="

# Check if virtual environment exists
if [ ! -d "raga_env" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source raga_env/bin/activate

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Device: {torch.device(\"mps\" if torch.backends.mps.is_available() else \"cpu\")}')
"

# Create batch configuration
echo "📝 Creating batch configuration..."
python3 scripts/run_batch_job.py --create-config

# Start batch training
echo "🚀 Starting batch training..."
echo "This will run for approximately 3-5 hours for 1,402 ragas (603 Carnatic + 799 Hindustani)"
echo "The training will automatically save checkpoints every 5 epochs"
echo "You can stop and resume at any time using Ctrl+C"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Run the training
python3 scripts/run_batch_job.py --config ml/batch_config.json

echo "✅ Batch training completed!"
