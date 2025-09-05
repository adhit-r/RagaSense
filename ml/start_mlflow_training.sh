#!/bin/bash
# MLflow Training Starter Script for RagaSense
# ============================================

echo "🚀 Starting MLflow-Enhanced RagaSense Training"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "raga_env" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv raga_env
    source raga_env/bin/activate
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment found"
    source raga_env/bin/activate
fi

# Install MLflow requirements
echo "📦 Installing MLflow requirements..."
pip install -r ml/requirements_v1.2.txt

# Check if dataset exists
if [ ! -d "carnatic-hindustani-dataset" ]; then
    echo "❌ Dataset not found. Please ensure carnatic-hindustani-dataset is available."
    echo "   You can download it or create a symlink to your dataset location."
    exit 1
else
    echo "✅ Dataset found"
fi

# Create MLflow directory
mkdir -p mlruns
echo "✅ MLflow directory created"

# Start MLflow training
echo "🤖 Starting MLflow-enhanced training..."
echo "   This will:"
echo "   - Create MLflow experiment"
echo "   - Log all hyperparameters and metrics"
echo "   - Save model artifacts"
echo "   - Generate visualizations"
echo ""

python3 ml/training/optimized_working_mve_v1.2.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "📊 To view results:"
    echo "   mlflow ui --backend-store-uri ./mlruns --port 5001"
    echo "   Then open: http://localhost:5001"
    echo ""
    echo "🔬 MLflow commands:"
    echo "   mlflow experiments list"
    echo "   mlflow runs list --experiment-id 0"
    echo "   mlflow models list"
    echo ""
    echo "🚀 To serve the model:"
    echo "   mlflow models serve -m ./mlruns/0/[run_id]/artifacts/optimized_working_mve_v1.2"
else
    echo ""
    echo "❌ Training failed. Check the logs for details."
    exit 1
fi
