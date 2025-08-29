#!/bin/bash

# Raga Detection System Startup Script
# This script starts the complete working raga detection system

echo "🎵 Starting Raga Detection System..."
echo "=================================="

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import librosa, sklearn, joblib, soundfile" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required packages not installed. Please install:"
    echo "   pip install librosa scikit-learn joblib soundfile"
    exit 1
fi

# Test the ML system
echo "🧠 Testing ML system..."
python scripts/test_raga_detection.py
if [ $? -ne 0 ]; then
    echo "❌ Error: ML system test failed"
    exit 1
fi

# Start the backend server
echo "🚀 Starting backend server..."
echo "   Backend will be available at: http://localhost:8000"
echo "   API docs will be available at: http://localhost:8000/docs"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI server
python -m backend.main

