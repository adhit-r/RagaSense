#!/bin/bash

# Start the development environment for RagaSense

echo "Starting RagaSense development environment..."
echo "========================================"

# Create required directories
echo "Creating required directories..."
mkdir -p uploads
mkdir -p ml/models

# Set environment variables
export DATABASE_URL=postgresql://raga_user:raga_pass@localhost:5432/ragasense_db
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if database is running
if ! pg_isready -h localhost -p 5432 -q; then
    echo "PostgreSQL is not running. Starting with Docker..."
    docker run --name ragasense-db -e POSTGRES_USER=raga_user \
        -e POSTGRES_PASSWORD=raga_pass \
        -e POSTGRES_DB=ragasense_db \
        -p 5432:5432 \
        -d postgres:13
    
    # Wait for database to be ready
    echo "Waiting for database to be ready..."
    sleep 5
    
    # Initialize database
    echo "Initializing database..."
    python3 init_db.py
    
    # Load sample data
    echo "Loading sample data..."
    python3 scripts/init_sample_data.py
fi

# Train a sample model if one doesn't exist
if [ ! -f "ml/models/raga_model.h5" ]; then
    echo "Training a sample model..."
    python3 scripts/train_sample_model.py
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
FASTAPI_PID=$!

# Start the frontend development server if the directory exists
if [ -d "frontend" ]; then
    echo "Starting frontend development server..."
    cd frontend
    npm install
    npm start &
    FRONTEND_PID=$!
    cd ..
fi

# Function to clean up on exit
cleanup() {
    echo "Shutting down..."
    kill $FASTAPI_PID 2>/dev/null
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set up trap to catch termination signals
trap cleanup INT TERM

# Keep the script running
wait $FASTAPI_PID

# If we get here, the FastAPI server has stopped
cleanup
