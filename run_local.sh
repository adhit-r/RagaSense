#!/bin/bash

# Start FastAPI backend (from backend/ directory)
echo "Starting FastAPI backend on http://localhost:8000 ..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Start React frontend (from frontend/ directory)
echo "Starting React frontend on http://localhost:3000 ..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait for both processes to finish
wait $BACKEND_PID $FRONTEND_PID
