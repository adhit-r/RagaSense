#!/bin/bash

# RagaSense Start Script
# Starts both backend and Flutter frontend (web/iOS/Android)

echo "🎵 Starting RagaSense..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

# Check if Flutter is available
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter is not installed. Please install Flutter SDK 3.16+"
    echo "   Visit: https://docs.flutter.dev/get-started/install"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo "🛑 Stopping RagaSense..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend server
echo "🚀 Starting backend server..."
cd backend
python3 main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check the logs above."
    exit 1
fi

echo "✅ Backend is running on http://localhost:8002"

# Start Flutter frontend
echo "🚀 Starting Flutter frontend..."
cd frontend

# Check if dependencies are installed
if [ ! -f "pubspec.lock" ]; then
    echo "📦 Installing Flutter dependencies..."
    flutter pub get
fi

# Start Flutter web by default
echo "🌐 Starting Flutter web app..."
flutter run -d chrome --web-port 3000 &
FRONTEND_PID=$!
cd ..

echo "✅ Flutter web is starting on http://localhost:3000"
echo ""
echo "🎵 RagaSense is running!"
echo "   🌐 Web App: http://localhost:3000"
echo "   🔧 Backend API: http://localhost:8002"
echo "   📚 API Docs: http://localhost:8002/docs"
echo ""
echo "📱 For mobile development:"
echo "   iOS: cd frontend && flutter run -d ios"
echo "   Android: cd frontend && flutter run -d android"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to stop
wait
