#!/bin/bash

# RagaSense Flutter Runner
# Run Flutter app on different platforms

echo "🎵 RagaSense Flutter App Runner"
echo ""

# Check if Flutter is available
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter is not installed. Please install Flutter SDK 3.16+"
    echo "   Visit: https://docs.flutter.dev/get-started/install"
    exit 1
fi

# Check if we're in the frontend directory
if [ ! -f "pubspec.yaml" ]; then
    echo "❌ Please run this script from the frontend directory"
    echo "   cd frontend && ./run_flutter.sh"
    exit 1
fi

# Install dependencies if needed
if [ ! -f "pubspec.lock" ]; then
    echo "📦 Installing Flutter dependencies..."
    flutter pub get
fi

# Function to cleanup
cleanup() {
    echo "🛑 Stopping Flutter app..."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "Choose platform to run:"
echo "1) Web (Chrome)"
echo "2) iOS Simulator"
echo "3) Android Emulator"
echo "4) All platforms (web + mobile)"
echo "5) Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🌐 Starting Flutter web app..."
        flutter run -d chrome --web-port 3000
        ;;
    2)
        echo "📱 Starting Flutter iOS app..."
        flutter run -d ios
        ;;
    3)
        echo "🤖 Starting Flutter Android app..."
        flutter run -d android
        ;;
    4)
        echo "🚀 Starting Flutter on all platforms..."
        echo "This will open multiple terminals"
        echo ""
        echo "🌐 Web: http://localhost:3000"
        echo "📱 iOS: iOS Simulator"
        echo "🤖 Android: Android Emulator"
        echo ""
        echo "Press Ctrl+C to stop all"
        
        # Start web in background
        flutter run -d chrome --web-port 3000 &
        WEB_PID=$!
        
        # Start iOS in background
        flutter run -d ios &
        IOS_PID=$!
        
        # Start Android in background
        flutter run -d android &
        ANDROID_PID=$!
        
        # Wait for user to stop
        wait
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac
