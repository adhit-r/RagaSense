# Flutter Cross-Platform Development Guide

RagaSense now uses **Flutter/Dart** for a **single codebase** that runs on **web, iOS, and Android**!

## 🎯 **Why Flutter?**

### **Single Codebase, Multiple Platforms**
- **Web**: Progressive Web App (PWA)
- **iOS**: Native iOS app
- **Android**: Native Android app
- **Desktop**: macOS, Windows, Linux (planned)

### **Benefits**
- ✅ **One codebase** for all platforms
- ✅ **Native performance** on mobile
- ✅ **Rich ecosystem** of packages
- ✅ **Hot reload** for fast development
- ✅ **Professional UI** with Material 3
- ✅ **Audio support** for recording and playback

## 🚀 **Quick Start**

### **1. Install Flutter**
```bash
# macOS (using Homebrew)
brew install --cask flutter

# Or download from https://docs.flutter.dev/get-started/install
```

### **2. Verify Installation**
```bash
flutter doctor
```

### **3. Run the App**

#### **Option A: Use the start script**
```bash
./start.sh  # Starts backend + Flutter web
```

#### **Option B: Manual start**
```bash
# Start backend
cd backend && python3 main.py

# Start Flutter (in new terminal)
cd frontend && flutter run -d chrome  # Web
cd frontend && flutter run -d ios      # iOS
cd frontend && flutter run -d android  # Android
```

#### **Option C: Use the Flutter runner**
```bash
cd frontend && ./run_flutter.sh
```

## 📱 **Platform-Specific Setup**

### **Web Development**
```bash
cd frontend
flutter run -d chrome --web-port 3000
```
- **URL**: http://localhost:3000
- **Features**: Full audio recording, file upload, responsive design
- **Browser Support**: Chrome, Firefox, Safari, Edge

### **iOS Development**
```bash
# Prerequisites
# - Xcode installed
# - iOS Simulator or physical device

cd frontend
flutter run -d ios
```
- **Features**: Native audio recording, camera access, push notifications
- **Requirements**: macOS + Xcode

### **Android Development**
```bash
# Prerequisites
# - Android Studio installed
# - Android emulator or physical device

cd frontend
flutter run -d android
```
- **Features**: Native audio recording, file system access, background processing
- **Requirements**: Android Studio + Android SDK

## 🛠️ **Development Workflow**

### **1. Code Structure**
```
frontend/lib/
├── constants/           # Colors, themes, constants
├── models/             # Data models (Raga, DetectionResult)
├── screens/            # Main app screens
├── services/           # API services, backend integration
├── widgets/            # Reusable UI components
└── main.dart          # App entry point
```

### **2. State Management**
```dart
// Using Riverpod for state management
class RagaDetectionProvider extends StateNotifier<RagaDetectionState> {
  // State management logic
}
```

### **3. API Integration**
```dart
// Backend API calls
class RagaDetectionService {
  static Future<RagaDetectionResult> detectRaga(String audioPath) async {
    // HTTP requests to FastAPI backend
  }
}
```

### **4. Audio Handling**
```dart
// Audio recording and playback
final AudioRecorder _audioRecorder = AudioRecorder();
await _audioRecorder.start(RecordConfig());
```

## 🎨 **UI/UX Features**

### **Design System**
- **Material 3**: Modern design language
- **Inter Font**: Professional typography
- **Color Palette**: Consistent branding
- **Responsive Layout**: Works on all screen sizes

### **Components**
- **Cards**: Elevated containers with borders
- **Buttons**: Primary, secondary, and text variants
- **Inputs**: Filled text fields with validation
- **Progress**: Loading indicators and sliders
- **Audio Player**: Built-in audio controls

### **Themes**
- **Light Theme**: Clean, professional appearance
- **Dark Theme**: Modern dark mode support
- **System Theme**: Automatic theme switching

## 🔧 **Configuration**

### **Environment Setup**
```bash
# Check Flutter installation
flutter doctor

# Get dependencies
cd frontend && flutter pub get

# Check available devices
flutter devices
```

### **Backend Integration**
```dart
// Update API URL in services/raga_detection_service.dart
static const String _baseUrl = 'http://localhost:8002';
```

### **Platform Permissions**

#### **iOS (Info.plist)**
```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs microphone access to record audio for raga detection.</string>
<key>NSDocumentsFolderUsageDescription</key>
<string>This app needs access to documents to save generated music.</string>
```

#### **Android (AndroidManifest.xml)**
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

## 🧪 **Testing**

### **Unit Tests**
```bash
cd frontend
flutter test
```

### **Widget Tests**
```bash
flutter test test/widget_test.dart
```

### **Integration Tests**
```bash
flutter test integration_test/
```

### **Manual Testing**
```bash
# Test on different platforms
flutter run -d chrome    # Web
flutter run -d ios       # iOS
flutter run -d android   # Android
```

## 📦 **Building for Production**

### **Web Build**
```bash
cd frontend
flutter build web
# Deploy build/web folder to hosting service
```

### **iOS Build**
```bash
cd frontend
flutter build ios --release
# Open ios/Runner.xcworkspace in Xcode
# Archive and upload to App Store Connect
```

### **Android Build**
```bash
cd frontend
flutter build appbundle --release
# Upload .aab file to Google Play Console
```

## 🚀 **Deployment**

### **Web Deployment**
```bash
# Build for web
cd frontend && flutter build web

# Deploy to hosting service (e.g., Netlify, Vercel, Firebase)
# Upload build/web folder
```

### **Mobile App Stores**
```bash
# iOS App Store
flutter build ios --release
# Use Xcode to archive and upload

# Google Play Store
flutter build appbundle --release
# Upload to Google Play Console
```

## 🔍 **Debugging**

### **Common Issues**

#### **1. Flutter not found**
```bash
# Install Flutter
brew install --cask flutter  # macOS
# Or download from flutter.dev
```

#### **2. Dependencies not installed**
```bash
cd frontend
flutter pub get
```

#### **3. Device not detected**
```bash
flutter devices
flutter doctor
```

#### **4. Audio permissions**
- **iOS**: Check Info.plist permissions
- **Android**: Check AndroidManifest.xml permissions
- **Web**: Check browser microphone access

### **Debug Commands**
```bash
# Check Flutter installation
flutter doctor

# List available devices
flutter devices

# Clean and rebuild
flutter clean
flutter pub get
flutter run

# Check for issues
flutter analyze
```

## 📊 **Performance**

### **Web Performance**
- **Bundle Size**: ~2MB (compressed)
- **Loading Time**: <3 seconds
- **Audio Processing**: Real-time
- **Memory Usage**: <100MB

### **Mobile Performance**
- **App Size**: ~50MB (iOS), ~30MB (Android)
- **Launch Time**: <2 seconds
- **Audio Recording**: Native performance
- **Memory Usage**: <200MB

## 🔄 **Development Tips**

### **1. Hot Reload**
- Press `r` in terminal for hot reload
- Press `R` for hot restart
- Press `q` to quit

### **2. Platform-Specific Code**
```dart
if (kIsWeb) {
  // Web-specific code
} else if (Platform.isIOS) {
  // iOS-specific code
} else if (Platform.isAndroid) {
  // Android-specific code
}
```

### **3. Responsive Design**
```dart
LayoutBuilder(
  builder: (context, constraints) {
    if (constraints.maxWidth > 600) {
      return DesktopLayout();
    } else {
      return MobileLayout();
    }
  },
)
```

### **4. Audio Handling**
```dart
// Request permissions
await Permission.microphone.request();

// Record audio
await _audioRecorder.start(RecordConfig());

// Stop recording
String? path = await _audioRecorder.stop();
```

## 🎯 **Next Steps**

1. **Test on all platforms**: Web, iOS, Android
2. **Add features**: User authentication, history, settings
3. **Optimize performance**: Bundle size, loading times
4. **Deploy**: Web hosting, app stores
5. **Monitor**: Analytics, crash reporting

---

**Flutter gives us the power to build once, run everywhere! 🚀**
