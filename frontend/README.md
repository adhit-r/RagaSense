# RagaSense Mobile App

Cross-platform mobile application for AI-powered Carnatic raga detection and music generation, built with Flutter.

## 🚀 **Features**

### **Raga Detection**
- **Live Recording**: Real-time audio recording and analysis
- **File Upload**: Support for MP3, WAV, M4A, AAC files
- **Multiple Models**: Choose from 4 detection methods
- **Visual Feedback**: Processing indicators and results
- **Confidence Scores**: Detailed prediction confidence

### **Music Generation**
- **Raga Selection**: Choose from popular Carnatic ragas
- **Style Options**: Carnatic, Hindustani, Fusion
- **Duration Control**: 10-120 seconds of generated music
- **Audio Playback**: Built-in audio player
- **Download Support**: Save generated music locally

### **Cross-Platform**
- **Web**: Progressive Web App (PWA)
- **iOS**: Native iOS app
- **Android**: Native Android app
- **Responsive Design**: Works on all screen sizes

## 🛠️ **Technology Stack**

### **Framework**
- **Flutter 3.16+** - Cross-platform UI framework
- **Dart 3.2+** - Programming language
- **Material 3** - Modern design system

### **State Management**
- **Riverpod** - State management and dependency injection
- **Provider** - Legacy state management support

### **Audio & Media**
- **record** - Audio recording
- **audioplayers** - Audio playback
- **file_picker** - File selection
- **permission_handler** - Device permissions

### **Networking & API**
- **http** - HTTP requests
- **dio** - Advanced HTTP client
- **retrofit** - API client generation

### **UI & Design**
- **Google Fonts** - Typography
- **flutter_svg** - SVG support
- **lottie** - Animations

### **Storage & Database**
- **shared_preferences** - Local storage
- **sqflite** - SQLite database
- **path_provider** - File system access

## 📱 **Platform Support**

| Platform | Status | Notes |
|----------|--------|-------|
| Web | ✅ Supported | PWA with audio recording |
| iOS | ✅ Supported | Native audio recording |
| Android | ✅ Supported | Native audio recording |
| macOS | 🔄 Planned | Desktop support |
| Windows | 🔄 Planned | Desktop support |
| Linux | 🔄 Planned | Desktop support |

## 🚀 **Quick Start**

### **Prerequisites**
- Flutter SDK 3.16+
- Dart SDK 3.2+
- Android Studio / Xcode (for mobile)
- Chrome (for web)

### **Installation**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ragasense_mobile
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Run the app**
   ```bash
   # For web
   flutter run -d chrome
   
   # For iOS simulator
   flutter run -d ios
   
   # For Android emulator
   flutter run -d android
   ```

## 🏗️ **Project Structure**

```
lib/
├── constants/
│   ├── app_colors.dart      # Color definitions
│   └── app_theme.dart       # Theme configuration
├── models/
│   ├── raga.dart            # Raga data model
│   └── raga_detection_result.dart # Detection results
├── screens/
│   └── home_screen.dart     # Main app screen
├── services/
│   └── raga_detection_service.dart # API integration
├── widgets/
│   ├── raga_detection_widget.dart  # Detection UI
│   └── music_generation_widget.dart # Generation UI
└── main.dart                # App entry point
```

## 🎨 **Design System**

### **Colors**
- **Primary**: Blue gradient (professional)
- **Success**: Green (positive feedback)
- **Warning**: Orange (cautions)
- **Error**: Red (errors)
- **Gray**: Neutral grays (text, borders)

### **Typography**
- **Font**: Inter (Google Fonts)
- **Weights**: Regular, Medium, SemiBold, Bold
- **Sizes**: 10px to 32px scale

### **Components**
- **Cards**: Elevated containers with borders
- **Buttons**: Primary, secondary, and text variants
- **Inputs**: Filled text fields with validation
- **Progress**: Loading indicators and sliders

## 🔧 **Configuration**

### **Backend Integration**
The app connects to the FastAPI backend for raga detection:

```dart
// Update this URL in services/raga_detection_service.dart
static const String _baseUrl = 'http://localhost:8002';
```

### **Environment Variables**
Create a `.env` file for configuration:

```env
BACKEND_URL=http://localhost:8002
API_KEY=your_api_key_here
```

## 📱 **Mobile Permissions**

### **iOS (Info.plist)**
```xml
<key>NSMicrophoneUsageDescription</key>
<string>This app needs microphone access to record audio for raga detection.</string>
<key>NSDocumentsFolderUsageDescription</key>
<string>This app needs access to documents to save generated music.</string>
```

### **Android (AndroidManifest.xml)**
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

## 🚀 **Deployment**

### **Web Deployment**
```bash
flutter build web
# Deploy the build/web folder to your hosting service
```

### **iOS App Store**
```bash
flutter build ios --release
# Open ios/Runner.xcworkspace in Xcode
# Archive and upload to App Store Connect
```

### **Google Play Store**
```bash
flutter build appbundle --release
# Upload the generated .aab file to Google Play Console
```

## 🧪 **Testing**

### **Unit Tests**
```bash
flutter test
```

### **Integration Tests**
```bash
flutter test integration_test/
```

### **Widget Tests**
```bash
flutter test test/widget_test.dart
```

## 🔄 **Development Workflow**

1. **Feature Development**
   - Create feature branch
   - Implement UI components
   - Add business logic
   - Write tests
   - Submit PR

2. **State Management**
   - Use Riverpod for global state
   - Local state with StatefulWidget
   - Provider for simple state

3. **API Integration**
   - Create service classes
   - Handle errors gracefully
   - Add loading states
   - Cache responses

## 📊 **Performance**

- **Bundle Size**: Optimized for mobile
- **Loading Time**: Fast initial load
- **Audio Processing**: Efficient audio handling
- **Memory Usage**: Optimized for mobile devices

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Follow Flutter best practices
4. Add tests for new features
5. Submit a pull request

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

## 🎵 **About Carnatic Music**

Carnatic music is a classical music tradition from South India, characterized by its complex melodic structures called "ragas" and rhythmic patterns called "talas". This application helps musicians and enthusiasts identify and understand these intricate musical patterns using AI.

---

**Built with ❤️ for the Carnatic music community**
