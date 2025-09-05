# RagaSense Flutter Frontend

A cross-platform Flutter application for AI-powered Indian classical music raga detection and generation.

## Features

- **Raga Detection**: Upload audio files or record live to detect ragas
- **Music Generation**: Generate music in different ragas and styles
- **Cross-Platform**: Works on Web, iOS, and Android
- **Professional UI**: Clean, modern design with Material 3
- **Real-time Status**: Monitor backend and ML model status

## Project Structure

```
lib/
├── main.dart                 # App entry point
├── models/                   # Data models
│   └── raga_model.dart      # Raga and prediction models
├── providers/                # State management (Riverpod)
│   └── app_providers.dart   # All app providers
├── screens/                  # App screens
│   ├── home_screen.dart     # Main raga detection screen
│   └── music_generation_screen.dart
├── services/                 # API and external services
│   └── api_service.dart     # Backend API communication
├── theme/                    # App theming
│   └── app_theme.dart       # Light/dark theme configuration
└── widgets/                  # Reusable UI components
    ├── app_drawer.dart      # Navigation drawer
    ├── audio_recorder_widget.dart
    ├── raga_result_widget.dart
    └── status_indicator_widget.dart
```

## Getting Started

### Prerequisites

- Flutter SDK 3.16+
- Dart 3.2+
- Backend server running on `http://localhost:8002`

### Installation

1. **Install Dependencies**
   ```bash
   flutter pub get
   ```

2. **Run the App**
   ```bash
   # For web
   flutter run -d chrome
   
   # For iOS
   flutter run -d ios
   
   # For Android
   flutter run -d android
   ```

### Development

The app uses:
- **Riverpod** for state management
- **Dio** for HTTP requests
- **Record** for audio recording
- **File Picker** for file selection
- **Google Fonts** for typography

## Key Components

### State Management
- `RagaDetectionProvider`: Manages raga detection state
- `AudioRecordingProvider`: Handles audio recording
- `MusicGenerationProvider`: Manages music generation
- `BackendHealthProvider`: Monitors backend status

### API Integration
- Health checks
- Raga detection
- Music generation
- Model status monitoring

### UI Features
- Professional B2C design
- Dark/light theme support
- Responsive layout
- Loading states and error handling
- Audio recording with visual feedback

## Backend Integration

The app communicates with the FastAPI backend at `http://localhost:8002`:

- `GET /health` - Backend health check
- `GET /api/models/status` - Model status
- `POST /api/detect-raga` - Raga detection
- `GET /api/ragas` - List supported ragas
- `POST /api/generate-music` - Music generation

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Web | ✅ Supported | PWA with audio recording |
| iOS | ✅ Supported | Native audio recording |
| Android | ✅ Supported | Native audio recording |
| macOS | 🔄 Planned | Desktop support |
| Windows | 🔄 Planned | Desktop support |
| Linux | 🔄 Planned | Desktop support |

## Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure backend is running on port 8002
   - Check network connectivity
   - Verify CORS settings

2. **Audio Recording Issues**
   - Grant microphone permissions
   - Check platform-specific audio settings
   - Ensure HTTPS for web recording

3. **File Upload Problems**
   - Verify file format (MP3, WAV, M4A, AAC)
   - Check file size limits
   - Ensure proper file permissions

### Debug Mode

Enable debug logging:
```dart
// In main.dart
void main() {
  runApp(
    ProviderScope(
      child: RagaSenseApp(),
    ),
  );
}
```

## Contributing

1. Follow Flutter best practices
2. Use proper state management with Riverpod
3. Maintain consistent theming
4. Add proper error handling
5. Test on multiple platforms

## License

MIT License - see [LICENSE](../LICENSE) for details.
