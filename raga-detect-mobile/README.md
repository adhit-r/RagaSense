# Raga Detect Mobile

A high-performance mobile application for detecting Indian classical ragas in real-time audio. Built with Lynx.js for optimal performance and cross-platform compatibility.

## Features

- Real-time audio recording and processing
- High-accuracy raga detection
- Minimal latency audio pipeline
- Cross-platform (iOS & Android)
- Clean, modern UI with dark theme

## Tech Stack

- **Lynx.js** - Core framework
- **Web Audio API** - Audio processing
- **FFT** - Frequency analysis
- **Native Modules** - For performance-critical operations

## Getting Started

### Prerequisites

- Node.js >= 16.0.0
- Lynx CLI (`npm install -g @lynx-js/cli`)
- Xcode (for iOS development)
- Android Studio (for Android development)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/raga-detect-mobile.git
cd raga-detect-mobile

# Install dependencies
npm install

# Run on iOS
npm run ios

# Run on Android
npm run android
```

## Project Structure

```
src/
  ├── components/     # React components
  │   └── AppRoot.js  # Main application component
  ├── services/      # Business logic
  │   ├── audio.js   # Audio processing
  │   └── raga-detector.js  # Raga detection logic
  ├── assets/        # Static assets
  └── index.js       # Application entry point
```

## Audio Processing Pipeline

1. **Audio Capture**: Records audio from device microphone
2. **Pre-processing**: Applies noise reduction and normalization
3. **FFT Analysis**: Converts time-domain to frequency-domain
4. **Pitch Detection**: Identifies fundamental frequencies
5. **Note Recognition**: Maps frequencies to musical notes
6. **Raga Matching**: Compares note patterns with raga database

## Performance Optimizations

- Web Workers for background processing
- Efficient FFT implementation
- Minimal UI re-renders
- Optimized audio buffer handling

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Indian classical music theory
- Signal processing research
- Open source audio libraries
