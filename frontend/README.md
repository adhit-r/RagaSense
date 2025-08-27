# RagaSense Frontend - Lynx Framework

This is the frontend for RagaSense, built with the [Lynx framework](http://lynxjs.org/index.html) to achieve a beautiful, Sazhaam-like user experience.

## Features

- **Sazhaam-like UX**: Modern, clean interface with smooth animations and transitions
- **Cross-platform**: Write once, render anywhere (Web, iOS, Android)
- **Real-time Raga Detection**: Upload audio files or record live
- **Beautiful Design**: Gradient backgrounds, rounded corners, and modern typography
- **Responsive**: Works perfectly on all device sizes

## Technology Stack

- **Lynx Framework**: Cross-platform development framework
- **ReactLynx**: Official React framework for Lynx
- **Rspeedy**: Rspack-based Lynx build tool
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework

## Getting Started

### Prerequisites

1. **Node.js 18 or later** (requires Node.js 18.19 when using TypeScript as configuration)

2. **Install Lynx Explorer** (for testing):
   - **iOS Simulator**: Download from [Lynx Quick Start Guide](https://lynxjs.org/guide/start/quick-start.html#ios-simulator-platform=macos-arm64,explorer-platform=ios-simulator)
   - **Android**: Scan QR code from GitHub Release or build from source
   - **HarmonyOS**: Download pre-built app or build from source

3. **Install Rspeedy** (Lynx build tool):
```bash
npm create rspeedy@latest
```

### Installation

1. Install dependencies:
```bash
bun install
```

2. Start the development server:
```bash
bun run dev
```

3. **Testing with Lynx Explorer**:
   - You will see a QR code in the terminal
   - Scan with your Lynx Explorer App, or
   - If using simulator, copy the bundle URL and paste it in the "Enter Card URL" input in Lynx Explorer App and hit "Go"

### Building

Build for web:
```bash
bun run build:web
```

Build for iOS:
```bash
bun run build:ios
```

Build for Android:
```bash
bun run build:android
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── RagaDetector.tsx    # Main raga detection component
│   ├── styles/
│   │   └── globals.css         # Global styles and animations
│   └── App.tsx                # Main app component
├── index.html                 # HTML template
├── rspeedy.config.ts         # Rspeedy configuration
├── tailwind.config.js        # Tailwind CSS configuration
└── package.json              # Dependencies and scripts
```

## Sazhaam-like UX Features

### Visual Design
- **Gradient Backgrounds**: Purple to blue gradients for a modern look
- **Rounded Corners**: 3xl border radius for cards and containers
- **Shadow Effects**: Subtle shadows for depth and hierarchy
- **Smooth Animations**: Fade-in, slide-up, and scale animations

### User Experience
- **Touch-friendly**: Optimized for mobile touch interactions
- **Live Recording**: Record audio directly in the app
- **Real-time Feedback**: Loading states and progress indicators
- **Error Handling**: Beautiful error states with helpful messages

### Responsive Design
- **Mobile-first**: Optimized for mobile devices
- **Cross-platform**: Works on iOS, Android, and Web
- **Adaptive Layout**: Responsive grid and flexible components

## API Integration

The frontend integrates with the FastAPI backend for raga detection:

- **Upload Endpoint**: `/api/ragas/detect` for audio file processing
- **Real-time Processing**: Live feedback during analysis
- **Error Handling**: Graceful error states and retry mechanisms

## Development Workflow

### Using Lynx Explorer

1. **Start Development Server**:
   ```bash
   bun run dev
   ```

2. **Connect to Lynx Explorer**:
   - Scan QR code with mobile app, or
   - Copy bundle URL to simulator

3. **Live Reload**: Make changes to `src/App.tsx` and see updates automatically

### Debugging

1. **Download Lynx DevTool**: Visit [Lynx DevTool](https://lynxjs.org/guide/debugging/devtool.html) to download the desktop application

2. **Connect Device**: Use USB cable to connect debugging device

3. **Start Debugging**: Follow the [Debugging Guide](https://lynxjs.org/guide/debugging/)

## Customization

### Colors
The app uses a custom color palette:
- Primary: Purple gradient (#8b5cf6)
- Success: Green gradient (#10b981)
- Error: Red gradient (#ef4444)

### Animations
Custom animations are defined in `globals.css`:
- `fadeIn`: Smooth fade-in effect
- `slideIn`: Slide-in from left
- `scaleIn`: Scale-in effect for cards

## Deployment

### Web Deployment
Build the web version and deploy to any static hosting service:
```bash
bun run build:web
```

### Mobile Deployment
Use Lynx's built-in mobile deployment:
```bash
bun run build:ios
bun run build:android
```

## Resources

- [Lynx Official Documentation](http://lynxjs.org/index.html)
- [Quick Start Guide](https://lynxjs.org/guide/start/quick-start.html)
- [ReactLynx Tutorial](https://lynxjs.org/guide/start/quick-start.html#reactlynx)
- [Debugging Guide](https://lynxjs.org/guide/debugging/)

## Contributing

1. Follow the existing code style and patterns
2. Add TypeScript types for all new components
3. Test on multiple devices and screen sizes
4. Ensure smooth animations and transitions

## License

This project is licensed under the MIT License.
