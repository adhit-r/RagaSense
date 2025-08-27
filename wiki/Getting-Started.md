# Getting Started with RagaSense

This guide will help you get up and running with RagaSense quickly.

## Prerequisites

Before you begin, make sure you have the following installed:

- **Node.js 18** or later
- **Python 3.9** or later
- **Git**
- **Lynx Explorer** (for frontend testing)

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/raga_detector.git
   cd raga_detector
   ```

2. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Local Development

#### Backend Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**:
   ```bash
   python init_db.py
   ```

4. **Start the backend server**:
   ```bash
   python -m backend.main
   ```

#### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   bun install
   ```

3. **Start the development server**:
   ```bash
   bun run dev
   ```

4. **Test with Lynx Explorer**:
   - Install Lynx Explorer from the [official guide](http://lynxjs.org/guide/start/quick-start.html)
   - Scan the QR code or copy the bundle URL
   - Test the app on your device

## First Steps

### 1. Test Raga Detection

1. **Upload an audio file**:
   - Go to the main interface
   - Click "Upload Audio File" or drag and drop
   - Supported formats: WAV, MP3, OGG, FLAC, M4A
   - Maximum duration: 30 seconds

2. **Record live audio**:
   - Click "Record Live" button
   - Allow microphone access
   - Play or sing a raga
   - Stop recording to get results

### 2. View Results

The system will display:
- **Detected Raga**: The most likely raga
- **Confidence Score**: How confident the system is
- **All Predictions**: List of possible ragas with probabilities
- **Supported Ragas**: Currently supported ragas

### 3. API Testing

Test the backend API directly:

```bash
# Health check
curl http://localhost:8000/api/ragas/health

# Get supported ragas
curl http://localhost:8000/api/ragas/supported-ragas

# Upload audio file
curl -X POST -F "audio=@your_audio_file.wav" http://localhost:8000/api/ragas/detect
```

## Supported Ragas

Currently, RagaSense supports these ragas:
- **Yaman** (Evening raga)
- **Bhairav** (Morning raga)
- **Kafi** (Late evening raga)

More ragas will be added in future releases.

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Kill process using port 8000
   lsof -ti:8000 | xargs kill -9
   ```

2. **Python dependencies not found**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Frontend build fails**:
   ```bash
   cd frontend
   rm -rf node_modules
   bun install
   ```

4. **Database connection issues**:
   ```bash
   # Reset database
   python init_db.py
   ```

### Getting Help

- **Check the logs**: Look for error messages in the terminal
- **Verify prerequisites**: Ensure all required software is installed
- **Search issues**: Check [GitHub Issues](https://github.com/your-username/raga_detector/issues)
- **Ask the community**: Use [GitHub Discussions](https://github.com/your-username/raga_detector/discussions)

## Next Steps

Now that you have RagaSense running, you can:

- **Explore the API**: Check out the [API Documentation](API-Documentation)
- **Contribute**: Read the [Contributing Guide](Contributing-Guide)
- **Learn more**: Visit the [Architecture Overview](Architecture-Overview)
- **Report issues**: Create an issue on GitHub

## Support

If you need help:

- **Documentation**: Check the [docs](docs/) folder
- **Wiki**: Browse this wiki for detailed guides
- **Issues**: Report bugs on GitHub
- **Discussions**: Ask questions in GitHub Discussions

Welcome to RagaSense! 🎵
