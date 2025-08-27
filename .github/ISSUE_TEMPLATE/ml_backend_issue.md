---
name: ML Backend Issue
about: Report issues with FastAPI backend, ML model, or raga detection system
title: '[ML-BACKEND] '
labels: ['ml', 'backend', 'fastapi', 'raga-detection', 'needs-triage']
assignees: ''
---

## Issue Description
Describe the ML backend or FastAPI problem you encountered.

## Issue Type
- [ ] Raga Detection Accuracy
- [ ] Model Performance
- [ ] API Endpoint Error
- [ ] Audio Processing Issue
- [ ] Feature Extraction Problem
- [ ] Model Loading Error
- [ ] Training Data Issue
- [ ] Performance/Speed Issue
- [ ] Other

## Error Details
If applicable, provide the error message or stack trace:

```python
# Example error
Traceback (most recent call last):
  File "ml/working_raga_detector.py", line 45, in detect_raga
    features = extract_features(audio_file)
ValueError: Audio file format not supported
```

## API Endpoint
- **Endpoint**: [e.g. POST /detect, GET /health]
- **Request Method**: [GET, POST, PUT, DELETE]
- **Request Body**: [if applicable]

## Audio File Information
- **Format**: [WAV, MP3, OGG, FLAC, M4A]
- **Duration**: [e.g. 30 seconds]
- **Sample Rate**: [e.g. 22050 Hz]
- **Channels**: [Mono, Stereo]
- **File Size**: [e.g. 2.5 MB]
- **Quality**: [High, Medium, Low]

## Detection Results
If this is a detection accuracy issue:

**Expected Result:**
```json
{
  "predicted_raga": "Expected Raga",
  "confidence": 0.95,
  "processing_time": 0.06
}
```

**Actual Result:**
```json
{
  "predicted_raga": "Actual Raga",
  "confidence": 0.45,
  "processing_time": 0.06
}
```

## Environment
- **Python Version**: [e.g. 3.9, 3.10, 3.11]
- **FastAPI Version**: [e.g. 0.104.0]
- **ML Libraries**: [TensorFlow, Librosa, Scikit-learn versions]
- **OS**: [Windows, macOS, Linux]
- **Backend URL**: [e.g. http://localhost:8000]

## Model Information
- **Model Type**: [RandomForest, Neural Network, etc.]
- **Model Version**: [e.g. v1.0.0]
- **Training Data**: [Synthetic, Real, Mixed]
- **Last Updated**: [Date]

## Steps to Reproduce
1. Start backend server: `python -m backend.main`
2. Send request to endpoint: `...`
3. Expected: `...`
4. Actual: `...`

## Performance Metrics
- **Processing Time**: [e.g. 0.06 seconds]
- **Memory Usage**: [if applicable]
- **CPU Usage**: [if applicable]
- **Response Time**: [API response time]

## Expected Behavior
A clear description of what should happen.

## Actual Behavior
A clear description of what actually happened.

## Logs
If applicable, add backend logs:

```bash
# Backend logs
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
ERROR:    Exception in ASGI application
```

## Additional Context
- Is this happening with all audio files or specific ones?
- Does this happen consistently or intermittently?
- Any recent changes to the ML model or backend?
- Is this a regression from a previous version?

## Checklist
- [ ] I have checked the backend logs
- [ ] I have verified the audio file format is supported
- [ ] I have tested with different audio files
- [ ] I have confirmed the backend server is running
- [ ] I have checked the ML model is loaded correctly
- [ ] This is not a duplicate of an existing issue
