# Raga Detection ML Module

This module implements a machine learning-based raga detection system using neural networks and mel spectrogram features, inspired by the approach described in the Carnatic Raga Recognition article.

## Features

- **Mel Spectrogram Feature Extraction**: Uses librosa to extract mel spectrogram features from audio files
- **Neural Network Classification**: Deep learning model with dropout layers for robust classification
- **Database Integration**: Seamlessly integrates with the FastAPI backend database
- **Real-time Prediction**: Supports real-time raga prediction from uploaded audio files
- **Model Training**: Complete training pipeline for custom datasets

## Architecture

### Feature Extraction
- **Mel Spectrogram**: 128 mel frequency bins
- **Sample Rate**: 22050 Hz
- **Hop Length**: 512 samples
- **Feature Dimension**: 128 features per audio clip

### Neural Network Model
```
Input (128 features)
    ↓
Dense(256, relu) + Dropout(0.3)
    ↓
Dense(128, relu) + Dropout(0.3)
    ↓
Dense(64, relu) + Dropout(0.2)
    ↓
Dense(num_classes, softmax)
```

## Usage

### 1. Basic Prediction

```python
from app.ml.raga_classifier import RagaClassifier

# Create classifier
classifier = RagaClassifier()

# Predict raga from audio file
result = classifier.predict_from_upload("path/to/audio.wav")
print(f"Predicted raga: {result['predicted_raga']}")
print(f"Confidence: {result['confidence']}")
```

### 2. Training a Model

```bash
# Install ML dependencies
pip install -r requirements_ml.txt

# Train model with your dataset
python app/ml/train_model.py \
    --data-dir /path/to/audio/dataset \
    --model-save-path models/raga_classifier \
    --epochs 100
```

### 3. Using Trained Model

```python
from app.ml.raga_classifier import RagaClassifier

# Load trained model
classifier = RagaClassifier("models/raga_classifier")
result = classifier.predict_from_upload("test_audio.wav")
```

## Dataset Structure

Organize your audio files in the following structure:

```
dataset/
├── Yaman/
│   ├── yaman_sample1.wav
│   ├── yaman_sample2.wav
│   └── ...
├── Bhairav/
│   ├── bhairav_sample1.wav
│   ├── bhairav_sample2.wav
│   └── ...
└── ...
```

## API Endpoints

### Predict Raga
```
POST /api/predict
Content-Type: multipart/form-data

file: audio_file.wav
```

Response:
```json
{
  "predicted_raga": "Yaman",
  "confidence": 0.85,
  "top_predictions": [
    {"raga": "Yaman", "confidence": 0.85},
    {"raga": "Bhairav", "confidence": 0.10},
    {"raga": "Khamaj", "confidence": 0.03}
  ],
  "raga_info": {
    "name": "Yaman",
    "aroha": ["Sa", "Re", "Ga", "Ma", "Pa", "Dha", "Ni", "Sa"],
    "avaroha": ["Sa", "Ni", "Dha", "Pa", "Ma", "Ga", "Re", "Sa"],
    "vadi": "Ga",
    "samvadi": "Ni",
    "time": ["Evening"],
    "mood": ["Peaceful", "Romantic"]
  }
}
```

### Model Information
```
GET /api/model-info
```

## Testing

### Run Basic Test
```bash
python app/ml/test_model.py
```

### Test with Real Audio
```bash
python app/ml/test_model.py path/to/audio.wav
```

## Model Training Process

1. **Data Preparation**: Organize audio files by raga in directories
2. **Feature Extraction**: Extract mel spectrogram features from all audio files
3. **Data Splitting**: Split into training and validation sets (80/20)
4. **Model Training**: Train neural network with early stopping
5. **Model Saving**: Save model and preprocessing objects

## Performance Optimization

- **Early Stopping**: Prevents overfitting by monitoring validation accuracy
- **Dropout Layers**: Regularization to improve generalization
- **Feature Scaling**: StandardScaler for consistent feature ranges
- **Batch Processing**: Efficient training with batch size of 32

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)
- OGG (.ogg)

## Dependencies

Install ML-specific dependencies:
```bash
pip install -r requirements_ml.txt
```

Key dependencies:
- `tensorflow`: Neural network framework
- `librosa`: Audio processing and feature extraction
- `scikit-learn`: Data preprocessing and model evaluation
- `numpy`: Numerical computations
- `soundfile`: Audio file I/O

## Future Improvements

1. **Advanced Features**: Add MFCC, chroma, and tonnetz features
2. **Data Augmentation**: Pitch shifting, time stretching, noise addition
3. **Ensemble Methods**: Combine multiple models for better accuracy
4. **Real-time Processing**: Stream audio for live raga detection
5. **Transfer Learning**: Use pre-trained models for better performance

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all ML dependencies are installed
2. **Memory Issues**: Reduce batch size or use smaller audio clips
3. **Audio Format Issues**: Convert audio to supported formats
4. **Model Loading Errors**: Check model file paths and permissions

### Performance Tips

1. **GPU Acceleration**: Use TensorFlow with GPU support for faster training
2. **Audio Preprocessing**: Normalize audio files before training
3. **Feature Caching**: Cache extracted features to speed up training
4. **Model Optimization**: Use TensorFlow Lite for deployment

## References

- [Carnatic Raga Recognition Article](https://medium.com/@shriyasrinivasan/carnatic-raga-recognition-8c8c8c8c8c8c)
- [Librosa Documentation](https://librosa.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/) 