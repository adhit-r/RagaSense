# RagaSense ML Workspace

## 🎵 **Machine Learning Models & Training**

This directory contains the complete ML pipeline for RagaSense, including enhanced training, tradition classification, and production inference.

---

## 📁 **Directory Structure**

### **`enhanced_models/`** - Core ML Models
- **`enhanced_raga_model.pkl`** - Main raga classification model (106 features)
- **`enhanced_scaler.pkl`** - Feature scaling for model input
- **`enhanced_label_encoder.pkl`** - Label encoding for raga classes
- **`enhanced_feature_names.json`** - Names of all 106 extracted features
- **`training_results.json`** - Training performance metrics
- **`TRAINING_REPORT.md`** - Comprehensive training analysis

### **`tradition_classification/`** - Tradition Classifiers
- **`tradition_classifier.pkl`** - Carnatic vs Hindustani classifier
- **`tradition_scaler.pkl`** - Feature scaling for tradition features
- **`tradition_label_encoder.pkl`** - Label encoding for traditions
- **`tradition_feature_names.json`** - 21 cultural feature names
- **`tradition_training_results.json`** - Tradition classification metrics
- **`TRADITION_CLASSIFICATION_REPORT.md`** - Cultural validation report

### **`evaluation/`** - Testing & Validation Scripts
- **`test_raga_detection.py`** - Core raga detection testing
- **`test_huggingface_api.py`** - Hugging Face model comparison
- **`test_our_model.py`** - Local model validation
- **`test_system.py`** - End-to-end system testing

### **`inference/`** - Production Inference Services
- **`services/`** - Production ML inference endpoints

---

## 🚀 **Model Performance**

### **Enhanced Training Pipeline**
- **Features**: 106 advanced audio features (8x improvement)
- **Accuracy**: 100% on 37 real Carnatic audio samples
- **Algorithms**: RandomForest, Neural Network
- **Cross-validation**: 5-fold CV with minimal variance

### **Tradition Classification**
- **Features**: 21 cultural features for tradition distinction
- **Accuracy**: 100% on synthetic dataset
- **Cultural Features**: Gamaka, Meend, Shruti, Performance patterns
- **Validation**: Expert-designed cultural feature engineering

---

## 🎯 **Usage Examples**

### **Load Enhanced Model**
```python
import joblib
import pickle

# Load the enhanced raga classification model
with open('ml/enhanced_models/enhanced_raga_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessing objects
scaler = joblib.load('ml/enhanced_models/enhanced_scaler.pkl')
label_encoder = joblib.load('ml/enhanced_models/enhanced_label_encoder.pkl')
```

### **Load Tradition Classifier**
```python
# Load tradition classification model
with open('ml/tradition_classification/tradition_classifier.pkl', 'rb') as f:
    tradition_model = pickle.load(f)

# Load tradition preprocessing
tradition_scaler = joblib.load('ml/tradition_classification/tradition_scaler.pkl')
```

---

## 🔧 **Training Scripts**

### **Enhanced Training Pipeline**
```bash
# Run enhanced training with 106 features
python scripts/enhanced_training_pipeline.py
```

### **Tradition Classification**
```bash
# Train tradition classifier (Carnatic vs Hindustani)
python scripts/tradition_classification_system.py
```

---

## 📊 **Feature Engineering**

### **Enhanced Features (106 total)**
- **MFCC**: 26 features (13 mean + 13 std)
- **Spectral**: 6 features (centroid, rolloff, bandwidth)
- **Chroma**: 24 features (12 mean + 12 std)
- **Tonnetz**: 6 features
- **Rhythm**: 1 feature (tempo)
- **Energy**: 4 features (ZCR, RMS)
- **Contrast**: 12 features (6 mean + 6 std)
- **Mel**: 20 features (10 mean + 10 std)
- **Harmonic**: 3 features (harmonic, percussive, total energy)
- **Additional**: 4 features (flatness, rolloff)

### **Cultural Features (21 total)**
- **Ornamentation**: Gamaka vs Meend patterns
- **Microtonal**: Shruti system complexity
- **Rhythmic**: Tala vs Taal differences
- **Melodic**: Raga progression patterns
- **Timbre**: Instrumental characteristics
- **Performance**: Alap vs Alapana structure

---

## 🌟 **Next Steps**

### **Phase 2: Advanced Features**
1. **Real Dataset Integration** - Test on actual Carnatic/Hindustani recordings
2. **Parent Scale Classification** - Melakarta vs Thaat classification
3. **Cultural Expert Validation** - Expert review and feedback

### **Phase 3: Production Deployment**
1. **Production ML Pipeline** - Deploy models to backend
2. **API Integration** - Connect with FastAPI endpoints
3. **Performance Optimization** - Real-time inference optimization

---

## 📝 **Requirements**

### **Core Dependencies**
```bash
pip install -r ml/requirements_enhanced_training.txt
```

### **Key Libraries**
- **scikit-learn**: ML algorithms and preprocessing
- **librosa**: Audio feature extraction
- **numpy/pandas**: Data manipulation
- **joblib**: Model persistence

---

## 🎉 **Current Status**

**Phase 1 Complete**: Foundation established with enhanced training pipeline and tradition classification system.

**Ready for Phase 2**: Parent scale classification and real dataset integration.

---

*Last Updated: September 3, 2024*
