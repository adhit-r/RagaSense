# ML Evaluation & Testing

## 🧪 **Testing Scripts for RagaSense ML Models**

This directory contains essential testing and validation scripts for our ML pipeline.

---

## 📁 **Test Files**

### **`test_raga_detection.py`** - Core Raga Detection Testing
- Tests the main raga classification functionality
- Validates feature extraction and model inference
- Essential for model validation

### **`test_huggingface_api.py`** - Hugging Face Model Comparison
- Compares our models with Hugging Face alternatives
- API connectivity testing
- Performance benchmarking

### **`test_our_model.py`** - Local Model Validation
- Tests our trained models locally
- Feature extraction validation
- Model loading and inference testing

### **`test_system.py`** - End-to-End System Testing
- Complete system validation
- Integration testing
- Performance and accuracy validation

---

## 🚀 **Running Tests**

### **Test Enhanced Model**
```bash
python ml/evaluation/test_raga_detection.py
```

### **Test Tradition Classifier**
```bash
python ml/evaluation/test_our_model.py
```

### **Test Hugging Face Integration**
```bash
python ml/evaluation/test_huggingface_api.py
```

---

## 📊 **Test Coverage**

- **Feature Extraction**: Audio processing and feature engineering
- **Model Loading**: Model persistence and loading
- **Inference**: Prediction accuracy and performance
- **Integration**: End-to-end system functionality
- **Comparison**: Model performance benchmarking

---

*Keep only essential test files for production use.*
