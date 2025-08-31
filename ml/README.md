# ML/MLOps Pipeline

## Structure

```
ml/
├── data/                    # Data pipeline
│   ├── collection/         # Data collection scripts
│   └── preprocessing/      # Data preprocessing
├── models/                 # Model training & deployment
│   ├── training/          # Training scripts
│   └── deployment/        # Model deployment
├── inference/              # Model inference services
│   └── services/          # Inference APIs
├── evaluation/             # Model evaluation
└── mlops/                  # MLOps tools
```

## Quick Start

1. **Install dependencies**: `pip install -r requirements_deep_learning.txt`
2. **Train model**: `python models/training/train_raga_classifier.py`
3. **Deploy model**: `python models/deployment/deploy_model.py`
4. **Run inference**: `python inference/services/raga_detection_service.py`

## Data Sources

- **Convex Database**: Raga metadata and user data
- **External Datasets**: CompMusic, Saraga, Sanidha
- **Hugging Face**: Pre-trained models and datasets

## Model Types

1. **Local Custom Model**: Trained on our data
2. **Hugging Face Cloud**: API-based inference
3. **Local Hugging Face**: Downloaded model for offline use
4. **Ensemble**: Combination of all models

## Current Files

### Data Pipeline
- `data/preprocessing/` - Feature extraction scripts
- `data/collection/` - Data collection utilities

### Model Training
- `models/training/` - Training scripts and models
- `models/deployment/` - Model deployment utilities

### Inference Services
- `inference/services/` - Model inference APIs

### Evaluation
- `evaluation/` - Model evaluation and testing

### MLOps
- `mlops/` - Monitoring and versioning tools
