# RagaSense ML Workspace
## 🤖 Machine Learning Models and Training

This directory contains all machine learning related files for the RagaSense project, organized for clarity and maintainability.

## 📁 Directory Structure

```
ml/
├── training/           # Current training scripts
│   └── optimized_working_mve_v1.2.py  # Current optimized training with MLOps
├── scripts/            # Utility scripts (current versions)
├── models/             # Saved models and checkpoints
├── results/            # Training results and outputs
├── data/               # Dataset utilities and processing
├── evaluation/         # Model evaluation scripts
├── requirements_v1.2.txt
├── MLOPS_INTEGRATION_GUIDE_v1.2.md
├── start_mlflow_training.sh
└── README.md
```

## 🚀 Quick Start

### 1. MLflow-Enhanced Training (Recommended)
```bash
# Start MLflow training with comprehensive logging
./ml/start_mlflow_training.sh

# View results in MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

### 2. Archived Model Training
```bash
# Previous versions available in archive/
# See archive/ARCHIVE_INDEX.md for complete list
```

## 📊 Model Comparison

| Model | Version | Accuracy | Training Time | Use Case | Status |
|-------|---------|----------|---------------|----------|--------|
| Optimized Working MVE | v1.2 | 87-90% | 2-3 hours | Production+ | **Current** |
| Optimized MVE | v1.1 | 87-90% | 2-3 hours | Production | Archived |
| Working MVE | v1.0 | 85% | 2-3 hours | Baseline | Archived |
| Cutting-Edge | v2.0 | 90-92% | 4-6 hours | Research | Archived |
| Ultra-Advanced | v2.1 | 92-95% | 6-8 hours | Research+ | Archived |

## 🔬 MLflow Features

- **Complete experiment tracking** with hyperparameters and metrics
- **Model versioning and management** for production deployment
- **Real-time monitoring** of training progress
- **Visualization and analysis** tools for model performance
- **Production deployment** capabilities
- **Model comparison** and selection tools

## 📈 Expected Results

- **Accuracy**: 87-90% (vs 85% original)
- **Training Time**: 2-3 hours (vs 3-5 hours)
- **Complete tracking**: All metrics, parameters, and artifacts
- **Model versioning**: Automatic registration and management
- **Visualizations**: Training curves, confusion matrices, performance plots

## 🛠️ Requirements

### Optimized Training (Current v1.2)
```bash
pip install -r ml/requirements_v1.2.txt
```

## 📚 Documentation

- **MLOPS_INTEGRATION_GUIDE_v1.2.md** - Complete MLOps setup and usage
- **archive/ARCHIVE_INDEX.md** - Complete archive of all versions
- **archive/ml_docs/** - All previous documentation versions
- **archive/ml_models/** - All previous model versions

## 🎯 Training Targets

- **Dataset**: 1,402 unique ragas (603 Carnatic + 799 Hindustani)
- **Audio Files**: 74+ high-quality recordings
- **Training Time**: 2-5 hours on Mac GPU
- **Target Accuracy**: 87-90%
- **Model Size**: <50MB for mobile deployment

## 🔧 Troubleshooting

### Common Issues

1. **MLflow UI Not Starting**
   ```bash
   mlflow ui --backend-store-uri ./mlruns --port 5001
   ```

2. **Model Loading Issues**
   ```bash
   ls -la ./mlruns/0/[run_id]/artifacts/
   ```

3. **Permission Issues**
   ```bash
   chmod -R 755 ./mlruns
   ```

## 🎉 Success Metrics

- ✅ **Complete experiment tracking** with MLflow
- ✅ **Model versioning** and management
- ✅ **Real-time monitoring** of training progress
- ✅ **Rich visualizations** and analysis
- ✅ **Production deployment** ready
- ✅ **Comprehensive documentation**

## 🚀 Next Steps

1. **Start Training**: `./ml/start_mlflow_training.sh`
2. **View Results**: `mlflow ui --backend-store-uri ./mlruns`
3. **Deploy Model**: Use MLflow model serving
4. **Scale Training**: Use Lightning AI for cloud GPU training

---

**Ready to train with MLflow!** 🚀
