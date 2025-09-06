# RagaSense ML Training - Naming Conventions

## 📋 **File Naming Standards**

### **Python Scripts**
- **Format**: `{component}_{type}_{version}.py`
- **Examples**:
  - `raga_classifier_cnn_v1.0.py` - CNN-based raga classifier v1.0
  - `audio_processor_advanced_v2.0.py` - Advanced audio processor v2.0
  - `dataset_expander_v1.0.py` - Dataset expansion utility v1.0

### **Model Files**
- **Format**: `{model_type}_{version}.pt` or `{model_type}_{version}.pkl`
- **Examples**:
  - `raga_classifier_cnn_v1.0.pt` - CNN model v1.0
  - `yue_raga_detector_v1.0.pt` - YuE-based detector v1.0

### **Test Files**
- **Format**: `test_{component}_{version}.py`
- **Examples**:
  - `test_raga_classifier_cnn_v1.0.py` - Test for CNN classifier v1.0
  - `test_audio_processor_advanced_v2.0.py` - Test for advanced processor v2.0

### **Results Files**
- **Format**: `{type}_{component}_{version}.{extension}`
- **Examples**:
  - `evaluation_cnn_v1.0.json` - CNN evaluation results v1.0
  - `training_curves_cnn_v1.0.png` - CNN training curves v1.0
  - `confusion_matrix_cnn_v1.0.png` - CNN confusion matrix v1.0

### **Configuration Files**
- **Format**: `config_{component}_{version}.json`
- **Examples**:
  - `config_training_cnn_v1.0.json` - CNN training configuration v1.0
  - `config_dataset_expansion_v1.0.json` - Dataset expansion config v1.0

## 📁 **Directory Structure**

```
ml/training/
├── 📁 archive/                    # Archived/obsolete code
│   ├── 📁 obsolete/              # Completely obsolete files
│   ├── 📁 experiments/           # Experimental/development code
│   └── 📁 old_versions/          # Previous versions of working code
│
├── 📁 models/                    # Trained model files
│   ├── raga_classifier_cnn_v1.0.pt
│   └── yue_raga_detector_v1.0.pt
│
├── 📁 results/                   # Analysis results and visualizations
│   ├── evaluation_cnn_v1.0.json
│   ├── training_curves_cnn_v1.0.png
│   └── confusion_matrix_cnn_v1.0.png
│
├── 📁 data/                      # Processed datasets
├── 📁 logs/                      # Training and processing logs
├── 📁 checkpoints/               # Model checkpoints during training
│
├── raga_classifier_cnn_v1.0.py   # Current working CNN classifier
├── test_raga_classifier_cnn_v1.0.py  # Test for CNN classifier
└── NAMING_CONVENTIONS.md         # This file
```

## 🏷️ **Version Numbering**

### **Version Format**: `v{major}.{minor}`
- **Major Version**: Significant changes, breaking changes, new architecture
- **Minor Version**: Bug fixes, improvements, new features

### **Examples**:
- `v1.0` - Initial stable release
- `v1.1` - Bug fixes and minor improvements
- `v2.0` - Major architecture change or new approach

## 📦 **Archiving Rules**

### **Archive to `obsolete/`**:
- Files that are completely outdated and won't be used again
- Failed experiments that didn't work
- Superseded by better implementations

### **Archive to `experiments/`**:
- Experimental code that might be useful for reference
- Development versions that aren't production-ready
- Research prototypes and proof-of-concepts

### **Archive to `old_versions/`**:
- Previous versions of working code
- Files that were replaced by newer versions
- Historical versions for comparison

## 🔄 **Migration Process**

### **When Creating New Files**:
1. Follow naming convention exactly
2. Include version number
3. Place in appropriate directory
4. Update this document if needed

### **When Archiving Files**:
1. Determine appropriate archive category
2. Move to correct archive subdirectory
3. Update any references in documentation
4. Commit changes with clear message

### **When Updating Files**:
1. Increment version number appropriately
2. Archive old version if significant changes
3. Update all references and imports
4. Update test files accordingly

## ✅ **Current Active Files**

### **Production Ready**:
- `raga_classifier_cnn_v1.0.py` - Working CNN classifier
- `test_raga_classifier_cnn_v1.0.py` - Associated tests

### **Archived**:
- `archive/obsolete/` - Obsolete YouTube processing and Saraga integration
- `archive/experiments/` - YuE experimental implementations
- `archive/old_versions/` - Previous working versions

## 🎯 **Benefits**

- **Clear Versioning**: Easy to track changes and rollbacks
- **Organized Archives**: Old code preserved but not cluttering workspace
- **Consistent Naming**: Easy to find and understand files
- **Professional Structure**: Industry-standard organization
- **Maintainable**: Easy to update and manage over time

---

**Last Updated**: September 6, 2025  
**Maintained by**: Adhithya Rajasekaran (@adhit-r)
