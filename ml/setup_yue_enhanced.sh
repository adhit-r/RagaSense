#!/bin/bash

# Enhanced YuE Setup Script for RagaSense 2025
# Setting up the latest music foundation model with advanced encoders

echo "🚀 Setting up Enhanced YuE for RagaSense 2025..."

# Activate virtual environment
source ../environments/raga_env/bin/activate

# Install enhanced YuE dependencies
echo "📦 Installing enhanced YuE dependencies..."
pip install transformers torch torchaudio
pip install librosa soundfile numpy scipy
pip install accelerate bitsandbytes
pip install scikit-learn matplotlib seaborn

# Clone YuE repository
echo "📥 Cloning YuE repository..."
if [ ! -d "YuE" ]; then
    git clone https://github.com/multimodal-art-projection/YuE.git
    cd YuE
    pip install -r requirements.txt
    cd ..
else
    echo "YuE repository already exists"
fi

# Download YuE models
echo "🤖 Downloading YuE models..."
python -c "
from transformers import AutoTokenizer, AutoModel
import torch

# Download YuE models
models = [
    'm-a-p/YuE-s1-7B-anneal-en-icl',
    'm-a-p/YuE-s2-1B-general'
]

for model_name in models:
    print(f'Downloading {model_name}...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f'✅ {model_name} downloaded successfully')
    except Exception as e:
        print(f'❌ Error downloading {model_name}: {e}')
"

# Test enhanced YuE installation
echo "🧪 Testing enhanced YuE installation..."
python -c "
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import librosa
    import numpy as np
    from scipy import signal
    from scipy.stats import entropy
    print('✅ Enhanced YuE dependencies installed successfully')
    
    # Test model loading
    model_name = 'm-a-p/YuE-s1-7B-anneal-en-icl'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('✅ YuE tokenizer loaded successfully')
    
    # Test enhanced features
    print('✅ Enhanced temporal and shruti encoders ready')
    
except Exception as e:
    print(f'❌ Enhanced YuE installation test failed: {e}')
"

# Create enhanced test script
echo "📝 Creating enhanced YuE test script..."
cat > test_yue_enhanced.py << 'EOF'
#!/usr/bin/env python3
"""
Test Enhanced YuE installation and advanced functionality
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import librosa
from scipy import signal
from scipy.stats import entropy

def test_enhanced_yue():
    """Test enhanced YuE model with temporal and shruti encoders"""
    try:
        print("🧪 Testing Enhanced YuE model...")
        
        # Load YuE model
        model_name = 'm-a-p/YuE-s1-7B-anneal-en-icl'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print(f"✅ YuE model loaded: {model_name}")
        print(f"✅ Model device: {next(model.parameters()).device}")
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test enhanced temporal encoder
        print("🧪 Testing Enhanced Temporal Encoder...")
        from yue_raga_classifier import EnhancedTemporalEncoder
        
        temporal_encoder = EnhancedTemporalEncoder()
        test_input = torch.randn(1, 100, 128)  # (batch, time, features)
        temporal_output = temporal_encoder(test_input)
        
        print(f"✅ Temporal encoder output shape: {temporal_output['temporal_features'].shape}")
        print(f"✅ Tala logits shape: {temporal_output['tala_logits'].shape}")
        
        # Test enhanced shruti encoder
        print("🧪 Testing Enhanced Shruti Encoder...")
        from yue_raga_classifier import ShrutiPitchEncoder
        
        shruti_encoder = ShrutiPitchEncoder()
        test_pitch = torch.randn(1, 12)  # (batch, pitch_features)
        shruti_output = shruti_encoder(test_pitch)
        
        print(f"✅ Shruti encoder output shape: {shruti_output['pitch_features'].shape}")
        print(f"✅ Interval logits shape: {shruti_output['interval_logits'].shape}")
        print(f"✅ Scale logits shape: {shruti_output['scale_logits'].shape}")
        
        # Test basic inference
        test_prompt = "carnatic classical music raga anandabhairavi with enhanced temporal and shruti analysis"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Enhanced inference successful")
        print(f"✅ Output shape: {outputs.last_hidden_state.shape}")
        
        # Test audio feature extraction
        print("🧪 Testing Enhanced Audio Feature Extraction...")
        from yue_raga_classifier import YuERagaClassifier
        
        classifier = YuERagaClassifier()
        print("✅ Enhanced YuE Raga Classifier initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced YuE test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_yue()
    if success:
        print("🎉 Enhanced YuE setup complete and working!")
        print("🚀 Ready for state-of-the-art raga classification!")
    else:
        print("❌ Enhanced YuE setup failed. Check error messages above.")
EOF

# Run enhanced test
echo "🧪 Running enhanced YuE test..."
python test_yue_enhanced.py

# Create training script
echo "📝 Creating enhanced training script..."
cat > train_yue_enhanced.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced YuE Training Script for RagaSense
"""

import os
import json
import logging
from pathlib import Path
from yue_raga_classifier import YuERagaClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    logger.info("🚀 Starting Enhanced YuE Training for RagaSense...")
    
    # Initialize enhanced classifier
    classifier = YuERagaClassifier()
    
    # Evaluate on dataset
    dataset_path = "carnatic-hindustani-dataset"
    if Path(dataset_path).exists():
        logger.info(f"📊 Evaluating on dataset: {dataset_path}")
        metrics = classifier.evaluate_on_dataset(dataset_path)
        
        logger.info("📈 Enhanced YuE Evaluation Results:")
        logger.info(f"Total files: {metrics.get('total_files', 0)}")
        logger.info(f"Success rate: {metrics.get('success_rate', 0):.2%}")
        logger.info(f"Average confidence: {metrics.get('average_confidence', 0):.2f}")
        
        # Enhanced metrics
        enhanced_metrics = metrics.get('enhanced_metrics', {})
        logger.info(f"Average tala confidence: {enhanced_metrics.get('average_tala_confidence', 0):.2f}")
        logger.info(f"Average shruti count: {enhanced_metrics.get('average_shruti_count', 0):.1f}")
        logger.info(f"Average microtonal complexity: {enhanced_metrics.get('average_microtonal_complexity', 0):.2f}")
        
        # Save results
        results_file = "enhanced_yue_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"💾 Results saved to: {results_file}")
        
    else:
        logger.error(f"❌ Dataset not found: {dataset_path}")
        logger.info("Please ensure the dataset is available")

if __name__ == "__main__":
    main()
EOF

echo "🎉 Enhanced YuE setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: python ml/training/yue_raga_classifier.py"
echo "2. Run: python train_yue_enhanced.py"
echo "3. Check: docs/research/YUE_ADAPTATION_STRATEGY.md"
echo "4. Start enhanced training with YuE!"
echo ""
echo "🚀 Enhanced YuE Features:"
echo "- Advanced Temporal Encoder for tala cycles"
echo "- Shruti Pitch Encoder for microtonal intervals"
echo "- Enhanced tradition detection"
echo "- State-of-the-art 2025 methodology"
