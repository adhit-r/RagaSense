#!/bin/bash

# YuE Setup Script for RagaSense
# Setting up the latest 2025 music foundation model

echo "🚀 Setting up YuE for RagaSense..."

# Activate virtual environment
source ../raga_env/bin/activate

# Install YuE dependencies
echo "📦 Installing YuE dependencies..."
pip install transformers torch torchaudio
pip install librosa soundfile numpy
pip install accelerate bitsandbytes

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

# Test YuE installation
echo "🧪 Testing YuE installation..."
python -c "
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    print('✅ YuE dependencies installed successfully')
    
    # Test model loading
    model_name = 'm-a-p/YuE-s1-7B-anneal-en-icl'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('✅ YuE tokenizer loaded successfully')
    
except Exception as e:
    print(f'❌ YuE installation test failed: {e}')
"

# Create test script
echo "📝 Creating YuE test script..."
cat > test_yue.py << 'EOF'
#!/usr/bin/env python3
"""
Test YuE installation and basic functionality
"""

import torch
from transformers import AutoTokenizer, AutoModel

def test_yue():
    """Test YuE model loading and basic functionality"""
    try:
        print("🧪 Testing YuE model...")
        
        # Load YuE model
        model_name = 'm-a-p/YuE-s1-7B-anneal-en-icl'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print(f"✅ YuE model loaded: {model_name}")
        print(f"✅ Model device: {next(model.parameters()).device}")
        print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test basic inference
        test_prompt = "carnatic classical music raga anandabhairavi"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Basic inference successful")
        print(f"✅ Output shape: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ YuE test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_yue()
    if success:
        print("🎉 YuE setup complete and working!")
    else:
        print("❌ YuE setup failed. Check error messages above.")
EOF

# Run test
echo "🧪 Running YuE test..."
python test_yue.py

echo "🎉 YuE setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: python ml/yue_raga_classifier.py"
echo "2. Check: docs/YUE_INTEGRATION_PLAN.md"
echo "3. Start training with YuE!"
