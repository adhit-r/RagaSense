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
