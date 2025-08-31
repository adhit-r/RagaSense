#!/usr/bin/env python3
"""
Test script to load the trained model checkpoint safely
"""

import torch
from torch.serialization import safe_globals
from ruamel.yaml.scalarfloat import ScalarFloat

try:
    # Try safe loading first
    with safe_globals([ScalarFloat]):
        checkpoint = torch.load(
            'carnatic-raga-classifier/ckpts/resnet_0.7/150classes_alldata_cliplength30/training_checkpoints/best_ckpt.tar', 
            map_location='cpu', 
            weights_only=True
        )
        print('✅ Checkpoint loaded successfully with safe loading')
        print(f'Keys: {list(checkpoint.keys())}')
        print(f'Model state keys: {len(list(checkpoint["model_state"].keys()))}')
        
except Exception as e:
    print(f"❌ Safe loading failed: {e}")
    try:
        # Try unsafe loading as fallback
        checkpoint = torch.load(
            'carnatic-raga-classifier/ckpts/resnet_0.7/150classes_alldata_cliplength30/training_checkpoints/best_ckpt.tar', 
            map_location='cpu', 
            weights_only=False
        )
        print('✅ Checkpoint loaded successfully with unsafe loading')
        print(f'Keys: {list(checkpoint.keys())}')
        print(f'Model state keys: {len(list(checkpoint["model_state"].keys()))}')
        
    except Exception as e2:
        print(f"❌ Unsafe loading also failed: {e2}")
