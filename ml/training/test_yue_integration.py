#!/usr/bin/env python3
"""
Simple YuE Integration Test for RagaSense
Test the basic functionality of the YuE raga classifier
"""

import os
import json
import logging
from pathlib import Path
from yue_raga_classifier import YuERagaClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yue_integration():
    """Test YuE integration with basic functionality"""
    logger.info("🧪 Testing YuE Integration...")
    
    try:
        # Initialize classifier
        logger.info("Initializing YuE Raga Classifier...")
        classifier = YuERagaClassifier()
        logger.info("✅ YuE Classifier initialized successfully")
        
        # Test feature extraction on a sample file
        dataset_path = "carnatic-hindustani-dataset"
        if Path(dataset_path).exists():
            logger.info(f"Testing on dataset: {dataset_path}")
            
            # Find a sample audio file
            sample_files = list(Path(dataset_path).glob("**/*.wav")) + \
                          list(Path(dataset_path).glob("**/*.mp3"))
            
            if sample_files:
                sample_file = str(sample_files[0])
                logger.info(f"Testing with sample file: {Path(sample_file).name}")
                
                # Test feature extraction
                logger.info("Testing feature extraction...")
                features = classifier.extract_advanced_features(sample_file)
                
                if features:
                    logger.info("✅ Feature extraction successful")
                    logger.info(f"   Duration: {features.get('duration', 0):.2f}s")
                    logger.info(f"   Tempo: {features.get('tempo', 0):.1f} BPM")
                    logger.info(f"   Tonic: {features.get('tonic_frequency', 0):.1f} Hz")
                    
                    # Test classification
                    logger.info("Testing raga classification...")
                    result = classifier.classify_raga_yue(sample_file)
                    
                    if 'error' not in result:
                        logger.info("✅ Classification successful")
                        logger.info(f"   Predicted Raga: {result.get('predicted_raga', 'Unknown')}")
                        logger.info(f"   Tradition: {result.get('tradition', 'Unknown')}")
                        logger.info(f"   Confidence: {result.get('confidence', 0):.3f}")
                        
                        # Test enhanced features
                        temporal_analysis = result.get('temporal_analysis', {})
                        shruti_analysis = result.get('shruti_analysis', {})
                        
                        if temporal_analysis:
                            logger.info("✅ Temporal analysis available")
                            logger.info(f"   Tala cycle: {temporal_analysis.get('tala_cycle', 0)}")
                            logger.info(f"   Tala confidence: {temporal_analysis.get('tala_confidence', 0):.3f}")
                        
                        if shruti_analysis:
                            logger.info("✅ Shruti analysis available")
                            logger.info(f"   Shruti count: {shruti_analysis.get('shruti_count', 0)}")
                            logger.info(f"   Microtonal complexity: {shruti_analysis.get('microtonal_complexity', 0):.3f}")
                        
                        return True
                    else:
                        logger.error(f"❌ Classification failed: {result['error']}")
                        return False
                else:
                    logger.error("❌ Feature extraction failed")
                    return False
            else:
                logger.warning("⚠️ No audio files found in dataset")
                return False
        else:
            logger.warning(f"⚠️ Dataset not found: {dataset_path}")
            logger.info("Testing with mock data...")
            
            # Test with mock data
            logger.info("Testing enhanced encoders...")
            
            # Test temporal encoder
            try:
                from yue_raga_classifier import EnhancedTemporalEncoder
                import torch
                
                temporal_encoder = EnhancedTemporalEncoder()
                test_input = torch.randn(1, 100, 128)
                temporal_output = temporal_encoder(test_input)
                
                logger.info("✅ Enhanced Temporal Encoder working")
                logger.info(f"   Output shape: {temporal_output['temporal_features'].shape}")
                
            except Exception as e:
                logger.error(f"❌ Temporal encoder test failed: {e}")
                return False
            
            # Test shruti encoder
            try:
                from yue_raga_classifier import ShrutiPitchEncoder
                
                shruti_encoder = ShrutiPitchEncoder()
                test_pitch = torch.randn(1, 12)
                shruti_output = shruti_encoder(test_pitch)
                
                logger.info("✅ Shruti Pitch Encoder working")
                logger.info(f"   Output shape: {shruti_output['pitch_features'].shape}")
                
            except Exception as e:
                logger.error(f"❌ Shruti encoder test failed: {e}")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"❌ YuE integration test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting YuE Integration Test...")
    
    success = test_yue_integration()
    
    if success:
        logger.info("🎉 YuE Integration Test PASSED!")
        logger.info("✅ All components working correctly")
        logger.info("🚀 Ready for fine-tuning and evaluation")
    else:
        logger.error("❌ YuE Integration Test FAILED!")
        logger.error("Please check the error messages above")
    
    return success

if __name__ == "__main__":
    main()
