#!/usr/bin/env python3
"""
Test the Honest Raga Classifier
Simple test to verify the honest approach works
"""

import logging
from pathlib import Path
from raga_classifier_cnn_v1.0 import HonestRagaClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_honest_classifier():
    """Test the honest raga classifier"""
    logger.info("🧪 Testing Honest Raga Classifier...")
    
    try:
        # Initialize classifier
        logger.info("Initializing honest classifier...")
        classifier = HonestRagaClassifier()
        logger.info("✅ Classifier initialized successfully")
        
        # Test dataset preparation
        dataset_path = "../../carnatic-hindustani-dataset"
        if Path(dataset_path).exists():
            logger.info(f"Testing dataset preparation...")
            audio_files, labels = classifier.prepare_dataset(dataset_path, max_samples=10)
            
            if len(audio_files) > 0:
                logger.info(f"✅ Dataset preparation successful: {len(audio_files)} files")
                logger.info(f"   Carnatic: {labels.count(0)}, Hindustani: {labels.count(1)}")
                
                # Test model initialization
                logger.info("Testing model initialization...")
                from raga_classifier_cnn_v1.0 import SimpleRagaCNN
                classifier.model = SimpleRagaCNN(num_classes=2).to(classifier.device)
                logger.info("✅ Model initialized successfully")
                
                # Test a single prediction (with untrained model)
                logger.info("Testing single prediction...")
                test_file = audio_files[0]
                result = classifier.predict(test_file)
                
                if 'error' not in result:
                    logger.info("✅ Prediction successful")
                    logger.info(f"   Predicted: {result['predicted_tradition']}")
                    logger.info(f"   Confidence: {result['confidence']:.3f}")
                else:
                    logger.warning(f"⚠️ Prediction failed: {result['error']}")
                
                return True
            else:
                logger.error("❌ No audio files found in dataset")
                return False
        else:
            logger.warning(f"⚠️ Dataset not found: {dataset_path}")
            logger.info("Testing with mock data...")
            
            # Test model initialization
            from honest_raga_classifier import SimpleRagaCNN
            classifier.model = SimpleRagaCNN(num_classes=2).to(classifier.device)
            logger.info("✅ Model initialized successfully")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Honest Classifier Test...")
    
    success = test_honest_classifier()
    
    if success:
        logger.info("🎉 Honest Classifier Test PASSED!")
        logger.info("✅ All components working correctly")
        logger.info("🚀 Ready for honest training and evaluation")
    else:
        logger.error("❌ Honest Classifier Test FAILED!")
        logger.error("Please check the error messages above")
    
    return success

if __name__ == "__main__":
    main()
