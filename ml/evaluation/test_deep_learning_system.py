#!/usr/bin/env python3
"""
🧪 Deep Learning System Test Script
===================================

Test script to verify all components of the deep learning raga detection system.

Author: RagaSense AI Team
Date: 2024
"""

import os
import sys
import json
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from deep_learning_raga_model import AudioFeatureExtractor, RagaDetectionModel
from prepare_training_data import DataPreparator
from convex_data_connector import ConvexConnector, ConvexDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepLearningSystemTester:
    """Test all components of the deep learning system"""
    
    def __init__(self):
        self.test_results = {}
        
    def run_all_tests(self):
        """Run all system tests"""
        logger.info("🧪 Starting Deep Learning System Tests")
        
        try:
            # Test 1: Audio Feature Extraction
            self.test_audio_feature_extraction()
            
            # Test 2: Model Architecture
            self.test_model_architecture()
            
            # Test 3: Data Preparation
            self.test_data_preparation()
            
            # Test 4: Convex Integration
            self.test_convex_integration()
            
            # Test 5: End-to-End Pipeline
            self.test_end_to_end_pipeline()
            
            # Generate test report
            self.generate_test_report()
            
            logger.info("✅ All tests completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {str(e)}")
            raise
    
    def test_audio_feature_extraction(self):
        """Test audio feature extraction"""
        logger.info("🎵 Testing Audio Feature Extraction...")
        
        try:
            # Create feature extractor
            extractor = AudioFeatureExtractor(
                sample_rate=22050,
                duration=10,  # Shorter for testing
                hop_length=512
            )
            
            # Create dummy audio data
            dummy_audio_path = "test_audio.wav"
            self._create_dummy_audio(dummy_audio_path)
            
            # Extract features
            features = extractor.extract_features(dummy_audio_path)
            
            # Verify features
            assert features is not None, "Feature extraction failed"
            assert 'mfcc' in features, "MFCC features missing"
            assert 'chroma' in features, "Chroma features missing"
            assert 'spectral_centroid' in features, "Spectral features missing"
            
            # Check feature shapes
            assert features['mfcc'].shape[0] == 13, "MFCC should have 13 coefficients"
            assert features['chroma'].shape[0] == 12, "Chroma should have 12 bins"
            
            self.test_results['audio_feature_extraction'] = {
                'status': 'PASS',
                'features_extracted': list(features.keys()),
                'mfcc_shape': features['mfcc'].shape,
                'chroma_shape': features['chroma'].shape
            }
            
            # Clean up
            if os.path.exists(dummy_audio_path):
                os.remove(dummy_audio_path)
                
            logger.info("✅ Audio feature extraction test passed")
            
        except Exception as e:
            self.test_results['audio_feature_extraction'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            raise
    
    def test_model_architecture(self):
        """Test model architecture creation"""
        logger.info("🏗️ Testing Model Architecture...")
        
        try:
            # Test parameters
            num_ragas = 10
            input_shape = (64, 64, 1)  # Small input for testing
            
            # Create model
            model = RagaDetectionModel(
                num_ragas=num_ragas,
                input_shape=input_shape,
                model_name="test_model"
            )
            
            # Verify model
            assert model.model is not None, "Model creation failed"
            assert model.num_ragas == num_ragas, "Number of ragas mismatch"
            
            # Test model compilation
            model.compile_model(learning_rate=0.001)
            
            # Test model summary
            model_summary = []
            model.model.summary(print_fn=lambda x: model_summary.append(x))
            
            # Verify output shape
            test_input = np.random.random((1,) + input_shape)
            test_output = model.model.predict(test_input)
            assert test_output.shape == (1, num_ragas), f"Output shape mismatch: {test_output.shape}"
            
            self.test_results['model_architecture'] = {
                'status': 'PASS',
                'num_ragas': num_ragas,
                'input_shape': input_shape,
                'output_shape': test_output.shape,
                'model_layers': len(model.model.layers)
            }
            
            logger.info("✅ Model architecture test passed")
            
        except Exception as e:
            self.test_results['model_architecture'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            raise
    
    def test_data_preparation(self):
        """Test data preparation pipeline"""
        logger.info("📊 Testing Data Preparation...")
        
        try:
            # Create data preparator
            preparator = DataPreparator(base_dir=".")
            
            # Test external data processing
            external_data = preparator.prepare_external_data()
            
            # Test Convex data processing
            convex_data = preparator.prepare_convex_data()
            
            # Test data combination
            combined_data = preparator.combine_datasets(external_data, convex_data)
            
            # Verify data
            assert isinstance(external_data, list), "External data should be a list"
            assert isinstance(convex_data, list), "Convex data should be a list"
            assert isinstance(combined_data, list), "Combined data should be a list"
            
            self.test_results['data_preparation'] = {
                'status': 'PASS',
                'external_data_count': len(external_data),
                'convex_data_count': len(convex_data),
                'combined_data_count': len(combined_data),
                'unique_ragas': len(set(item['raga_name'] for item in combined_data))
            }
            
            logger.info("✅ Data preparation test passed")
            
        except Exception as e:
            self.test_results['data_preparation'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            raise
    
    def test_convex_integration(self):
        """Test Convex database integration"""
        logger.info("🗄️ Testing Convex Integration...")
        
        try:
            # Create connector
            connector = ConvexConnector()
            processor = ConvexDataProcessor(connector)
            
            # Test data retrieval
            ragas = connector.get_all_ragas()
            detections = connector.get_raga_detections()
            audio_files = connector.get_audio_files()
            
            # Test data processing
            training_data = processor.prepare_training_data()
            
            # Verify data
            assert isinstance(ragas, list), "Ragas should be a list"
            assert isinstance(detections, list), "Detections should be a list"
            assert isinstance(audio_files, list), "Audio files should be a list"
            assert isinstance(training_data, dict), "Training data should be a dict"
            
            # Check required keys
            required_keys = ['ragas', 'detections', 'audio_files', 'statistics']
            for key in required_keys:
                assert key in training_data, f"Missing key: {key}"
            
            self.test_results['convex_integration'] = {
                'status': 'PASS',
                'ragas_count': len(ragas),
                'detections_count': len(detections),
                'audio_files_count': len(audio_files),
                'training_data_keys': list(training_data.keys())
            }
            
            logger.info("✅ Convex integration test passed")
            
        except Exception as e:
            self.test_results['convex_integration'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            raise
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline"""
        logger.info("🔄 Testing End-to-End Pipeline...")
        
        try:
            # Create test data
            test_data = self._create_test_dataset()
            
            # Test feature extraction on test data
            extractor = AudioFeatureExtractor()
            features_list = []
            labels_list = []
            
            for item in test_data[:5]:  # Test with first 5 items
                if os.path.exists(item['audio_path']):
                    features = extractor.extract_features(item['audio_path'])
                    if features is not None:
                        # Convert features to vector (simplified)
                        feature_vector = self._features_to_simple_vector(features)
                        features_list.append(feature_vector)
                        labels_list.append(item['raga_name'])
            
            # Verify we have some data
            assert len(features_list) > 0, "No features extracted"
            assert len(labels_list) > 0, "No labels extracted"
            
            # Test model creation with actual data
            if len(features_list) > 0:
                # Create simple model for testing
                num_ragas = len(set(labels_list))
                feature_dim = len(features_list[0])
                
                # Reshape for CNN (simplified)
                input_shape = (int(np.sqrt(feature_dim)), int(np.sqrt(feature_dim)), 1)
                
                model = RagaDetectionModel(
                    num_ragas=num_ragas,
                    input_shape=input_shape,
                    model_name="test_pipeline_model"
                )
                
                model.compile_model()
                
                self.test_results['end_to_end_pipeline'] = {
                    'status': 'PASS',
                    'features_extracted': len(features_list),
                    'unique_ragas': num_ragas,
                    'feature_dimension': feature_dim,
                    'model_created': True
                }
            
            logger.info("✅ End-to-end pipeline test passed")
            
        except Exception as e:
            self.test_results['end_to_end_pipeline'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            raise
    
    def _create_dummy_audio(self, file_path: str):
        """Create dummy audio file for testing"""
        try:
            import librosa
            import soundfile as sf
            
            # Create simple sine wave
            sample_rate = 22050
            duration = 5  # seconds
            frequency = 440  # Hz (A4 note)
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * frequency * t)
            
            # Save as WAV
            sf.write(file_path, audio, sample_rate)
            
        except ImportError:
            # If librosa not available, create empty file
            with open(file_path, 'w') as f:
                f.write("dummy")
    
    def _create_test_dataset(self):
        """Create test dataset"""
        return [
            {
                'raga_name': 'Mohanam',
                'tradition': 'carnatic',
                'audio_path': 'test_audio.wav',
                'metadata': {}
            },
            {
                'raga_name': 'Hamsadhvani',
                'tradition': 'carnatic',
                'audio_path': 'test_audio.wav',
                'metadata': {}
            }
        ]
    
    def _features_to_simple_vector(self, features: dict) -> np.ndarray:
        """Convert features to simple vector for testing"""
        vector = []
        
        # Add basic features
        if 'mfcc_mean' in features:
            vector.extend(features['mfcc_mean'])
        if 'chroma_mean' in features:
            vector.extend(features['chroma_mean'])
        if 'spectral_centroid' in features:
            vector.append(features['spectral_centroid'])
        if 'tempo' in features:
            vector.append(features['tempo'])
        
        # Pad to minimum size
        while len(vector) < 100:
            vector.append(0.0)
        
        return np.array(vector[:100])  # Take first 100 features
    
    def generate_test_report(self):
        """Generate test report"""
        logger.info("📋 Generating Test Report...")
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = total_tests - passed_tests
        
        # Create report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "timestamp": str(np.datetime64('now'))
        }
        
        # Save report
        report_file = "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🧪 DEEP LEARNING SYSTEM TEST REPORT")
        print("="*60)
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {report['test_summary']['success_rate']:.2%}")
        print("\n📋 DETAILED RESULTS:")
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"  {status_icon} {test_name}: {result['status']}")
            if result['status'] == 'FAIL' and 'error' in result:
                print(f"     Error: {result['error']}")
        
        print("="*60)
        print(f"📄 Full report saved to: {report_file}")
        
        logger.info(f"✅ Test report generated: {report_file}")

def main():
    """Main test function"""
    logger.info("🧪 Starting Deep Learning System Tests")
    
    # Initialize tester
    tester = DeepLearningSystemTester()
    
    try:
        # Run all tests
        tester.run_all_tests()
        
        print("\n🎉 All tests completed successfully!")
        print("🚀 Your deep learning system is ready to use!")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {str(e)}")
        print(f"\n❌ Test suite failed: {str(e)}")
        print("🔧 Please check the error messages above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()



