#!/usr/bin/env python3
"""
🧪 Basic Component Test Script
==============================

Test basic components without TensorFlow to avoid hardware compatibility issues.

Author: RagaSense AI Team
Date: 2024
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasicComponentTester:
    """Test basic components without TensorFlow"""
    
    def __init__(self):
        self.test_results = {}
        
    def run_basic_tests(self):
        """Run basic component tests"""
        logger.info("🧪 Starting Basic Component Tests")
        
        try:
            # Test 1: File Structure
            self.test_file_structure()
            
            # Test 2: Data Preparation
            self.test_data_preparation()
            
            # Test 3: Convex Integration
            self.test_convex_integration()
            
            # Test 4: Audio Processing (if librosa available)
            self.test_audio_processing()
            
            # Generate test report
            self.generate_test_report()
            
            logger.info("✅ Basic tests completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {str(e)}")
            raise
    
    def test_file_structure(self):
        """Test that all required files exist"""
        logger.info("📁 Testing File Structure...")
        
        required_files = [
            "deep_learning_raga_model.py",
            "prepare_training_data.py", 
            "convex_data_connector.py",
            "train_deep_learning_model.py",
            "requirements_deep_learning.txt",
            "README_DEEP_LEARNING.md"
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            self.test_results['file_structure'] = {
                'status': 'FAIL',
                'missing_files': missing_files
            }
            raise Exception(f"Missing files: {missing_files}")
        else:
            self.test_results['file_structure'] = {
                'status': 'PASS',
                'files_found': len(required_files)
            }
            
        logger.info("✅ File structure test passed")
    
    def test_data_preparation(self):
        """Test data preparation components"""
        logger.info("📊 Testing Data Preparation...")
        
        try:
            # Import data preparation module
            sys.path.append('.')
            from prepare_training_data import DataPreparator
            
            # Create preparator
            preparator = DataPreparator()
            
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
        """Test Convex integration components"""
        logger.info("🗄️ Testing Convex Integration...")
        
        try:
            # Import convex connector
            from convex_data_connector import ConvexConnector, ConvexDataProcessor
            
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
    
    def test_audio_processing(self):
        """Test audio processing (if librosa available)"""
        logger.info("🎵 Testing Audio Processing...")
        
        try:
            # Try to import librosa
            import librosa
            import soundfile as sf
            
            # Create dummy audio
            sample_rate = 22050
            duration = 2  # seconds
            frequency = 440  # Hz (A4 note)
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * frequency * t)
            
            # Save as WAV
            dummy_audio_path = "test_audio.wav"
            sf.write(dummy_audio_path, audio, sample_rate)
            
            # Test basic librosa functionality
            y, sr = librosa.load(dummy_audio_path, sr=sample_rate)
            
            # Clean up
            if os.path.exists(dummy_audio_path):
                os.remove(dummy_audio_path)
            
            self.test_results['audio_processing'] = {
                'status': 'PASS',
                'librosa_available': True,
                'audio_loaded': True,
                'sample_rate': sr
            }
            
            logger.info("✅ Audio processing test passed")
            
        except ImportError:
            self.test_results['audio_processing'] = {
                'status': 'SKIP',
                'librosa_available': False,
                'message': 'Librosa not installed - skipping audio tests'
            }
            logger.info("⚠️ Librosa not available - skipping audio tests")
            
        except Exception as e:
            self.test_results['audio_processing'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            raise
    
    def generate_test_report(self):
        """Generate test report"""
        logger.info("📋 Generating Test Report...")
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAIL')
        skipped_tests = sum(1 for result in self.test_results.values() if result['status'] == 'SKIP')
        
        # Create report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "skipped_tests": skipped_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "next_steps": self._get_next_steps()
        }
        
        # Save report
        report_file = "basic_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("🧪 BASIC COMPONENT TEST REPORT")
        print("="*60)
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⏭️ Skipped: {skipped_tests}")
        print(f"📈 Success Rate: {report['test_summary']['success_rate']:.2%}")
        print("\n📋 DETAILED RESULTS:")
        
        for test_name, result in self.test_results.items():
            if result['status'] == 'PASS':
                status_icon = "✅"
            elif result['status'] == 'FAIL':
                status_icon = "❌"
            else:
                status_icon = "⏭️"
            print(f"  {status_icon} {test_name}: {result['status']}")
            if result['status'] == 'FAIL' and 'error' in result:
                print(f"     Error: {result['error']}")
        
        print("\n🚀 NEXT STEPS:")
        for step in report['next_steps']:
            print(f"  • {step}")
        
        print("="*60)
        print(f"📄 Full report saved to: {report_file}")
        
        logger.info(f"✅ Test report generated: {report_file}")
    
    def _get_next_steps(self):
        """Get next steps based on test results"""
        steps = []
        
        if self.test_results.get('audio_processing', {}).get('status') == 'SKIP':
            steps.append("Install librosa: pip install librosa soundfile")
        
        if all(result['status'] == 'PASS' for result in self.test_results.values()):
            steps.append("Install TensorFlow: pip install tensorflow")
            steps.append("Run full test: python test_deep_learning_system.py")
            steps.append("Start training: python train_deep_learning_model.py")
        else:
            steps.append("Fix any failed tests before proceeding")
        
        return steps

def main():
    """Main test function"""
    logger.info("🧪 Starting Basic Component Tests")
    
    # Initialize tester
    tester = BasicComponentTester()
    
    try:
        # Run basic tests
        tester.run_basic_tests()
        
        print("\n🎉 Basic tests completed successfully!")
        print("🚀 Core components are working!")
        
    except Exception as e:
        logger.error(f"❌ Basic test suite failed: {str(e)}")
        print(f"\n❌ Basic test suite failed: {str(e)}")
        print("🔧 Please check the error messages above and fix any issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()



