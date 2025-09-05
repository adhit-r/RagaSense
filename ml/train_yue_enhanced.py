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
