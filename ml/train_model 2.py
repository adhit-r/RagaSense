#!/usr/bin/env python3
"""
Training script for the Raga Classifier ML model.
This script helps train the neural network model for raga detection.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
from app.ml.raga_classifier import RagaClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_audio_files(data_dir: str) -> Dict[str, List[str]]:
    """Find all audio files in the data directory organized by raga."""
    raga_files = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")
    
    # Walk through the directory structure
    for raga_dir in data_path.iterdir():
        if raga_dir.is_dir():
            raga_name = raga_dir.name
            audio_files = []
            
            # Find all audio files in this raga directory
            for audio_file in raga_dir.rglob("*"):
                if audio_file.is_file() and audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
                    audio_files.append(str(audio_file))
            
            if audio_files:
                raga_files[raga_name] = audio_files
                logger.info(f"Found {len(audio_files)} audio files for raga: {raga_name}")
    
    return raga_files

def prepare_training_data(raga_files: Dict[str, List[str]]) -> tuple:
    """Prepare training data from raga files."""
    audio_files = []
    raga_labels = []
    
    for raga_name, files in raga_files.items():
        audio_files.extend(files)
        raga_labels.extend([raga_name] * len(files))
    
    logger.info(f"Total audio files: {len(audio_files)}")
    logger.info(f"Total ragas: {len(set(raga_labels))}")
    
    return audio_files, raga_labels

def train_model(data_dir: str, model_save_path: str, epochs: int = 100, validation_split: float = 0.2):
    """Train the raga classifier model."""
    try:
        # Find audio files
        logger.info(f"Scanning for audio files in {data_dir}")
        raga_files = find_audio_files(data_dir)
        
        if not raga_files:
            raise ValueError("No audio files found in the data directory")
        
        # Prepare training data
        audio_files, raga_labels = prepare_training_data(raga_files)
        
        # Create and train classifier
        logger.info("Creating raga classifier...")
        classifier = RagaClassifier()
        
        logger.info("Starting model training...")
        training_result = classifier.train(
            audio_files=audio_files,
            raga_labels=raga_labels,
            validation_split=validation_split,
            epochs=epochs
        )
        
        # Save the trained model
        logger.info(f"Saving model to {model_save_path}")
        classifier.save_model(model_save_path)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {model_save_path}")
        logger.info(f"Number of classes: {training_result['num_classes']}")
        logger.info(f"Classes: {training_result['class_names']}")
        
        return classifier, training_result
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Train Raga Classifier Model")
    parser.add_argument("--data-dir", required=True, help="Directory containing audio files organized by raga")
    parser.add_argument("--model-save-path", required=True, help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio")
    
    args = parser.parse_args()
    
    try:
        train_model(
            data_dir=args.data_dir,
            model_save_path=args.model_save_path,
            epochs=args.epochs,
            validation_split=args.validation_split
        )
        print("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 