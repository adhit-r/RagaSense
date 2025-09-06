#!/usr/bin/env python3
"""
Kaggle Carnatic Song Database Collector
Downloads and processes the Kaggle Carnatic song database
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleCarnaticCollector:
    """Collector for Kaggle Carnatic song database"""
    
    def __init__(self, output_dir: str = "01_raw_data/kaggle_carnatic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = "sanjaynatesan/carnatic-song-database"
        
    def check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured"""
        try:
            # Check if kaggle.json exists
            kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_config.exists():
                logger.error("Kaggle API key not found. Please set up kaggle.json")
                return False
            
            # Test Kaggle API
            result = subprocess.run(["kaggle", "datasets", "list", "--user", "sanjaynatesan"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Kaggle API test failed: {result.stderr}")
                return False
            
            logger.info("✅ Kaggle API is properly configured")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Kaggle setup: {e}")
            return False
    
    def download_dataset(self) -> bool:
        """Download the Kaggle dataset"""
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            
            # Change to output directory
            os.chdir(self.output_dir)
            
            # Download dataset
            result = subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", self.dataset_name,
                "--unzip"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                return False
            
            logger.info("✅ Dataset downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    def analyze_dataset_structure(self) -> Dict:
        """Analyze the structure of the downloaded dataset"""
        try:
            logger.info("Analyzing dataset structure...")
            
            structure = {
                "files": [],
                "directories": [],
                "total_size": 0,
                "audio_files": [],
                "metadata_files": []
            }
            
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    file_path = Path(root) / file
                    file_size = file_path.stat().st_size
                    structure["files"].append({
                        "path": str(file_path.relative_to(self.output_dir)),
                        "size": file_size,
                        "type": file_path.suffix
                    })
                    structure["total_size"] += file_size
                    
                    # Categorize files
                    if file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                        structure["audio_files"].append(str(file_path.relative_to(self.output_dir)))
                    elif file_path.suffix.lower() in ['.json', '.csv', '.txt', '.xml']:
                        structure["metadata_files"].append(str(file_path.relative_to(self.output_dir)))
            
            # Save structure analysis
            structure_file = self.output_dir / "dataset_structure.json"
            with open(structure_file, 'w') as f:
                json.dump(structure, f, indent=2)
            
            logger.info(f"✅ Dataset structure analyzed: {len(structure['files'])} files, {len(structure['audio_files'])} audio files")
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing dataset structure: {e}")
            return {}
    
    def extract_metadata(self) -> Dict:
        """Extract metadata from the dataset"""
        try:
            logger.info("Extracting metadata...")
            
            metadata = {
                "songs": [],
                "ragas": set(),
                "artists": set(),
                "total_duration": 0
            }
            
            # Look for metadata files
            for metadata_file in Path(self.output_dir).glob("**/*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                metadata["songs"].append(item)
                                if "raga" in item:
                                    metadata["ragas"].add(item["raga"])
                                if "artist" in item:
                                    metadata["artists"].add(item["artist"])
                    elif isinstance(data, dict):
                        metadata["songs"].append(data)
                        if "raga" in data:
                            metadata["ragas"].add(data["raga"])
                        if "artist" in data:
                            metadata["artists"].add(data["artist"])
                            
                except Exception as e:
                    logger.warning(f"Could not parse metadata file {metadata_file}: {e}")
            
            # Convert sets to lists for JSON serialization
            metadata["ragas"] = list(metadata["ragas"])
            metadata["artists"] = list(metadata["artists"])
            
            # Save metadata
            metadata_file = self.output_dir / "extracted_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Metadata extracted: {len(metadata['songs'])} songs, {len(metadata['ragas'])} ragas")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def collect(self) -> bool:
        """Main collection method"""
        try:
            logger.info("Starting Kaggle Carnatic dataset collection...")
            
            # Check Kaggle setup
            if not self.check_kaggle_setup():
                return False
            
            # Download dataset
            if not self.download_dataset():
                return False
            
            # Analyze structure
            structure = self.analyze_dataset_structure()
            if not structure:
                return False
            
            # Extract metadata
            metadata = self.extract_metadata()
            if not metadata:
                return False
            
            logger.info("✅ Kaggle dataset collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in collection process: {e}")
            return False

def main():
    """Main function"""
    collector = KaggleCarnaticCollector()
    success = collector.collect()
    
    if success:
        print("🎉 Kaggle dataset collection completed successfully!")
    else:
        print("❌ Kaggle dataset collection failed. Check logs for details.")

if __name__ == "__main__":
    main()
