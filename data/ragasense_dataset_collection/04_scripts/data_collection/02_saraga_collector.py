#!/usr/bin/env python3
"""
Saraga Dataset Collector
Processes the existing Saraga dataset from MTG
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SaragaCollector:
    """Collector for Saraga dataset"""
    
    def __init__(self, source_dir: str = "../../ml/saraga", output_dir: str = "01_raw_data/saraga_mtg"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_source_exists(self) -> bool:
        """Check if source Saraga dataset exists"""
        if not self.source_dir.exists():
            logger.error(f"Source Saraga dataset not found at: {self.source_dir}")
            return False
        
        logger.info(f"✅ Source Saraga dataset found at: {self.source_dir}")
        return True
    
    def copy_dataset(self) -> bool:
        """Copy Saraga dataset to processing directory"""
        try:
            logger.info("Copying Saraga dataset...")
            
            # Copy entire directory structure
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            
            shutil.copytree(self.source_dir, self.output_dir)
            
            logger.info("✅ Saraga dataset copied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error copying Saraga dataset: {e}")
            return False
    
    def analyze_saraga_structure(self) -> Dict:
        """Analyze the structure of Saraga dataset"""
        try:
            logger.info("Analyzing Saraga dataset structure...")
            
            structure = {
                "carnatic": {
                    "files": [],
                    "audio_files": [],
                    "metadata_files": [],
                    "total_size": 0
                },
                "hindustani": {
                    "files": [],
                    "audio_files": [],
                    "metadata_files": [],
                    "total_size": 0
                },
                "total_files": 0,
                "total_size": 0
            }
            
            # Analyze Carnatic section
            carnatic_dir = self.output_dir / "carnatic"
            if carnatic_dir.exists():
                structure["carnatic"] = self._analyze_directory(carnatic_dir)
            
            # Analyze Hindustani section
            hindustani_dir = self.output_dir / "hindustani"
            if hindustani_dir.exists():
                structure["hindustani"] = self._analyze_directory(hindustani_dir)
            
            # Calculate totals
            structure["total_files"] = len(structure["carnatic"]["files"]) + len(structure["hindustani"]["files"])
            structure["total_size"] = structure["carnatic"]["total_size"] + structure["hindustani"]["total_size"]
            
            # Save structure analysis
            structure_file = self.output_dir / "saraga_structure.json"
            with open(structure_file, 'w') as f:
                json.dump(structure, f, indent=2)
            
            logger.info(f"✅ Saraga structure analyzed: {structure['total_files']} files, {structure['total_size']/1024/1024/1024:.2f} GB")
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing Saraga structure: {e}")
            return {}
    
    def _analyze_directory(self, directory: Path) -> Dict:
        """Analyze a specific directory"""
        analysis = {
            "files": [],
            "audio_files": [],
            "metadata_files": [],
            "total_size": 0
        }
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                file_size = file_path.stat().st_size
                
                analysis["files"].append({
                    "path": str(file_path.relative_to(self.output_dir)),
                    "size": file_size,
                    "type": file_path.suffix
                })
                analysis["total_size"] += file_size
                
                # Categorize files
                if file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a']:
                    analysis["audio_files"].append(str(file_path.relative_to(self.output_dir)))
                elif file_path.suffix.lower() in ['.json', '.csv', '.txt', '.xml']:
                    analysis["metadata_files"].append(str(file_path.relative_to(self.output_dir)))
        
        return analysis
    
    def extract_saraga_metadata(self) -> Dict:
        """Extract metadata from Saraga dataset"""
        try:
            logger.info("Extracting Saraga metadata...")
            
            metadata = {
                "carnatic": {
                    "songs": [],
                    "ragas": set(),
                    "artists": set(),
                    "total_duration": 0
                },
                "hindustani": {
                    "songs": [],
                    "ragas": set(),
                    "artists": set(),
                    "total_duration": 0
                }
            }
            
            # Process Carnatic metadata
            carnatic_dir = self.output_dir / "carnatic"
            if carnatic_dir.exists():
                metadata["carnatic"] = self._extract_metadata_from_dir(carnatic_dir)
            
            # Process Hindustani metadata
            hindustani_dir = self.output_dir / "hindustani"
            if hindustani_dir.exists():
                metadata["hindustani"] = self._extract_metadata_from_dir(hindustani_dir)
            
            # Convert sets to lists for JSON serialization
            metadata["carnatic"]["ragas"] = list(metadata["carnatic"]["ragas"])
            metadata["carnatic"]["artists"] = list(metadata["carnatic"]["artists"])
            metadata["hindustani"]["ragas"] = list(metadata["hindustani"]["ragas"])
            metadata["hindustani"]["artists"] = list(metadata["hindustani"]["artists"])
            
            # Save metadata
            metadata_file = self.output_dir / "saraga_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            total_ragas = len(metadata["carnatic"]["ragas"]) + len(metadata["hindustani"]["ragas"])
            total_songs = len(metadata["carnatic"]["songs"]) + len(metadata["hindustani"]["songs"])
            
            logger.info(f"✅ Saraga metadata extracted: {total_songs} songs, {total_ragas} ragas")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting Saraga metadata: {e}")
            return {}
    
    def _extract_metadata_from_dir(self, directory: Path) -> Dict:
        """Extract metadata from a specific directory"""
        metadata = {
            "songs": [],
            "ragas": set(),
            "artists": set(),
            "total_duration": 0
        }
        
        # Look for metadata files
        for metadata_file in directory.glob("**/*.json"):
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
        
        return metadata
    
    def collect(self) -> bool:
        """Main collection method"""
        try:
            logger.info("Starting Saraga dataset collection...")
            
            # Check if source exists
            if not self.check_source_exists():
                return False
            
            # Copy dataset
            if not self.copy_dataset():
                return False
            
            # Analyze structure
            structure = self.analyze_saraga_structure()
            if not structure:
                return False
            
            # Extract metadata
            metadata = self.extract_saraga_metadata()
            if not metadata:
                return False
            
            logger.info("✅ Saraga dataset collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in Saraga collection process: {e}")
            return False

def main():
    """Main function"""
    collector = SaragaCollector()
    success = collector.collect()
    
    if success:
        print("🎉 Saraga dataset collection completed successfully!")
    else:
        print("❌ Saraga dataset collection failed. Check logs for details.")

if __name__ == "__main__":
    main()
