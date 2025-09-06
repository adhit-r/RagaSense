#!/usr/bin/env python3
"""
Ramanarunachalam Music Repository Collector
Downloads and processes the GitHub repository containing Carnatic and Hindustani music
"""

import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RamanarunachalamMusicCollector:
    """Collector for ramanarunachalam/Music GitHub repository"""
    
    def __init__(self, output_dir: str = "01_raw_data/ramanarunachalam_music"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.repo_url = "https://github.com/ramanarunachalam/Music.git"
        self.repo_name = "ramanarunachalam/Music"
        
    def clone_repository(self) -> bool:
        """Clone the GitHub repository"""
        try:
            logger.info(f"Cloning repository: {self.repo_url}")
            
            # Remove existing directory if it exists
            repo_dir = self.output_dir / "Music"
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            
            # Clone the repository
            result = subprocess.run([
                "git", "clone", self.repo_url, str(repo_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to clone repository: {result.stderr}")
                return False
            
            logger.info("✅ Repository cloned successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return False
    
    def analyze_repository_structure(self) -> Dict:
        """Analyze the structure of the cloned repository"""
        try:
            logger.info("Analyzing repository structure...")
            
            repo_dir = self.output_dir / "Music"
            if not repo_dir.exists():
                logger.error("Repository directory not found")
                return {}
            
            structure = {
                "carnatic": {
                    "files": [],
                    "directories": [],
                    "total_size": 0,
                    "file_types": {}
                },
                "hindustani": {
                    "files": [],
                    "directories": [],
                    "total_size": 0,
                    "file_types": {}
                },
                "total_files": 0,
                "total_size": 0,
                "languages": {
                    "JavaScript": 0,
                    "HTML": 0,
                    "CSS": 0,
                    "Other": 0
                }
            }
            
            # Analyze Carnatic section
            carnatic_dir = repo_dir / "Carnatic"
            if carnatic_dir.exists():
                structure["carnatic"] = self._analyze_directory(carnatic_dir)
            
            # Analyze Hindustani section
            hindustani_dir = repo_dir / "Hindustani"
            if hindustani_dir.exists():
                structure["hindustani"] = self._analyze_directory(hindustani_dir)
            
            # Calculate totals
            structure["total_files"] = len(structure["carnatic"]["files"]) + len(structure["hindustani"]["files"])
            structure["total_size"] = structure["carnatic"]["total_size"] + structure["hindustani"]["total_size"]
            
            # Save structure analysis
            structure_file = self.output_dir / "repository_structure.json"
            with open(structure_file, 'w') as f:
                json.dump(structure, f, indent=2)
            
            logger.info(f"✅ Repository structure analyzed: {structure['total_files']} files, {structure['total_size']/1024/1024:.2f} MB")
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing repository structure: {e}")
            return {}
    
    def _analyze_directory(self, directory: Path) -> Dict:
        """Analyze a specific directory"""
        analysis = {
            "files": [],
            "directories": [],
            "total_size": 0,
            "file_types": {}
        }
        
        for root, dirs, files in os.walk(directory):
            # Add directories
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                analysis["directories"].append({
                    "name": dir_name,
                    "path": str(dir_path.relative_to(directory)),
                    "type": "directory"
                })
            
            # Add files
            for file in files:
                file_path = Path(root) / file
                file_size = file_path.stat().st_size
                file_ext = file_path.suffix.lower()
                
                analysis["files"].append({
                    "name": file,
                    "path": str(file_path.relative_to(directory)),
                    "size": file_size,
                    "type": file_ext
                })
                analysis["total_size"] += file_size
                
                # Count file types
                if file_ext in analysis["file_types"]:
                    analysis["file_types"][file_ext] += 1
                else:
                    analysis["file_types"][file_ext] = 1
        
        return analysis
    
    def extract_music_metadata(self) -> Dict:
        """Extract metadata from music files"""
        try:
            logger.info("Extracting music metadata...")
            
            metadata = {
                "carnatic": {
                    "compositions": [],
                    "ragas": set(),
                    "artists": set(),
                    "file_types": set()
                },
                "hindustani": {
                    "compositions": [],
                    "ragas": set(),
                    "artists": set(),
                    "file_types": set()
                },
                "repository_info": {
                    "name": self.repo_name,
                    "url": self.repo_url,
                    "stars": 27,
                    "forks": 17,
                    "languages": ["JavaScript", "HTML", "CSS"]
                }
            }
            
            repo_dir = self.output_dir / "Music"
            
            # Process Carnatic files
            carnatic_dir = repo_dir / "Carnatic"
            if carnatic_dir.exists():
                metadata["carnatic"] = self._extract_metadata_from_dir(carnatic_dir, "Carnatic")
            
            # Process Hindustani files
            hindustani_dir = repo_dir / "Hindustani"
            if hindustani_dir.exists():
                metadata["hindustani"] = self._extract_metadata_from_dir(hindustani_dir, "Hindustani")
            
            # Convert sets to lists for JSON serialization
            metadata["carnatic"]["ragas"] = list(metadata["carnatic"]["ragas"])
            metadata["carnatic"]["artists"] = list(metadata["carnatic"]["artists"])
            metadata["carnatic"]["file_types"] = list(metadata["carnatic"]["file_types"])
            metadata["hindustani"]["ragas"] = list(metadata["hindustani"]["ragas"])
            metadata["hindustani"]["artists"] = list(metadata["hindustani"]["artists"])
            metadata["hindustani"]["file_types"] = list(metadata["hindustani"]["file_types"])
            
            # Save metadata
            metadata_file = self.output_dir / "extracted_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            total_compositions = len(metadata["carnatic"]["compositions"]) + len(metadata["hindustani"]["compositions"])
            total_ragas = len(metadata["carnatic"]["ragas"]) + len(metadata["hindustani"]["ragas"])
            
            logger.info(f"✅ Metadata extracted: {total_compositions} compositions, {total_ragas} ragas")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def _extract_metadata_from_dir(self, directory: Path, tradition: str) -> Dict:
        """Extract metadata from a specific directory"""
        metadata = {
            "compositions": [],
            "ragas": set(),
            "artists": set(),
            "file_types": set()
        }
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                file_ext = file_path.suffix.lower()
                
                # Add file type
                metadata["file_types"].add(file_ext)
                
                # Extract metadata from filename and path
                composition_info = {
                    "filename": file,
                    "path": str(file_path.relative_to(directory)),
                    "tradition": tradition,
                    "file_type": file_ext,
                    "size": file_path.stat().st_size
                }
                
                # Try to extract raga and artist from filename/path
                filename_lower = file.lower()
                path_lower = str(file_path.relative_to(directory)).lower()
                
                # Common raga names to look for
                common_ragas = [
                    "yaman", "bhairavi", "kalyani", "kambhoji", "shankarabharanam",
                    "mohanam", "hindolam", "sankarabharanam", "todi", "kharaharapriya",
                    "begada", "sahana", "vasantha", "sri", "nattakurinji", "kapi",
                    "anandabhairavi", "madhuvanti", "bageshri", "desh", "bihag"
                ]
                
                # Look for raga in filename or path
                for raga in common_ragas:
                    if raga in filename_lower or raga in path_lower:
                        composition_info["raga"] = raga.title()
                        metadata["ragas"].add(raga.title())
                        break
                
                # Look for artist indicators
                artist_indicators = ["artist", "singer", "vocalist", "performer"]
                for indicator in artist_indicators:
                    if indicator in filename_lower or indicator in path_lower:
                        # Extract potential artist name
                        parts = filename_lower.split(indicator)
                        if len(parts) > 1:
                            potential_artist = parts[1].split('.')[0].strip('_-').title()
                            if potential_artist and len(potential_artist) > 2:
                                composition_info["artist"] = potential_artist
                                metadata["artists"].add(potential_artist)
                        break
                
                metadata["compositions"].append(composition_info)
        
        return metadata
    
    def analyze_content_quality(self) -> Dict:
        """Analyze the quality and type of content in the repository"""
        try:
            logger.info("Analyzing content quality...")
            
            quality_analysis = {
                "content_types": {
                    "audio_files": 0,
                    "notation_files": 0,
                    "documentation": 0,
                    "code_files": 0,
                    "other": 0
                },
                "audio_formats": {},
                "notation_formats": {},
                "documentation_formats": {},
                "code_languages": {},
                "total_content_size": 0,
                "quality_indicators": {
                    "has_audio": False,
                    "has_notation": False,
                    "has_documentation": False,
                    "has_code": False,
                    "organized_structure": False
                }
            }
            
            repo_dir = self.output_dir / "Music"
            if not repo_dir.exists():
                return quality_analysis
            
            # Analyze all files
            for root, dirs, files in os.walk(repo_dir):
                for file in files:
                    file_path = Path(root) / file
                    file_ext = file_path.suffix.lower()
                    file_size = file_path.stat().st_size
                    
                    quality_analysis["total_content_size"] += file_size
                    
                    # Categorize files
                    if file_ext in ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']:
                        quality_analysis["content_types"]["audio_files"] += 1
                        quality_analysis["quality_indicators"]["has_audio"] = True
                        if file_ext in quality_analysis["audio_formats"]:
                            quality_analysis["audio_formats"][file_ext] += 1
                        else:
                            quality_analysis["audio_formats"][file_ext] = 1
                    
                    elif file_ext in ['.mid', '.midi', '.xml', '.json', '.yaml', '.yml']:
                        quality_analysis["content_types"]["notation_files"] += 1
                        quality_analysis["quality_indicators"]["has_notation"] = True
                        if file_ext in quality_analysis["notation_formats"]:
                            quality_analysis["notation_formats"][file_ext] += 1
                        else:
                            quality_analysis["notation_formats"][file_ext] = 1
                    
                    elif file_ext in ['.md', '.txt', '.pdf', '.doc', '.docx']:
                        quality_analysis["content_types"]["documentation"] += 1
                        quality_analysis["quality_indicators"]["has_documentation"] = True
                        if file_ext in quality_analysis["documentation_formats"]:
                            quality_analysis["documentation_formats"][file_ext] += 1
                        else:
                            quality_analysis["documentation_formats"][file_ext] = 1
                    
                    elif file_ext in ['.js', '.html', '.css', '.py', '.java', '.cpp', '.c']:
                        quality_analysis["content_types"]["code_files"] += 1
                        quality_analysis["quality_indicators"]["has_code"] = True
                        if file_ext in quality_analysis["code_languages"]:
                            quality_analysis["code_languages"][file_ext] += 1
                        else:
                            quality_analysis["code_languages"][file_ext] = 1
                    
                    else:
                        quality_analysis["content_types"]["other"] += 1
            
            # Check for organized structure
            if (repo_dir / "Carnatic").exists() and (repo_dir / "Hindustani").exists():
                quality_analysis["quality_indicators"]["organized_structure"] = True
            
            # Save quality analysis
            quality_file = self.output_dir / "content_quality_analysis.json"
            with open(quality_file, 'w') as f:
                json.dump(quality_analysis, f, indent=2)
            
            logger.info("✅ Content quality analysis completed")
            return quality_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content quality: {e}")
            return {}
    
    def collect(self) -> bool:
        """Main collection method"""
        try:
            logger.info("Starting ramanarunachalam/Music repository collection...")
            
            # Clone repository
            if not self.clone_repository():
                return False
            
            # Analyze structure
            structure = self.analyze_repository_structure()
            if not structure:
                return False
            
            # Extract metadata
            metadata = self.extract_music_metadata()
            if not metadata:
                return False
            
            # Analyze content quality
            quality = self.analyze_content_quality()
            if not quality:
                return False
            
            logger.info("✅ ramanarunachalam/Music collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in ramanarunachalam/Music collection process: {e}")
            return False

def main():
    """Main function"""
    collector = RamanarunachalamMusicCollector()
    success = collector.collect()
    
    if success:
        print("🎉 ramanarunachalam/Music collection completed successfully!")
    else:
        print("❌ ramanarunachalam/Music collection failed. Check logs for details.")

if __name__ == "__main__":
    main()
