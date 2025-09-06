#!/usr/bin/env python3
"""
Master Data Collector for RagaSense Dataset
Orchestrates collection from all data sources
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Import individual collectors
from 01_kaggle_collector import KaggleCarnaticCollector
from 02_saraga_collector import SaragaCollector
from 03_google_audioset_collector import GoogleAudioSetCollector
from 04_compmusic_varnam_collector import CompMusicVarnamCollector
from 05_sangeet_xml_collector import SangeetXMLCollector
from 06_ramanarunachalam_music_collector import RamanarunachalamMusicCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MasterDataCollector:
    """Master collector that orchestrates all data collection"""
    
    def __init__(self, base_output_dir: str = "01_raw_data"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize collectors
        self.collectors = {
            "kaggle_carnatic": KaggleCarnaticCollector(f"{base_output_dir}/kaggle_carnatic"),
            "saraga_mtg": SaragaCollector(f"{base_output_dir}/saraga_mtg"),
            "google_audioset": GoogleAudioSetCollector(f"{base_output_dir}/google_audioset"),
            "compmusic_varnam": CompMusicVarnamCollector(f"{base_output_dir}/compmusic_varnam"),
            "sangeet_xml": SangeetXMLCollector(f"{base_output_dir}/sangeet_xml"),
            "ramanarunachalam_music": RamanarunachalamMusicCollector(f"{base_output_dir}/ramanarunachalam_music")
        }
        
        self.collection_results = {}
        self.overall_statistics = {}
    
    def collect_all_sources(self, selected_sources: Optional[List[str]] = None) -> bool:
        """Collect data from all or selected sources"""
        try:
            logger.info("Starting master data collection...")
            
            # If no sources specified, collect from all
            if selected_sources is None:
                selected_sources = list(self.collectors.keys())
            
            logger.info(f"Collecting from sources: {selected_sources}")
            
            # Collect from each source
            for source_name in selected_sources:
                if source_name not in self.collectors:
                    logger.warning(f"Unknown source: {source_name}")
                    continue
                
                logger.info(f"Collecting from {source_name}...")
                start_time = time.time()
                
                try:
                    collector = self.collectors[source_name]
                    success = collector.collect()
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    self.collection_results[source_name] = {
                        "success": success,
                        "duration": duration,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if success:
                        logger.info(f"✅ {source_name} collection completed in {duration:.2f} seconds")
                    else:
                        logger.error(f"❌ {source_name} collection failed")
                        
                except Exception as e:
                    logger.error(f"Error collecting from {source_name}: {e}")
                    self.collection_results[source_name] = {
                        "success": False,
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
            
            # Generate overall statistics
            self.generate_overall_statistics()
            
            # Save collection report
            self.save_collection_report()
            
            # Check overall success
            successful_collections = sum(1 for result in self.collection_results.values() if result["success"])
            total_collections = len(self.collection_results)
            
            logger.info(f"Collection completed: {successful_collections}/{total_collections} sources successful")
            
            return successful_collections > 0
            
        except Exception as e:
            logger.error(f"Error in master collection process: {e}")
            return False
    
    def generate_overall_statistics(self) -> Dict:
        """Generate overall statistics from all collections"""
        try:
            logger.info("Generating overall statistics...")
            
            statistics = {
                "collection_summary": {
                    "total_sources": len(self.collectors),
                    "successful_collections": 0,
                    "failed_collections": 0,
                    "total_duration": 0
                },
                "data_sources": {},
                "content_summary": {
                    "total_audio_files": 0,
                    "total_metadata_files": 0,
                    "total_size_gb": 0,
                    "unique_ragas": set(),
                    "unique_artists": set(),
                    "traditions": set()
                }
            }
            
            # Process each collection result
            for source_name, result in self.collection_results.items():
                if result["success"]:
                    statistics["collection_summary"]["successful_collections"] += 1
                    statistics["collection_summary"]["total_duration"] += result.get("duration", 0)
                    
                    # Load source-specific statistics
                    source_dir = self.base_output_dir / source_name
                    if source_dir.exists():
                        source_stats = self._load_source_statistics(source_dir)
                        statistics["data_sources"][source_name] = source_stats
                        
                        # Aggregate content summary
                        if "content_summary" in source_stats:
                            content = source_stats["content_summary"]
                            statistics["content_summary"]["total_audio_files"] += content.get("audio_files", 0)
                            statistics["content_summary"]["total_metadata_files"] += content.get("metadata_files", 0)
                            statistics["content_summary"]["total_size_gb"] += content.get("size_gb", 0)
                            
                            if "ragas" in content:
                                statistics["content_summary"]["unique_ragas"].update(content["ragas"])
                            if "artists" in content:
                                statistics["content_summary"]["unique_artists"].update(content["artists"])
                            if "traditions" in content:
                                statistics["content_summary"]["traditions"].update(content["traditions"])
                else:
                    statistics["collection_summary"]["failed_collections"] += 1
            
            # Convert sets to lists for JSON serialization
            statistics["content_summary"]["unique_ragas"] = list(statistics["content_summary"]["unique_ragas"])
            statistics["content_summary"]["unique_artists"] = list(statistics["content_summary"]["unique_artists"])
            statistics["content_summary"]["traditions"] = list(statistics["content_summary"]["traditions"])
            
            self.overall_statistics = statistics
            
            logger.info("✅ Overall statistics generated")
            return statistics
            
        except Exception as e:
            logger.error(f"Error generating overall statistics: {e}")
            return {}
    
    def _load_source_statistics(self, source_dir: Path) -> Dict:
        """Load statistics from a specific source directory"""
        try:
            stats = {}
            
            # Look for structure analysis
            structure_file = source_dir / "dataset_structure.json"
            if structure_file.exists():
                with open(structure_file, 'r') as f:
                    structure = json.load(f)
                    stats["structure"] = structure
                    
                    # Extract content summary
                    if "audio_files" in structure:
                        stats["content_summary"] = {
                            "audio_files": len(structure["audio_files"]),
                            "metadata_files": len(structure.get("metadata_files", [])),
                            "size_gb": structure.get("total_size", 0) / (1024**3)
                        }
            
            # Look for extracted metadata
            metadata_file = source_dir / "extracted_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    stats["metadata"] = metadata
                    
                    # Extract ragas, artists, traditions
                    if "content_summary" not in stats:
                        stats["content_summary"] = {}
                    
                    if "ragas" in metadata:
                        stats["content_summary"]["ragas"] = metadata["ragas"]
                    if "artists" in metadata:
                        stats["content_summary"]["artists"] = metadata["artists"]
                    if "tradition" in metadata:
                        stats["content_summary"]["traditions"] = [metadata["tradition"]]
            
            return stats
            
        except Exception as e:
            logger.warning(f"Could not load statistics from {source_dir}: {e}")
            return {}
    
    def save_collection_report(self) -> bool:
        """Save comprehensive collection report"""
        try:
            logger.info("Saving collection report...")
            
            report = {
                "collection_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_sources": len(self.collectors),
                    "successful_collections": sum(1 for r in self.collection_results.values() if r["success"]),
                    "failed_collections": sum(1 for r in self.collection_results.values() if not r["success"])
                },
                "collection_results": self.collection_results,
                "overall_statistics": self.overall_statistics,
                "data_sources_info": {
                    "kaggle_carnatic": {
                        "description": "Kaggle Carnatic Song Database",
                        "url": "https://www.kaggle.com/datasets/sanjaynatesan/carnatic-song-database",
                        "type": "Kaggle Dataset"
                    },
                    "saraga_mtg": {
                        "description": "Saraga Dataset - Professional Indian Art Music",
                        "url": "https://mtg.github.io/saraga/",
                        "type": "MTG Dataset"
                    },
                    "google_audioset": {
                        "description": "Google AudioSet Carnatic Music",
                        "url": "https://research.google.com/audioset/dataset/carnatic_music.html",
                        "type": "Google Research Dataset"
                    },
                    "compmusic_varnam": {
                        "description": "CompMusic Carnatic Varnam Dataset",
                        "url": "https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi:10.34810/data457",
                        "type": "Academic Dataset"
                    },
                    "sangeet_xml": {
                        "description": "SANGEET XML-based Hindustani Dataset",
                        "url": "https://arxiv.org/abs/2306.04148",
                        "type": "Research Dataset"
                    },
                    "ramanarunachalam_music": {
                        "description": "Ramanarunachalam Music Repository",
                        "url": "https://github.com/ramanarunachalam/Music",
                        "type": "GitHub Repository"
                    }
                }
            }
            
            # Save report
            report_file = self.base_output_dir / "collection_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save summary report
            summary_file = self.base_output_dir / "collection_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("RagaSense Dataset Collection Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Collection Date: {report['collection_info']['timestamp']}\n")
                f.write(f"Total Sources: {report['collection_info']['total_sources']}\n")
                f.write(f"Successful: {report['collection_info']['successful_collections']}\n")
                f.write(f"Failed: {report['collection_info']['failed_collections']}\n\n")
                
                if "overall_statistics" in report and "content_summary" in report["overall_statistics"]:
                    content = report["overall_statistics"]["content_summary"]
                    f.write("Content Summary:\n")
                    f.write(f"- Total Audio Files: {content.get('total_audio_files', 0)}\n")
                    f.write(f"- Total Metadata Files: {content.get('total_metadata_files', 0)}\n")
                    f.write(f"- Total Size: {content.get('total_size_gb', 0):.2f} GB\n")
                    f.write(f"- Unique Ragas: {len(content.get('unique_ragas', []))}\n")
                    f.write(f"- Unique Artists: {len(content.get('unique_artists', []))}\n")
                    f.write(f"- Traditions: {', '.join(content.get('traditions', []))}\n\n")
                
                f.write("Collection Results:\n")
                for source, result in report["collection_results"].items():
                    status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
                    duration = f" ({result.get('duration', 0):.2f}s)" if result["success"] else ""
                    f.write(f"- {source}: {status}{duration}\n")
            
            logger.info("✅ Collection report saved")
            return True
            
        except Exception as e:
            logger.error(f"Error saving collection report: {e}")
            return False
    
    def print_summary(self):
        """Print collection summary to console"""
        print("\n" + "=" * 60)
        print("RagaSense Dataset Collection Summary")
        print("=" * 60)
        
        if not self.collection_results:
            print("No collection results available.")
            return
        
        successful = sum(1 for r in self.collection_results.values() if r["success"])
        total = len(self.collection_results)
        
        print(f"Collection Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Sources: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print()
        
        if self.overall_statistics and "content_summary" in self.overall_statistics:
            content = self.overall_statistics["content_summary"]
            print("Content Summary:")
            print(f"- Total Audio Files: {content.get('total_audio_files', 0)}")
            print(f"- Total Metadata Files: {content.get('total_metadata_files', 0)}")
            print(f"- Total Size: {content.get('total_size_gb', 0):.2f} GB")
            print(f"- Unique Ragas: {len(content.get('unique_ragas', []))}")
            print(f"- Unique Artists: {len(content.get('unique_artists', []))}")
            print(f"- Traditions: {', '.join(content.get('traditions', []))}")
            print()
        
        print("Collection Results:")
        for source, result in self.collection_results.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            duration = f" ({result.get('duration', 0):.2f}s)" if result["success"] else ""
            print(f"- {source}: {status}{duration}")
        
        print("=" * 60)

def main():
    """Main function"""
    print("🎵 RagaSense Dataset Collection System")
    print("=" * 50)
    
    # Initialize master collector
    collector = MasterDataCollector()
    
    # Collect from all sources
    success = collector.collect_all_sources()
    
    # Print summary
    collector.print_summary()
    
    if success:
        print("\n🎉 Dataset collection completed successfully!")
        print("Check the collection report for detailed results.")
    else:
        print("\n⚠️ Dataset collection completed with some failures.")
        print("Check the logs and collection report for details.")

if __name__ == "__main__":
    main()
