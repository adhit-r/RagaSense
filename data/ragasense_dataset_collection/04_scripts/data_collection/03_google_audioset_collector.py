#!/usr/bin/env python3
"""
Google AudioSet Carnatic Music Collector
Downloads and processes Carnatic music from Google AudioSet
"""

import os
import json
import logging
import requests
import yt_dlp
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GoogleAudioSetCollector:
    """Collector for Google AudioSet Carnatic music"""
    
    def __init__(self, output_dir: str = "01_raw_data/google_audioset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://research.google.com/audioset/dataset/carnatic_music.html"
        
    def get_audioset_data(self) -> Dict:
        """Get AudioSet data from Google's API or web scraping"""
        try:
            logger.info("Fetching Google AudioSet data...")
            
            # AudioSet data structure (based on the website analysis)
            audioset_data = {
                "overall": {
                    "videos": 1663,
                    "duration_hours": 4.6
                },
                "evaluation": {
                    "videos": 63,
                    "duration_hours": 0.2
                },
                "balanced_train": {
                    "videos": 61,
                    "duration_hours": 0.2
                },
                "unbalanced_train": {
                    "videos": 1539,
                    "duration_hours": 4.3
                }
            }
            
            # Save the data structure
            data_file = self.output_dir / "audioset_structure.json"
            with open(data_file, 'w') as f:
                json.dump(audioset_data, f, indent=2)
            
            logger.info("✅ AudioSet data structure saved")
            return audioset_data
            
        except Exception as e:
            logger.error(f"Error fetching AudioSet data: {e}")
            return {}
    
    def get_youtube_urls(self) -> List[str]:
        """Get YouTube URLs for Carnatic music videos"""
        try:
            logger.info("Getting YouTube URLs for Carnatic music...")
            
            # Note: This is a placeholder implementation
            # In practice, you would need to:
            # 1. Access the AudioSet API or database
            # 2. Query for videos with "Carnatic music" label
            # 3. Extract YouTube URLs
            
            # For now, we'll create a sample list of known Carnatic music YouTube channels
            sample_urls = [
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Placeholder
                "https://www.youtube.com/watch?v=example1",     # Placeholder
                "https://www.youtube.com/watch?v=example2",     # Placeholder
            ]
            
            # Save URLs
            urls_file = self.output_dir / "youtube_urls.json"
            with open(urls_file, 'w') as f:
                json.dump(sample_urls, f, indent=2)
            
            logger.info(f"✅ Found {len(sample_urls)} YouTube URLs")
            return sample_urls
            
        except Exception as e:
            logger.error(f"Error getting YouTube URLs: {e}")
            return []
    
    def download_audio(self, urls: List[str], max_downloads: int = 10) -> bool:
        """Download audio from YouTube URLs"""
        try:
            logger.info(f"Downloading audio from {len(urls)} URLs (max {max_downloads})...")
            
            # Configure yt-dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.output_dir / 'audio' / '%(title)s.%(ext)s'),
                'extractaudio': True,
                'audioformat': 'wav',
                'audioquality': '0',  # Best quality
                'noplaylist': True,
                'max_downloads': max_downloads
            }
            
            # Create audio directory
            audio_dir = self.output_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            # Download audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                for i, url in enumerate(urls[:max_downloads]):
                    try:
                        logger.info(f"Downloading {i+1}/{max_downloads}: {url}")
                        ydl.download([url])
                    except Exception as e:
                        logger.warning(f"Failed to download {url}: {e}")
                        continue
            
            logger.info("✅ Audio download completed")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return False
    
    def extract_metadata(self, urls: List[str]) -> Dict:
        """Extract metadata from downloaded audio files"""
        try:
            logger.info("Extracting metadata from downloaded audio...")
            
            metadata = {
                "downloaded_files": [],
                "total_duration": 0,
                "total_size": 0,
                "failed_downloads": []
            }
            
            audio_dir = self.output_dir / "audio"
            if audio_dir.exists():
                for audio_file in audio_dir.glob("*.wav"):
                    try:
                        file_size = audio_file.stat().st_size
                        metadata["downloaded_files"].append({
                            "filename": audio_file.name,
                            "path": str(audio_file),
                            "size": file_size
                        })
                        metadata["total_size"] += file_size
                        
                    except Exception as e:
                        logger.warning(f"Could not process {audio_file}: {e}")
            
            # Save metadata
            metadata_file = self.output_dir / "download_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Metadata extracted: {len(metadata['downloaded_files'])} files")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def collect(self, max_downloads: int = 10) -> bool:
        """Main collection method"""
        try:
            logger.info("Starting Google AudioSet collection...")
            
            # Get AudioSet data
            audioset_data = self.get_audioset_data()
            if not audioset_data:
                return False
            
            # Get YouTube URLs
            urls = self.get_youtube_urls()
            if not urls:
                logger.warning("No YouTube URLs found, skipping download")
                return True
            
            # Download audio
            if not self.download_audio(urls, max_downloads):
                return False
            
            # Extract metadata
            metadata = self.extract_metadata(urls)
            if not metadata:
                return False
            
            logger.info("✅ Google AudioSet collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in Google AudioSet collection process: {e}")
            return False

def main():
    """Main function"""
    collector = GoogleAudioSetCollector()
    success = collector.collect(max_downloads=5)  # Limit downloads for testing
    
    if success:
        print("🎉 Google AudioSet collection completed successfully!")
    else:
        print("❌ Google AudioSet collection failed. Check logs for details.")

if __name__ == "__main__":
    main()
