#!/usr/bin/env python3
"""
YouTube Audio Downloader and Processor for Raga Classification
Following Harvard thesis methodology for 10-second segments and feature extraction
"""

import os
import json
import yt_dlp
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import Counter
import logging
from typing import List, Dict, Tuple
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeRagaProcessor:
    """Process YouTube videos for raga classification training"""
    
    def __init__(self, dataset_path: str = "carnatic-hindustani-dataset", output_path: str = "youtube_audio"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_path / "raw_audio").mkdir(exist_ok=True)
        (self.output_path / "segments").mkdir(exist_ok=True)
        (self.output_path / "features").mkdir(exist_ok=True)
        
        # Audio processing parameters (following Harvard thesis)
        self.sample_rate = 22050
        self.segment_length = 10  # seconds
        self.segment_samples = self.sample_rate * self.segment_length
        
    def extract_youtube_links(self) -> List[Dict]:
        """Extract all YouTube links from the dataset"""
        youtube_data = []
        
        logger.info("Scanning dataset for YouTube links...")
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if 'youtube.com' in content or 'youtu.be' in content:
                                data = json.loads(content)
                                
                                # Extract raga information
                                raga_name = "Unknown"
                                if 'info' in data:
                                    for info_item in data['info']:
                                        if info_item.get('H') == 'Raga':
                                            raga_name = info_item.get('V', 'Unknown')
                                            break
                                
                                # Extract YouTube links from both lyricsref and songs sections
                                if 'lyricsref' in data:
                                    for link_obj in data['lyricsref']:
                                        if 'links' in link_obj:
                                            for link_item in link_obj['links']:
                                                if 'youtube.com/watch' in link_item.get('L', ''):
                                                    # Extract video ID from URL
                                                    url = link_item['L']
                                                    if 'v=' in url:
                                                        youtube_id = url.split('v=')[-1].split('&')[0]
                                                        youtube_data.append({
                                                            'youtube_id': youtube_id,
                                                            'raga_name': raga_name,
                                                            'url': url,
                                                            'source_file': str(file_path)
                                                        })
                                
                                # Also check songs section for YouTube IDs
                                if 'songs' in data:
                                    for song in data['songs']:
                                        if 'I' in song and song['I']:  # YouTube ID
                                            youtube_id = song['I']
                                            youtube_data.append({
                                                'youtube_id': youtube_id,
                                                'raga_name': raga_name,
                                                'duration': song.get('D', '0:00'),
                                                'views': song.get('V', '0'),
                                                'source_file': str(file_path)
                                            })
                    except Exception as e:
                        logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Found {len(youtube_data)} YouTube links across {len(set(item['raga_name'] for item in youtube_data))} ragas")
        return youtube_data
    
    def download_audio(self, youtube_id: str, raga_name: str, output_dir: Path) -> str:
        """Download audio from YouTube video"""
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        output_file = output_dir / f"{raga_name}_{youtube_id}.wav"
        
        # Skip if already downloaded
        if output_file.exists():
            logger.info(f"Audio already exists: {output_file}")
            return str(output_file)
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_file.with_suffix('.%(ext)s')),
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': '0',  # Best quality
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Rename to .wav if needed
            if not output_file.exists():
                # Find the downloaded file
                for ext in ['.webm', '.m4a', '.mp3']:
                    temp_file = output_file.with_suffix(ext)
                    if temp_file.exists():
                        temp_file.rename(output_file)
                        break
            
            logger.info(f"Downloaded: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to download {youtube_id}: {e}")
            return None
    
    def process_audio_segments(self, audio_file: str, raga_name: str, youtube_id: str) -> List[str]:
        """Process audio into 10-second segments following Harvard thesis methodology"""
        try:
            # Load audio with error handling
            try:
                y, sr = librosa.load(audio_file, sr=self.sample_rate)
            except Exception as e:
                logger.warning(f"Librosa failed, trying alternative: {e}")
                # Try with different backend
                y, sr = librosa.load(audio_file, sr=self.sample_rate, res_type='kaiser_fast')
            
            # Ensure we have enough audio
            if len(y) < self.segment_samples:
                logger.warning(f"Audio too short ({len(y)} samples), skipping")
                return []
            
            # Extract 10-second segments
            segments = []
            segment_files = []
            
            for i in range(0, len(y) - self.segment_samples, self.segment_samples):
                segment = y[i:i + self.segment_samples]
                
                # Save segment
                segment_file = self.output_path / "segments" / f"{raga_name}_{youtube_id}_seg_{i//self.segment_samples}.wav"
                sf.write(segment_file, segment, sr)
                segment_files.append(str(segment_file))
                segments.append(segment)
            
            logger.info(f"Created {len(segments)} segments from {raga_name}_{youtube_id}")
            return segment_files
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_file}: {e}")
            return []
    
    def extract_features(self, audio_file: str) -> np.ndarray:
        """Extract 50 numerical features following Harvard thesis methodology"""
        try:
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            features = []
            
            # 1. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.median(spectral_centroids)
            ])
            
            # 2. Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.extend([
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff)
            ])
            
            # 3. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])
            
            # 4. MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features.extend([
                    np.mean(mfccs[i]),
                    np.std(mfccs[i])
                ])
            
            # 5. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features.extend([
                    np.mean(chroma[i]),
                    np.std(chroma[i])
                ])
            
            # 6. Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            for i in range(6):
                features.extend([
                    np.mean(tonnetz[i]),
                    np.std(tonnetz[i])
                ])
            
            # 7. Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(7):
                features.extend([
                    np.mean(contrast[i]),
                    np.std(contrast[i])
                ])
            
            # Ensure we have exactly 50 features
            features = features[:50]
            if len(features) < 50:
                features.extend([0.0] * (50 - len(features)))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_file}: {e}")
            return np.zeros(50)
    
    def process_youtube_dataset(self, max_videos: int = None) -> Dict:
        """Process the entire YouTube dataset"""
        logger.info("Starting YouTube dataset processing...")
        
        # Extract YouTube links
        youtube_data = self.extract_youtube_links()
        
        if max_videos:
            youtube_data = youtube_data[:max_videos]
        
        # Statistics
        stats = {
            'total_videos': len(youtube_data),
            'successful_downloads': 0,
            'successful_segments': 0,
            'total_segments': 0,
            'raga_counts': Counter(),
            'failed_downloads': []
        }
        
        # Process each video
        for i, video_data in enumerate(youtube_data):
            youtube_id = video_data['youtube_id']
            raga_name = video_data['raga_name']
            
            logger.info(f"Processing {i+1}/{len(youtube_data)}: {raga_name} - {youtube_id}")
            
            # Download audio
            audio_file = self.download_audio(
                youtube_id, 
                raga_name, 
                self.output_path / "raw_audio"
            )
            
            if audio_file:
                stats['successful_downloads'] += 1
                stats['raga_counts'][raga_name] += 1
                
                # Process segments
                segment_files = self.process_audio_segments(audio_file, raga_name, youtube_id)
                stats['successful_segments'] += len(segment_files)
                stats['total_segments'] += len(segment_files)
                
                # Extract features for each segment
                for segment_file in segment_files:
                    features = self.extract_features(segment_file)
                    feature_file = Path(segment_file).with_suffix('.npy')
                    np.save(feature_file, features)
                
                logger.info(f"Processed {len(segment_files)} segments")
            else:
                stats['failed_downloads'].append(youtube_id)
            
            # Small delay to be respectful to YouTube
            time.sleep(1)
        
        # Save statistics
        stats_file = self.output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'total_videos': stats['total_videos'],
                'successful_downloads': stats['successful_downloads'],
                'successful_segments': stats['successful_segments'],
                'total_segments': stats['total_segments'],
                'raga_counts': dict(stats['raga_counts']),
                'failed_downloads': stats['failed_downloads']
            }, f, indent=2)
        
        logger.info(f"Processing complete!")
        logger.info(f"Total videos: {stats['total_videos']}")
        logger.info(f"Successful downloads: {stats['successful_downloads']}")
        logger.info(f"Total segments: {stats['total_segments']}")
        logger.info(f"Ragas processed: {len(stats['raga_counts'])}")
        
        return stats

def main():
    """Main function to run the YouTube processor"""
    processor = YouTubeRagaProcessor()
    
    # Process a small subset first for testing
    logger.info("Processing YouTube dataset (first 10 videos for testing)...")
    stats = processor.process_youtube_dataset(max_videos=10)
    
    print("\n" + "="*50)
    print("YOUTUBE PROCESSING RESULTS")
    print("="*50)
    print(f"Total videos processed: {stats['total_videos']}")
    print(f"Successful downloads: {stats['successful_downloads']}")
    print(f"Total segments created: {stats['total_segments']}")
    print(f"Ragas processed: {len(stats['raga_counts'])}")
    print("\nTop ragas:")
    for raga, count in stats['raga_counts'].most_common(5):
        print(f"  {raga}: {count} videos")
    
    if stats['failed_downloads']:
        print(f"\nFailed downloads: {len(stats['failed_downloads'])}")
        print("Failed IDs:", stats['failed_downloads'][:5])

if __name__ == "__main__":
    main()
