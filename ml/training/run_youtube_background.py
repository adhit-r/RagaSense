#!/usr/bin/env python3
"""
Background YouTube Processing Script
Runs YouTube audio processing in the background with progress tracking
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path
from youtube_processor import YouTubeRagaProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackgroundProcessor:
    """Background YouTube processor with progress tracking"""
    
    def __init__(self):
        self.processor = YouTubeRagaProcessor()
        self.running = True
        self.stats = {
            'start_time': time.time(),
            'processed_videos': 0,
            'total_videos': 0,
            'successful_downloads': 0,
            'total_segments': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def save_progress(self):
        """Save current progress to file"""
        progress_file = Path("youtube_processing_progress.json")
        import json
        
        progress_data = {
            'stats': self.stats,
            'running': self.running,
            'timestamp': time.time()
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_progress(self):
        """Load previous progress if exists"""
        progress_file = Path("youtube_processing_progress.json")
        if progress_file.exists():
            import json
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.stats.update(progress_data.get('stats', {}))
                    logger.info(f"Loaded previous progress: {self.stats['processed_videos']} videos processed")
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
    
    def process_batch(self, batch_size=50, max_videos=None):
        """Process videos in batches"""
        logger.info("Starting background YouTube processing...")
        
        # Load previous progress
        self.load_progress()
        
        # Extract YouTube links
        youtube_data = self.processor.extract_youtube_links()
        self.stats['total_videos'] = len(youtube_data)
        
        if max_videos:
            youtube_data = youtube_data[:max_videos]
            self.stats['total_videos'] = min(len(youtube_data), max_videos)
        
        logger.info(f"Found {len(youtube_data)} YouTube videos to process")
        
        # Process in batches
        for i in range(0, len(youtube_data), batch_size):
            if not self.running:
                logger.info("Stopping processing due to shutdown signal")
                break
                
            batch = youtube_data[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: videos {i+1}-{min(i+batch_size, len(youtube_data))}")
            
            for j, video_data in enumerate(batch):
                if not self.running:
                    break
                    
                youtube_id = video_data['youtube_id']
                raga_name = video_data['raga_name']
                
                logger.info(f"Processing {self.stats['processed_videos']+1}/{self.stats['total_videos']}: {raga_name} - {youtube_id}")
                
                try:
                    # Download audio
                    audio_file = self.processor.download_audio(
                        youtube_id, 
                        raga_name, 
                        self.processor.output_path / "raw_audio"
                    )
                    
                    if audio_file:
                        self.stats['successful_downloads'] += 1
                        
                        # Process segments
                        segment_files = self.processor.process_audio_segments(audio_file, raga_name, youtube_id)
                        self.stats['total_segments'] += len(segment_files)
                        
                        # Extract features for each segment
                        for segment_file in segment_files:
                            features = self.processor.extract_features(segment_file)
                            feature_file = Path(segment_file).with_suffix('.npy')
                            import numpy as np
                            np.save(feature_file, features)
                        
                        logger.info(f"Processed {len(segment_files)} segments")
                    else:
                        logger.warning(f"Failed to download {youtube_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing {youtube_id}: {e}")
                
                self.stats['processed_videos'] += 1
                
                # Save progress every 10 videos
                if self.stats['processed_videos'] % 10 == 0:
                    self.save_progress()
                    self.print_progress()
                
                # Small delay to be respectful to YouTube
                time.sleep(1)
            
            # Save progress after each batch
            self.save_progress()
            self.print_progress()
            
            # Longer delay between batches
            if i + batch_size < len(youtube_data):
                logger.info("Waiting 30 seconds before next batch...")
                time.sleep(30)
        
        # Final save
        self.save_progress()
        self.print_final_results()
    
    def print_progress(self):
        """Print current progress"""
        elapsed = time.time() - self.stats['start_time']
        rate = self.stats['processed_videos'] / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"YOUTUBE PROCESSING PROGRESS")
        print(f"{'='*60}")
        print(f"Processed: {self.stats['processed_videos']}/{self.stats['total_videos']} videos")
        print(f"Successful downloads: {self.stats['successful_downloads']}")
        print(f"Total segments: {self.stats['total_segments']}")
        print(f"Processing rate: {rate:.2f} videos/hour")
        print(f"Elapsed time: {elapsed/3600:.2f} hours")
        print(f"{'='*60}\n")
    
    def print_final_results(self):
        """Print final results"""
        elapsed = time.time() - self.stats['start_time']
        
        print(f"\n{'='*60}")
        print(f"YOUTUBE PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Total videos processed: {self.stats['processed_videos']}")
        print(f"Successful downloads: {self.stats['successful_downloads']}")
        print(f"Total segments created: {self.stats['total_segments']}")
        success_rate = (self.stats['successful_downloads']/self.stats['processed_videos']*100) if self.stats['processed_videos'] > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"Average rate: {self.stats['processed_videos']/elapsed*3600:.2f} videos/hour")
        print(f"{'='*60}\n")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Background YouTube Processing')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--max-videos', type=int, default=None, help='Maximum videos to process')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    
    args = parser.parse_args()
    
    processor = BackgroundProcessor()
    
    try:
        processor.process_batch(
            batch_size=args.batch_size,
            max_videos=args.max_videos
        )
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        processor.save_progress()

if __name__ == "__main__":
    main()
