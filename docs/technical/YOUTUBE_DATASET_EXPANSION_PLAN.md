# YOUTUBE DATASET EXPANSION PLAN

## Current YouTube Dataset Analysis

### Overview
- **Files with YouTube links**: 65
- **Total YouTube links**: 1,192
- **Unique ragas with YouTube links**: 49
- **Average links per raga**: ~24

### Top Ragas with YouTube Links
| Rank | Raga Name | YouTube Links |
|------|-----------|---------------|
| 1 | Kapi | 3 |
| 2 | Ragamalika | 3 |
| 3 | Kalyani | 3 |
| 4 | Unknownraga | 3 |
| 5 | Anandabhairavi | 2 |
| 6 | Sahana | 2 |
| 7 | Vandanadharini | 2 |
| 8 | Abheri | 1 |
| 9 | Huseni | 1 |
| 10 | Vakulabharanam | 1 |

## YouTube Data Structure

### JSON Structure
```json
{
  "songs": [
    {
      "I": "y6bkIE5Y9gQ",  // YouTube ID
      "T": 0,              // Type
      "S": 3991,           // Song ID
      "R": 17,             // Raga ID
      "C": 11,             // Composer ID
      "A": 215,            // Artist ID
      "D": "6:45",         // Duration
      "V": "56",           // Views
      "J": "m"             // Quality
    }
  ],
  "info": [
    {
      "H": "Raga",
      "V": "Anandabhairavi",  // Raga name
      "P": "AnaMdhabhairavi",
      "I": 17
    }
  ]
}
```

## Implementation Strategy

### Phase 1: YouTube Audio Extraction
1. **Download YouTube Audio**
   - Use `yt-dlp` or `youtube-dl` to extract audio
   - Convert to WAV format (44.1kHz, 16-bit)
   - Extract 10-second segments (following Harvard thesis methodology)

2. **Audio Processing Pipeline**
   - Extract mel-spectrograms
   - Extract 50 numerical features
   - Apply data augmentation (9x increase)

### Phase 2: Dataset Integration
1. **Merge with Existing Dataset**
   - Combine with current 1,459 ragas
   - Maintain raga labels from JSON metadata
   - Ensure balanced representation

2. **Quality Control**
   - Audio quality filtering
   - Raga label validation
   - Duplicate detection

### Phase 3: Training Enhancement
1. **Expanded Training Set**
   - Current: 6,182 files
   - With YouTube: 6,182 + 1,192 = 7,374 files
   - With augmentation: 7,374 × 9 = 66,366 files

2. **Improved Model Performance**
   - More diverse audio sources
   - Better raga coverage
   - Enhanced generalization

## Technical Implementation

### Required Tools
```bash
# Install YouTube downloader
pip install yt-dlp

# Install audio processing
pip install librosa soundfile

# Install ML tools
pip install torch torchvision torchaudio
```

### Download Script
```python
import yt_dlp
import os
import json
from pathlib import Path

def download_youtube_audio(youtube_id, output_dir, raga_name):
    """Download audio from YouTube video"""
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/{raga_name}_{youtube_id}.%(ext)s',
        'extractaudio': True,
        'audioformat': 'wav',
        'audioquality': '0',  # Best quality
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
```

### Audio Processing Script
```python
import librosa
import numpy as np
from pathlib import Path

def process_youtube_audio(audio_file, raga_name, output_dir):
    """Process YouTube audio into training format"""
    # Load audio
    y, sr = librosa.load(audio_file, sr=22050)
    
    # Extract 10-second segments
    segment_length = 10 * sr  # 10 seconds
    segments = []
    
    for i in range(0, len(y) - segment_length, segment_length):
        segment = y[i:i + segment_length]
        segments.append(segment)
    
    # Save segments
    for i, segment in enumerate(segments):
        output_file = output_dir / f"{raga_name}_{i}.wav"
        librosa.output.write_wav(output_file, segment, sr)
```

## Benefits of YouTube Integration

### 1. Massive Dataset Expansion
- **Current**: 6,182 files
- **With YouTube**: 7,374 files (+19% increase)
- **With augmentation**: 66,366 files (10x increase)

### 2. Diverse Audio Sources
- Different recording qualities
- Various performance styles
- Multiple artists and interpretations
- Real-world audio conditions

### 3. Better Raga Coverage
- 49 additional ragas with YouTube links
- More balanced dataset
- Rare raga representation

### 4. Improved Model Robustness
- Better generalization to real-world audio
- Enhanced noise resistance
- More diverse feature learning

## Implementation Timeline

### Week 1: Setup and Download
- [ ] Install required tools
- [ ] Create download pipeline
- [ ] Download 1,192 YouTube videos
- [ ] Quality control and filtering

### Week 2: Processing and Integration
- [ ] Extract 10-second segments
- [ ] Generate mel-spectrograms
- [ ] Extract numerical features
- [ ] Apply data augmentation

### Week 3: Training and Evaluation
- [ ] Integrate with existing dataset
- [ ] Train enhanced model
- [ ] Evaluate performance improvements
- [ ] Compare with baseline

## Expected Results

### Performance Improvements
- **Accuracy**: 93.6% → 95%+ (following Harvard thesis)
- **Robustness**: Better real-world performance
- **Coverage**: More ragas recognized
- **Generalization**: Better unseen data performance

### Dataset Statistics
- **Total files**: 66,366 (with augmentation)
- **Unique ragas**: 1,459+
- **Training time**: ~2-3 hours (with GPU)
- **Model size**: ~50MB

## Next Steps

1. **Implement YouTube downloader**
2. **Create audio processing pipeline**
3. **Integrate with existing training system**
4. **Train enhanced model**
5. **Evaluate performance improvements**

This YouTube integration will significantly enhance our raga classification system and bring us closer to the 93.6% accuracy achieved in the Harvard thesis!
