#!/bin/bash

# YouTube Audio Processing Script
# Following Harvard thesis methodology for raga classification

echo "🎵 Starting YouTube Audio Processing for Raga Classification"
echo "=========================================================="

# Activate virtual environment
source ../raga_env/bin/activate

# Create output directory
mkdir -p youtube_audio/{raw_audio,segments,features}

# Run the YouTube processor
echo "📥 Processing YouTube videos..."
python youtube_processor.py

echo "✅ YouTube processing complete!"
echo ""
echo "📊 Results:"
echo "- Raw audio files: youtube_audio/raw_audio/"
echo "- 10-second segments: youtube_audio/segments/"
echo "- Extracted features: youtube_audio/features/"
echo "- Processing stats: youtube_audio/processing_stats.json"
echo ""
echo "🚀 Next steps:"
echo "1. Review processing statistics"
echo "2. Integrate with existing dataset"
echo "3. Train enhanced model"
echo "4. Evaluate performance improvements"
