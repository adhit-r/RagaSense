# RagaSense Dataset Collection System

## 🎯 **Overview**

This system collects and integrates data from multiple sources to create a comprehensive dataset for Indian classical music (Carnatic and Hindustani) research and development.

## 📊 **Data Sources**

### **1. Kaggle Carnatic Song Database**
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/sanjaynatesan/carnatic-song-database)
- **Content**: Carnatic song database
- **Access**: Kaggle API required

### **2. Saraga Dataset (MTG)**
- **Source**: [MTG Saraga](https://mtg.github.io/saraga/)
- **Content**: Professional Indian art music with annotations
- **Access**: Open source

### **3. Google AudioSet - Carnatic Music**
- **Source**: [Google AudioSet](https://research.google.com/audioset/dataset/carnatic_music.html)
- **Content**: 1,663 videos, 4.6 hours of Carnatic music
- **Access**: Public dataset

### **4. SANGEET Dataset (Hindustani)**
- **Source**: [SANGEET Paper](https://arxiv.org/abs/2306.04148)
- **Content**: XML-based Hindustani compositions
- **Access**: Research dataset

### **5. CompMusic Varnam Dataset**
- **Source**: [CompMusic Dataverse](https://dataverse.csuc.cat/dataset.xhtml?persistentId=doi:10.34810/data457)
- **Content**: 7 varnams in 7 ragas, professional singers
- **Access**: Academic dataset

### **6. Ramanarunachalam Music Repository**
- **Source**: [GitHub Repository](https://github.com/ramanarunachalam/Music)
- **Content**: Carnatic and Hindustani music files
- **Access**: Public GitHub repository

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Create virtual environment
python -m venv ragasense_env
source ragasense_env/bin/activate  # On Windows: ragasense_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Kaggle API (for Kaggle dataset)
# Download kaggle.json from your Kaggle account
# Place it in ~/.kaggle/kaggle.json
```

### **2. Run Data Collection**
```bash
# Collect from all sources
python 04_scripts/data_collection/master_collector.py

# Or collect from specific sources
python 04_scripts/data_collection/01_kaggle_collector.py
python 04_scripts/data_collection/02_saraga_collector.py
python 04_scripts/data_collection/03_google_audioset_collector.py
python 04_scripts/data_collection/04_compmusic_varnam_collector.py
python 04_scripts/data_collection/05_sangeet_xml_collector.py
python 04_scripts/data_collection/06_ramanarunachalam_music_collector.py
```

### **3. Check Results**
```bash
# View collection report
cat 01_raw_data/collection_report.json

# View summary
cat 01_raw_data/collection_summary.txt
```

## 📁 **Directory Structure**

```
data/ragasense_dataset_collection/
├── 01_raw_data/                    # Raw collected data
│   ├── kaggle_carnatic/           # Kaggle dataset
│   ├── saraga_mtg/                # Saraga dataset
│   ├── local_carnatic_hindustani/ # Local dataset
│   ├── google_audioset/           # Google AudioSet
│   ├── sangeet_xml/               # SANGEET XML dataset
│   ├── compmusic_varnam/          # CompMusic varnam dataset
│   ├── ramanarunachalam_music/    # GitHub repository
│   ├── collection_report.json     # Collection report
│   └── collection_summary.txt     # Summary report
├── 02_processed_data/             # Processed data
├── 03_unified_dataset/            # Final unified dataset
├── 04_scripts/                    # Collection scripts
├── 05_analysis/                   # Analysis results
├── 06_publication/                # Publication materials
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 **Individual Collectors**

### **Kaggle Collector**
- Downloads Carnatic song database from Kaggle
- Requires Kaggle API key
- Extracts metadata and audio files

### **Saraga Collector**
- Processes existing Saraga dataset
- Analyzes Carnatic and Hindustani sections
- Extracts annotations and metadata

### **Google AudioSet Collector**
- Downloads YouTube videos from AudioSet
- Extracts audio using yt-dlp
- Processes metadata

### **CompMusic Varnam Collector**
- Downloads academic varnam dataset
- Processes professional recordings
- Extracts taala annotations

### **SANGEET XML Collector**
- Parses XML-based Hindustani compositions
- Extracts structural and melodic information
- Processes metadata

### **Ramanarunachalam Music Collector**
- Clones GitHub repository
- Analyzes file structure
- Extracts music metadata

## 📊 **Expected Output**

### **Dataset Statistics**
- **Total Duration**: 100+ hours
- **Carnatic**: 60+ hours
- **Hindustani**: 40+ hours
- **Unique Ragas**: 200+
- **Unique Artists**: 100+

### **File Formats**
- **Audio**: WAV, MP3, FLAC
- **Metadata**: JSON, XML, CSV
- **Annotations**: JSON, YAML
- **Notations**: XML, MIDI

## ⚠️ **Requirements**

### **System Requirements**
- **Python**: 3.8+
- **Storage**: 100GB+ free space
- **RAM**: 8GB+ recommended
- **Internet**: Stable connection for downloads

### **API Keys Required**
- **Kaggle**: For Kaggle dataset access
- **YouTube**: For AudioSet video downloads (optional)

### **Dependencies**
- See `requirements.txt` for full list
- Key packages: pandas, librosa, requests, kaggle, yt-dlp

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Kaggle API Error**
```bash
# Ensure kaggle.json is in correct location
ls ~/.kaggle/kaggle.json

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

#### **YouTube Download Issues**
```bash
# Update yt-dlp
pip install --upgrade yt-dlp

# Check for blocked videos
# Some videos may be region-restricted
```

#### **Memory Issues**
```bash
# Reduce batch sizes in collectors
# Process smaller chunks of data
# Use streaming for large files
```

### **Logs and Debugging**
- Check individual collector logs
- Review `collection_report.json` for detailed results
- Use `--verbose` flag for detailed output

## 📈 **Performance**

### **Collection Times**
- **Kaggle**: 5-10 minutes
- **Saraga**: 2-5 minutes
- **Google AudioSet**: 30-60 minutes (depends on video count)
- **CompMusic**: 10-20 minutes
- **SANGEET**: 5-10 minutes
- **Ramanarunachalam**: 2-5 minutes

### **Storage Usage**
- **Raw Data**: ~50GB
- **Processed Data**: ~30GB
- **Final Dataset**: ~20GB

## 🔄 **Updates and Maintenance**

### **Regular Updates**
- Check for new dataset versions
- Update collectors for API changes
- Refresh metadata and annotations

### **Quality Control**
- Validate audio file integrity
- Check metadata completeness
- Verify raga classifications

## 📚 **Documentation**

- **Integration Analysis**: `DATASET_INTEGRATION_ANALYSIS.md`
- **Collection Scripts**: `04_scripts/data_collection/`
- **Individual Reports**: `01_raw_data/*/`

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add new data sources or collectors
4. Test thoroughly
5. Submit pull request

## 📄 **License**

This dataset collection system is part of the RagaSense project. Please refer to the main project license for usage terms.

## 📞 **Support**

For issues and questions:
- Check the troubleshooting section
- Review collection logs
- Open an issue in the main repository

---

**Created by**: Adhithya Rajasekaran (@adhit-r)  
**Date**: September 6, 2025  
**Version**: 1.0
