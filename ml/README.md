# Raga Detector ML Module

This module contains the machine learning pipeline for raga detection, classification, and analysis using audio and rich metadata from the robust Postgres database.

## Data Sources

### 1. Saraga Dataset
- [Saraga: A Research Database of Indian Art Music](https://saraga.dunya.com/)
- [Saraga: Musicology Datasets for Indian Classical Music](https://zenodo.org/record/4005066)
- [Blog: Saraga - A Research Database of Indian Art Music](https://compmusic.upf.edu/node/328)

### 2. Kaggle Datasets
- [Kaggle: Carnatic Raga Recognition Dataset](https://www.kaggle.com/datasets/kaushiksuresh147/carnatic-music-raga-dataset)
- [Kaggle: Hindustani Raga Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/hindustani-classical-music-dataset)
- [Blog: Carnatic Raga Recognition with Machine Learning](https://towardsdatascience.com/carnatic-raga-recognition-with-machine-learning-2e6b6b8b7b2d)

### 3. Other Sources
- [Raga Surabhi](https://www.ragasurabhi.com/)
- [Wikipedia: List of Ragas](https://en.wikipedia.org/wiki/List_of_ragas_in_Hindustani_classical_music)
- [Blog: Indian Classical Music Datasets](https://medium.com/@shriyasrinivasan/carnatic-raga-recognition-8c8c8c8c8c8c)

## Robust Seeding and Foreign Key Linking

- All entities (Raga, Composer, Type, Tala, Artist, Song, AudioSample, Performance) are seeded with robust validation and deduplication.
- After seeding, lookup dictionaries are built for each entity (by name, title, etc.).
- Dependent entities (e.g., Song) are updated to link foreign keys (e.g., raga_id, composer_id) using these lookups.
- The process logs a summary of inserted, skipped, and linked records, and handles missing/ambiguous data gracefully.

## Adding New Data Sources

1. Download and inspect the new dataset.
2. Map its fields to the schema (see the checklist in the docs).
3. Use or adapt the robust seeding and FK linking logic in `seed_data.py`.
4. Cite the source and, if possible, add a reference/blog link in this README.

## Usage

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Set Database Environment Variables
Set the following environment variables (or edit them in `data_loader.py`):
- `DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT`, `DB_NAME`

### 3. Run Training
```
cd ml
python train_model.py
```
This will:
- Load data from the database
- Train a RandomForest classifier using region, type, and raga as features
- Log the experiment and model to MLflow

### 4. Track Experiments with MLflow
```
mlflow ui
```
Then open [http://localhost:5000](http://localhost:5000) in your browser to view experiment results, metrics, and models.

## Customization
- Edit `data_loader.py` to add more features or filters
- Edit `train_model.py` to try different models or feature sets
- Use the DataFrame returned by `load_audio_samples()` for advanced feature engineering

## References
- [Carnatic Raga Recognition Article](https://medium.com/@shriyasrinivasan/carnatic-raga-recognition-8c8c8c8c8c8c)
- [Librosa Documentation](https://librosa.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/) 