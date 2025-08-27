import os
import pandas as pd
from sqlalchemy import create_engine, text
import librosa
import numpy as np
import random

DB_USER = os.getenv('DB_USER', 'raga_user')
DB_PASS = os.getenv('DB_PASS', 'raga_pass')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'ragasense_db')

DB_URL = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

def load_audio_samples(region=None, raga=None, type_=None, min_duration=None):
    """
    Load audio samples and all relevant metadata from the database as a DataFrame.
    Optionally filter by region, raga, or type.
    """
    engine = create_engine(DB_URL)
    query = '''
    SELECT
        a.id AS audio_id,
        a.file_path,
        a.type AS audio_type,
        a.region AS audio_region,
        a.raga_id,
        r.name AS raga_name,
        r.region AS raga_region,
        r.melakarta_number,
        r.arohana,
        r.avarohana,
        s.id AS song_id,
        s.title AS song_title,
        s.region AS song_region,
        t.id AS type_id,
        t.name AS type_name,
        t.region AS type_region,
        ta.id AS tala_id,
        ta.name AS tala_name,
        ta.region AS tala_region,
        c.id AS composer_id,
        c.name AS composer_name,
        c.region AS composer_region,
        ar.id AS artist_id,
        ar.name AS artist_name,
        ar.region AS artist_region
    FROM audio_samples a
    LEFT JOIN ragas r ON a.raga_id = r.id
    LEFT JOIN songs s ON a.song_id = s.id
    LEFT JOIN types t ON s.type_id = t.id
    LEFT JOIN talas ta ON s.tala_id = ta.id
    LEFT JOIN composers c ON s.composer_id = c.id
    LEFT JOIN artists ar ON a.artist_id = ar.id
    WHERE 1=1
    '''
    params = {}
    if region:
        query += ' AND (a.region = :region OR r.region = :region OR s.region = :region)'
        params['region'] = region
    if raga:
        query += ' AND r.name = :raga'
        params['raga'] = raga
    if type_:
        query += ' AND t.name = :type_'
        params['type_'] = type_
    if min_duration:
        query += ' AND a.duration >= :min_duration'
        params['min_duration'] = min_duration
    df = pd.read_sql(text(query), engine, params=params)
    return df

# --- Audio Feature Extraction ---
def augment_audio(y, sr):
    """Apply random augmentation(s) to an audio signal."""
    # Randomly choose which augmentations to apply
    if random.random() < 0.5:
        # Pitch shift by -2 to +2 semitones
        n_steps = random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr, n_steps)
    if random.random() < 0.5:
        # Time stretch between 0.8x and 1.2x
        rate = random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate)
    if random.random() < 0.5:
        # Add Gaussian noise
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    return y

def extract_audio_features(file_path, sr=22050, n_mfcc=13, augment=False):
    """
    Extract MFCCs, chroma, and STFT features from an audio file.
    Optionally apply augmentation.
    Returns a 1D numpy array of concatenated features.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)
        if augment:
            y = augment_audio(y, sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        stft = np.abs(librosa.stft(y))
        stft_mean = stft.mean(axis=1)
        features = np.concatenate([mfcc_mean, chroma_mean, stft_mean])
        return features
    except Exception as e:
        print(f"[WARN] Could not extract features from {file_path}: {e}")
        return None

# --- DataFrame to Feature Matrix ---
def dataframe_to_features_labels(df, feature_cols, label_col, augment=False):
    """
    For each row in df, extract audio features and concatenate with metadata features.
    Optionally apply augmentation.
    Returns X (features) and y (labels) for ML.
    """
    X, y = [], []
    for _, row in df.iterrows():
        audio_feats = extract_audio_features(row['file_path'], augment=augment)
        if audio_feats is None:
            continue
        meta_feats = row[feature_cols].values.astype(float)
        feats = np.concatenate([audio_feats, meta_feats])
        X.append(feats)
        y.append(row[label_col])
    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == '__main__':
    df = load_audio_samples()
    print(df.head()) 