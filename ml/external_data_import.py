import os
import json
import pandas as pd
from sqlalchemy import create_engine

# --- DB connection (edit as needed) ---
DB_USER = os.getenv('DB_USER', 'raga_user')
DB_PASS = os.getenv('DB_PASS', 'raga_pass')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'ragasense_db')
DB_URL = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

def load_saraga(metadata_path):
    if metadata_path.endswith('.csv'):
        df = pd.read_csv(metadata_path)
    else:
        df = pd.read_json(metadata_path)
    print('[Saraga] Sample rows:')
    print(df.head())
    print('[Saraga] Unique ragas:', df['raga'].nunique() if 'raga' in df.columns else 'N/A')
    print('[Saraga] Unique artists:', df['artist'].nunique() if 'artist' in df.columns else 'N/A')
    return df

def map_and_insert_saraga(df):
    df = df.copy()
    df.rename(columns={
        'raga': 'raga_name',
        'artist': 'artist_name',
        'tala': 'tala_name',
        'audio_path': 'file_path',
        'title': 'song_title',
    }, inplace=True)
    keep_cols = ['file_path', 'raga_name', 'artist_name', 'tala_name', 'song_title']
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    df = df[keep_cols]
    insert_dataframe_to_db(df, 'audio_samples')

# --- Kaggle dataset loader template ---
def load_kaggle_dataset(kaggle_path):
    """
    Load a Kaggle dataset (CSV or JSON), preview, and return as DataFrame.
    User: Download the dataset manually from Kaggle and provide the file path.
    """
    if kaggle_path.endswith('.csv'):
        df = pd.read_csv(kaggle_path)
    else:
        df = pd.read_json(kaggle_path)
    print('[Kaggle] Sample rows:')
    print(df.head())
    print('[Kaggle] Columns:', df.columns.tolist())
    return df

def map_and_insert_kaggle(df, mapping, table_name):
    """
    Map Kaggle DataFrame columns to your DB schema using the provided mapping dict.
    Example mapping: {'raga': 'raga_name', 'artist': 'artist_name', ...}
    """
    df = df.copy()
    df.rename(columns=mapping, inplace=True)
    keep_cols = list(mapping.values())
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None
    df = df[keep_cols]
    insert_dataframe_to_db(df, table_name)

def insert_dataframe_to_db(df, table_name):
    engine = create_engine(DB_URL)
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f'Inserted {len(df)} rows into {table_name}')

if __name__ == '__main__':
    # --- Saraga integration ---
    saraga_metadata_path = '/path/to/saraga_metadata.json'  # Update this path
    if os.path.exists(saraga_metadata_path):
        saraga_df = load_saraga(saraga_metadata_path)
        map_and_insert_saraga(saraga_df)
    else:
        print('[WARN] Saraga metadata file not found.')

    # --- Kaggle integration example ---
    kaggle_path = '/path/to/kaggle_dataset.csv'  # Update this path
    if os.path.exists(kaggle_path):
        kaggle_df = load_kaggle_dataset(kaggle_path)
        # Example: define your mapping for this Kaggle dataset
        kaggle_mapping = {
            'raga': 'raga_name',
            'artist': 'artist_name',
            'tala': 'tala_name',
            'audio_path': 'file_path',
            'title': 'song_title',
        }
        map_and_insert_kaggle(kaggle_df, kaggle_mapping, 'audio_samples')
    else:
        print('[WARN] Kaggle dataset file not found.') 