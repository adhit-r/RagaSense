import asyncio
import os
import json
from collections import defaultdict
from backend.core.db import AsyncSessionLocal
from backend.models.raga import Raga
from backend.models.artist import Artist
from backend.models.performance import Performance
from backend.models.audio_sample import AudioSample
from backend.models.composer import Composer  # If exists
from backend.models.type import Type  # If exists
from sqlalchemy import select
from backend.models.raga_english_map import RagaEnglishMap
from backend.models.type_english_map import TypeEnglishMap
from backend.models.tala_english_map import TalaEnglishMap
from backend.models.tala import Tala  # Add this import if not present
import pathlib

# Updated paths to ragasense_backend/Music-data
BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ragasense_backend/Music-data'))
CARNATIC_PATH = os.path.join(BASE_DATA_PATH, 'Carnatic')
HINDUSTANI_PATH = os.path.join(BASE_DATA_PATH, 'Hindustani')

# Helper: Load all JSON files in a directory
def load_json_files(directory):
    data = {}
    for fname in os.listdir(directory):
        if fname.endswith('.json'):
            try:
                with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
                    data[os.path.splitext(fname)[0]] = json.load(f)
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    return data

async def seed():
    async with AsyncSessionLocal() as db:
        # --- 0. English Maps ---
        for system, path in [('Carnatic', CARNATIC_PATH), ('Hindustani', HINDUSTANI_PATH)]:
            english_map_path = os.path.join(path, 'english_map.json')
            if not os.path.exists(english_map_path):
                print(f"No english_map.json in {system}")
                continue
        with open(english_map_path, 'r', encoding='utf-8') as f:
            english_map = json.load(f)
        raga_map = english_map['phonetic']['raga']
        type_map = english_map['phonetic']['type']
        tala_map = english_map['phonetic']['tala']
        for k, v in raga_map.items():
            db.add(RagaEnglishMap(id=int(k), english_name=v))
        for k, v in type_map.items():
            db.add(TypeEnglishMap(id=int(k), english_name=v))
        for k, v in tala_map.items():
            db.add(TalaEnglishMap(id=int(k), english_name=v))
        await db.flush()

        # --- 1. Types ---
        for system, path in [('Carnatic', CARNATIC_PATH), ('Hindustani', HINDUSTANI_PATH)]:
            type_dir = os.path.join(path, 'type')
            if not os.path.exists(type_dir):
                continue
            for fname in os.listdir(type_dir):
                if not fname.endswith('.json'):
                    continue
                with open(os.path.join(type_dir, fname), 'r', encoding='utf-8') as f:
                    tdata = json.load(f)
                tname = tdata.get('name')
                if not tname:
                    continue
                    obj = Type(
                        name=tname,
                    english_name=tdata.get('english_name'),
                    description=tdata.get('description', ''),
                    )
                    db.merge(obj)
                    await db.flush()

        # --- 2. Talas ---
        for system, path in [('Carnatic', CARNATIC_PATH), ('Hindustani', HINDUSTANI_PATH)]:
            tala_dir = os.path.join(path, 'tala')
            if not os.path.exists(tala_dir):
                continue
            for fname in os.listdir(tala_dir):
                if not fname.endswith('.json'):
                    continue
                with open(os.path.join(tala_dir, fname), 'r', encoding='utf-8') as f:
                    tdata = json.load(f)
                tname = tdata.get('name')
                obj = Tala(
                    name=tname,
                    english_name=tdata.get('english_name'),
                )
                db.add(obj)
            await db.flush()

        # --- 3. Ragas ---
        for system, path in [('Carnatic', CARNATIC_PATH), ('Hindustani', HINDUSTANI_PATH)]:
            raga_dir = os.path.join(path, 'raga')
        ragas = load_json_files(raga_dir) if os.path.exists(raga_dir) else {}
        for rname, rdata in ragas.items():
            obj = Raga(
                name=rname,
                melakarta_number=rdata.get('melakarta'),
                thaat=rdata.get('type'),
                    arohana=rdata.get('structure', {}).get('arohana'),
                    avarohana=rdata.get('structure', {}).get('avarohana'),
                characteristic_phrases=rdata.get('notes'),
                description=rdata.get('description'),
                    alternate_names=[rdata.get('title', {}).get('V')] if rdata.get('title', {}).get('V') else None,
                    icon=rdata.get('title', {}).get('I'),
                    melakarta_name=None,
                stats=rdata.get('stats'),
                    info=rdata.get('info'),
                songs=rdata.get('songs'),
                keyboard=rdata.get('keyboard'),
                    english_name=None,
            )
            db.add(obj)
        await db.flush()

        # --- 4. Composers ---
        for system, path in [('Carnatic', CARNATIC_PATH), ('Hindustani', HINDUSTANI_PATH)]:
            composer_dir = os.path.join(path, 'composer')
        composers = load_json_files(composer_dir) if os.path.exists(composer_dir) else {}
        for cname, cdata in composers.items():
            obj = Composer(
                name=cname,
                bio=cdata.get('bio', ''),
                country=cdata.get('country', ''),
                era=cdata.get('era', ''),
            )
            db.add(obj)
        await db.flush()

        # --- 5. Artists ---
        for system, path in [('Carnatic', CARNATIC_PATH), ('Hindustani', HINDUSTANI_PATH)]:
            artist_dir = os.path.join(path, 'artist')
        artists = load_json_files(artist_dir) if os.path.exists(artist_dir) else {}
        for aname, adata in artists.items():
            obj = Artist(
                name=aname,
                bio=adata.get('bio', ''),
                country=adata.get('country', ''),
                era=adata.get('era', ''),
            )
            db.add(obj)
        await db.flush()

        # --- 6. Songs/Performances/Audio ---
        for system, path in [('Carnatic', CARNATIC_PATH), ('Hindustani', HINDUSTANI_PATH)]:
            song_dir = os.path.join(path, 'song')
            audio_dir = os.path.join(path, 'audio')
        songs = load_json_files(song_dir) if os.path.exists(song_dir) else {}
        for stitle, sdata in songs.items():
            # Link raga, composer, artist, type
                # (You may need to query for these by name if not in memory)
            # Audio
            audio_file = sdata.get('audio')
            audio_obj = None
            if audio_file:
                audio_path = os.path.join(audio_dir, audio_file)
                if os.path.exists(audio_path):
                    audio_obj = AudioSample(
                        file_path=audio_path,
                        type='performance',
                            audio_metadata={},
                    )
                    db.add(audio_obj)
                    await db.flush()
            # Performance
            perf = Performance(
                date=sdata.get('date'),
                venue=sdata.get('venue'),
                notes=sdata.get('notes'),
                    title=stitle,
                audio_sample=audio_obj,
            )
            db.add(perf)
            await db.flush()
        await db.commit()
        print("Seeding complete for all tables from ragasense_backend/Music-data.")

if __name__ == "__main__":
    asyncio.run(seed()) 