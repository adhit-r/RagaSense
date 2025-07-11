import asyncio
import os
import json
from collections import defaultdict
from ragasense_backend.app.core.db import AsyncSessionLocal
from ragasense_backend.app.models.raga import Raga
from ragasense_backend.app.models.artist import Artist
from ragasense_backend.app.models.performance import Performance
from ragasense_backend.app.models.audio_sample import AudioSample
from ragasense_backend.app.models.composer import Composer
from ragasense_backend.app.models.type import Type
from ragasense_backend.app.models.tala import Tala
from ragasense_backend.app.models.song import Song
from sqlalchemy import select
import inspect
import glob
from sqlalchemy import delete
from rapidfuzz import process

CARNATIC_PATH = os.path.join(os.path.dirname(__file__), '../../Music-data/Carnatic')
HINDUSTANI_PATH = os.path.join(os.path.dirname(__file__), '../../Music-data/Hindustani')

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

# Helper to filter dict by model columns

def filter_model_fields(model, data):
    allowed = set(c.name for c in model.__table__.columns)
    return {k: v for k, v in data.items() if k in allowed}

# Helper to compute tala count from angas string
TALA_ANGAS_MAP = {'Laghu': 4, 'Dhruta': 2, 'Anudhruta': 1, 'Visarjita': 3, 'Beat': 1}  # Default Laghu=4, can be extended

def compute_tala_count(angas):
    if not angas:
        return None
    total = 0
    for part in angas.split(','):
        part = part.strip()
        for k in TALA_ANGAS_MAP:
            if part.startswith(k):
                try:
                    n = int(part.split('-')[1])
                except Exception:
                    n = 1
                total += TALA_ANGAS_MAP[k] * n if k != 'Laghu' else n * 4  # Laghu is variable, default 4
    return total if total > 0 else None

# Helper to load canonical raga names from id.json
ID_JSON_PATH = os.path.join(CARNATIC_PATH, 'id.json')
CANONICAL_RAGA_NAMES = set()
if os.path.exists(ID_JSON_PATH):
    with open(ID_JSON_PATH, 'r', encoding='utf-8') as f:
        id_data = json.load(f)
        for v in id_data.get('raga', {}).values():
            if v and isinstance(v, list) and v[0]:
                CANONICAL_RAGA_NAMES.add(v[0].strip())

# --- Type descriptions for Carnatic and Hindustani ---
CARNATIC_TYPE_DESCRIPTIONS = {
    "Varnam": "A pedagogical and concert piece that encapsulates the core phrases and grammar of a raga. Structured in two parts (Purvanga and Uttaranga), it features both swara and sahitya passages, and is used for voice training and raga exposition.",
    "Kriti": "The principal compositional form in Carnatic music, typically in three sections: Pallavi, Anupallavi, and Charanam. Kritis are highly structured, devotional, and serve as the main vehicle for raga and lyrical expression.",
    "Keerthana": "Similar to kritis but generally simpler in structure and melody. Keerthanas are devotional songs, often with repetitive refrains and less complex rhythmic patterns.",
    "Geetam": "Introductory compositions for beginners, with simple melodies and rhythms. Geetams are set in straightforward talas and lack improvisational elements.",
    "Swarajati": "A form that bridges geetam and varnam, combining swara passages and lyrics. Swarajatis are used for both learning and performance, often in dance.",
    "Jatiswaram": "An instrumental or dance composition consisting of swara passages (no lyrics), set to a specific tala. Used in Bharatanatyam and as a technical exercise.",
    "Padam": "Slow, expressive compositions focusing on abhinaya (expression) and lyrical content. Padams are central to dance repertoire and explore themes of love and devotion.",
    "Javali": "Short, light classical compositions, often romantic or playful, with a focus on lyrical and melodic beauty. Common in dance performances.",
    "Thillana": "A rhythmic, lively composition, usually performed at the end of a concert or dance recital. Thillanas feature repetitive jatis (syllables) and brisk tempo.",
    "Ragam-Tanam-Pallavi": "An advanced, improvisational suite comprising ragam (melodic improvisation), tanam (rhythmic improvisation), and pallavi (theme with variations). RTPs are the centerpiece of a concert, showcasing creativity and technical mastery.",
    "Chittaswaram": "A set of swara passages inserted into kritis or varnams, providing melodic and rhythmic embellishment.",
    "Mangalam": "A short, auspicious composition sung at the end of a concert to invoke blessings and mark closure.",
    "Utsava Sampradaya Keerthana": "Compositions by Tyagaraja for temple festivals, characterized by simple melodies and devotional lyrics.",
    "Divyanama Keerthana": "Devotional songs with simple structure, composed by Tyagaraja, meant for group singing and processions.",
    "Daruvu": "A composition used in dance dramas, with alternating sections of lyrics and swaras, often narrating a story.",
    "Shabdam": "A dance composition that combines sahitya and swaras, used in the early part of a Bharatanatyam recital.",
    "Vachana": "Compositions with spoken or recited text, often used in dance for narrative purposes.",
    "Sankirtana": "Devotional group songs, often with call-and-response structure, used in bhajan and temple contexts.",
    "Tana Varnam": "A type of varnam with a focus on swara passages and rhythmic complexity, used for advanced training and performance.",
    "Pada Varnam": "A varnam with more elaborate sahitya, often used in dance for expressive abhinaya.",
}
HINDUSTANI_TYPE_DESCRIPTIONS = {
    "Khyal": "The principal vocal genre in Hindustani music, characterized by improvisation and flexibility in raga development, usually performed in slow (bada) and fast (chhota) tempos.",
    "Dhrupad": "The oldest surviving form of Hindustani vocal music, known for its meditative, austere style and strict adherence to raga and tala.",
    "Tarana": "A composition using mnemonic syllables (bols) set to a fast tempo, often performed at the end of a concert.",
    "Thumri": "A semi-classical vocal form, romantic and expressive, focusing on lyrical beauty and subtle raga nuances.",
    "Tappa": "A fast, intricate vocal form with rapid, complex ornamentation, originating from folk traditions.",
    "Bandish": "The fixed, composed part of a raga performance, serving as the main theme for improvisation.",
    "Bhajan": "A devotional song, often simple and repetitive, used in religious and spiritual contexts.",
    "Qawwali": "A Sufi devotional form, performed in groups with call-and-response, clapping, and spiritual lyrics.",
    "Dadra": "A light classical form, similar to thumri but set in dadra tala (6 beats).",
    "Chaiti": "A seasonal semi-classical form, sung during the Chaitra month, with folk influences.",
    "Kajri": "A semi-classical form sung during the rainy season, with folk roots.",
    "Hori": "A semi-classical form associated with the festival of Holi, playful and expressive.",
    "Jhula": "A folk-inspired form, often depicting the swinging of a cradle or swing.",
    "Abhang": "A devotional song form from Maharashtra, often sung in praise of Lord Vithoba.",
    "Rabindra sangeet": "Songs composed by Rabindranath Tagore, blending classical, folk, and Western elements.",
    "Jugalbandi": "A duet performance, typically between two soloists, showcasing interplay and improvisation.",
    "Lakshan geet": "Didactic compositions that describe the characteristics of a raga.",
    "Sargam geet": "Compositions using solfège syllables (sa, re, ga, ma, etc.) to teach raga structure.",
    "Sadra": "A classical composition set in a specific tala, often performed instrumentally.",
    "Dhamar": "A classical vocal form set in dhamar tala (14 beats), often associated with Holi.",
    "Haveli sangeet": "Temple music tradition, devotional in nature, performed in Vaishnavite temples.",
    "Chotta khyal": "A fast-tempo khyal composition, usually following a bada khyal in performance.",
    "Bada khyal": "A slow-tempo khyal composition, allowing for extensive improvisation.",
    "Bhavgeet": "A light classical or semi-classical song, often with emotional or romantic themes.",
    "Lecdem": "Lecture-demonstration, an educational presentation with musical examples.",
    "Raga": "A melodic framework for improvisation and composition in Indian classical music.",
    "Dhun": "A light instrumental piece, often based on folk tunes or film songs.",
}
GENERIC_TYPE_DESCRIPTION = "A classical composition type in Indian music."

# --- Seeding Types ---
async def seed_types(db, type_dir, region, desc_map):
    if not os.path.exists(type_dir):
        print(f"[WARN] Type directory not found: {type_dir}")
        return 0
    count = 0
    for fname in os.listdir(type_dir):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(type_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[WARN] Could not parse {fpath}: {e}")
                continue
        title = data.get('title', {})
        name = title.get('H')
        english_name = title.get('V')
        if not name:
            print(f"[WARN] Skipping type file with no name: {fname}")
            continue
        desc = desc_map.get(name, GENERIC_TYPE_DESCRIPTION)
        # Check for duplicate
        exists = await db.execute(select(Type).where(Type.name == name))
        if exists.scalar():
            print(f"[INFO] Type '{name}' already exists, skipping.")
            continue
        t = Type(name=name, english_name=english_name, description=desc, region=region)
        db.add(t)
        count += 1
    await db.commit()
    print(f"[INFO] Seeded types for region: {region}")
    return count

# Helper to truncate all main tables
async def truncate_all_tables(db):
    for model in [Performance, AudioSample, Song, Raga, Tala, Type, Composer, Artist]:
        await db.execute(delete(model))
    await db.commit()
    print("[INFO] All main tables truncated.")

# --- Robust async safe_insert functions for all main entities ---
async def safe_insert_entity(db, Model, unique_field, data, verbose=True):
    """
    Insert a record if it doesn't already exist, with validation and logging.
    - db: async SQLAlchemy session
    - Model: SQLAlchemy model class
    - unique_field: str, field name to check for duplicates (e.g., 'name')
    - data: dict, fields to insert
    """
    value = data.get(unique_field)
    if not value or not isinstance(value, str) or not value.strip():
        if verbose:
            print(f"[SKIP] Missing or invalid {unique_field}: {data}")
        return None
    # Check for existing record using scalar_one_or_none()
    result = await db.execute(select(Model).where(getattr(Model, unique_field) == value))
    existing = result.scalar_one_or_none()
    if existing:
        if verbose:
            print(f"[SKIP] {Model.__name__} '{value}' already exists.")
        return None
    # Insert new record
    obj = Model(**data)
    db.add(obj)
    await db.commit()
    if verbose:
        print(f"[INSERT] {Model.__name__} '{value}' inserted.")
    return obj

# Convenience wrappers for each entity
async def safe_insert_raga(db, data, verbose=True):
    return await safe_insert_entity(db, Raga, 'name', data, verbose)
async def safe_insert_composer(db, data, verbose=True):
    return await safe_insert_entity(db, Composer, 'name', data, verbose)
async def safe_insert_type(db, data, verbose=True):
    return await safe_insert_entity(db, Type, 'name', data, verbose)
async def safe_insert_tala(db, data, verbose=True):
    return await safe_insert_entity(db, Tala, 'name', data, verbose)
async def safe_insert_artist(db, data, verbose=True):
    return await safe_insert_entity(db, Artist, 'name', data, verbose)
async def safe_insert_song(db, data, verbose=True):
    return await safe_insert_entity(db, Song, 'title', data, verbose)
async def safe_insert_audio_sample(db, data, verbose=True):
    return await safe_insert_entity(db, AudioSample, 'file_path', data, verbose)
async def safe_insert_performance(db, data, verbose=True):
    # For performance, use a composite key if needed, here just 'date' as example
    return await safe_insert_entity(db, Performance, 'date', data, verbose)

def build_name_lookup(entity_list, main_field='name', alternate_field='alternate_names'):
    lookup = {}
    for entity in entity_list:
        # Add main name
        main_name = (getattr(entity, main_field, None) or '').strip().lower()
        if main_name:
            lookup[main_name] = entity
        # Add alternates
        for alt in getattr(entity, alternate_field, []) or []:
            alt_name = (alt or '').strip().lower()
            if alt_name:
                lookup[alt_name] = entity
    return lookup

def fuzzy_match(name, lookup_keys, threshold=85):
    if not name:
        return None
    result = process.extractOne(name, lookup_keys)
    if result and result[1] >= threshold:
        return result[0]
    return None

async def seed():
    async with AsyncSessionLocal() as db:
        print("[INFO] Truncating all main tables before reseeding...")
        await truncate_all_tables(db)
        # --- Counters for summary ---
        raga_count = 0
        composer_count = 0
        artist_count = 0
        performance_count = 0
        type_count = 0
        tala_count = 0
        song_count = 0
        audio_count = 0
        # --- Seed Carnatic Types ---
        carnatic_type_dir = os.path.join(CARNATIC_PATH, 'type')
        type_count += await seed_types(db, carnatic_type_dir, "Carnatic", CARNATIC_TYPE_DESCRIPTIONS)
        # --- Seed Hindustani Types ---
        hindustani_type_dir = os.path.join(HINDUSTANI_PATH, 'type')
        type_count += await seed_types(db, hindustani_type_dir, "Hindustani", HINDUSTANI_TYPE_DESCRIPTIONS)

        # --- 2. Talas ---
        carnatic_map = os.path.join(CARNATIC_PATH, 'english_map.json')
        hindustani_map = os.path.join(HINDUSTANI_PATH, 'english_map.json')
        talas_seeded = set()
        if os.path.exists(carnatic_map):
            with open(carnatic_map, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tala_names = data.get('tala name', {})
            tala_angas = data.get('tala angas', {})
            for k, name in tala_names.items():
                angas = None
                for ak, av in tala_angas.items():
                    if k.lower() in ak.lower() or name.lower() in ak.lower():
                        angas = av
                        break
                obj = await safe_insert_tala(db, {
                    'name': name,
                    'english_name': k,
                    'description': angas,
                    'region': 'Carnatic',
                })
                if obj:
                    tala_count += 1
                talas_seeded.add(name)
        if os.path.exists(hindustani_map):
            with open(hindustani_map, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tala_names = data.get('phonetic', {}).get('tala', {})
            for k, name in tala_names.items():
                if name in talas_seeded or not name or name == '?':
                    continue
                obj = await safe_insert_tala(db, {
                    'name': name,
                    'english_name': k,
                    'region': 'Hindustani',
                })
                if obj:
                    tala_count += 1
        await db.commit()

        # --- 3. Ragas ---
        for region, raga_dir in [("Carnatic", os.path.join(CARNATIC_PATH, 'raga')), ("Hindustani", os.path.join(HINDUSTANI_PATH, 'raga'))]:
            if os.path.exists(raga_dir):
                for raga_file in glob.glob(os.path.join(raga_dir, '*.json')):
                    with open(raga_file, 'r', encoding='utf-8') as f:
                        try:
                            raga_json = json.load(f)
                        except Exception as e:
                            print(f"[WARN] Could not parse {raga_file}: {e}")
                            continue
                    name = raga_json.get('title', {}).get('H')
                    if not name or not isinstance(name, str):
                        print(f"[WARN] Raga file {raga_file} missing canonical name, skipping.")
                        continue
                    info = {item['H']: item['V'] for item in raga_json.get('info', []) if 'H' in item and 'V' in item}
                    melakarta_number = None
                    if 'Melakartha' in info and isinstance(info['Melakartha'], str):
                        parts = info['Melakartha'].split()
                        if parts and parts[0].isdigit():
                            melakarta_number = int(parts[0])
                    arohana = info.get('Arohana')
                    avarohana = info.get('Avarohana')
                    obj = await safe_insert_raga(db, {
                        'name': name,
                        'melakarta_number': melakarta_number,
                        'arohana': arohana,
                        'avarohana': avarohana,
                        'region': region
                    })
                    if obj:
                        raga_count += 1
                await db.commit()

        # --- 4. Composers ---
        for region, composer_dir in [("Carnatic", os.path.join(CARNATIC_PATH, 'composer')), ("Hindustani", os.path.join(HINDUSTANI_PATH, 'composer'))]:
            if not os.path.exists(composer_dir):
                continue
            composers = load_json_files(composer_dir)
            for cname, cdata in composers.items():
                obj = await safe_insert_composer(db, {
                    'name': cname,
                    'bio': cdata.get('bio'),
                    'country': cdata.get('country'),
                    'era': cdata.get('era'),
                    'birth_year': cdata.get('birth_year'),
                    'death_year': cdata.get('death_year'),
                    'region': region
                })
                if obj:
                    composer_count += 1
            await db.commit()

        # --- 5. Artists ---
        for region, artist_dir in [("Carnatic", os.path.join(CARNATIC_PATH, 'artist')), ("Hindustani", os.path.join(HINDUSTANI_PATH, 'artist'))]:
            if not os.path.exists(artist_dir):
                continue
            artists = load_json_files(artist_dir)
            for aname, adata in artists.items():
                obj = await safe_insert_artist(db, {
                    'name': aname,
                    'bio': adata.get('bio'),
                    'country': adata.get('country'),
                    'era': adata.get('era'),
                    'birth_year': adata.get('birth_year'),
                    'death_year': adata.get('death_year'),
                    'genres': adata.get('genres'),
                    'region': region
                })
                if obj:
                    artist_count += 1
            await db.commit()

        # --- 6. Songs ---
        for region, song_dir in [("Carnatic", os.path.join(CARNATIC_PATH, 'song')), ("Hindustani", os.path.join(HINDUSTANI_PATH, 'song'))]:
            if not os.path.exists(song_dir):
                continue
            songs = load_json_files(song_dir)
            # for i, (stitle, sdata) in enumerate(songs.items()):
            #     if i < 5:
            #         print("DEBUG SONG DATA:", stitle, sdata, flush=True)
            def extract_info_field(info_list, field):
                for entry in info_list:
                    if entry.get('H', '').strip().lower() == field:
                        return entry.get('V')
                return None
            for i, (stitle, sdata) in enumerate(songs.items()):
                if i < 5:
                    print("DEBUG SONG DATA:", stitle, sdata, flush=True)
                info = sdata.get('info', [])
                raga = extract_info_field(info, 'raga')
                composer = extract_info_field(info, 'composer')
                obj = await safe_insert_song(db, {
                    'title': stitle,
                    'lyrics': sdata.get('lyrics'),
                    'language': sdata.get('language'),
                    'region': region,
                    'raw_raga_name': raga,
                    'raw_composer_name': composer,
                })
                if obj:
                    song_count += 1
            await db.commit()

        # --- 7. AudioSamples ---
        for region, audio_dir in [("Carnatic", os.path.join(CARNATIC_PATH, 'audio')), ("Hindustani", os.path.join(HINDUSTANI_PATH, 'audio'))]:
            if not os.path.exists(audio_dir):
                continue
            for fname in os.listdir(audio_dir):
                if not fname.endswith('.mp3'):
                    continue
                file_path = os.path.join(audio_dir, fname)
                obj = await safe_insert_audio_sample(db, {
                    'file_path': file_path,
                    'type': 'performance',
                    'region': region
                })
                if obj:
                    audio_count += 1
            await db.commit()

        # --- 8. Performances ---
        # Example: Only Carnatic for now, can be extended
        song_dir = os.path.join(CARNATIC_PATH, 'song')
        songs = load_json_files(song_dir) if os.path.exists(song_dir) else {}
        for stitle, sdata in songs.items():
            obj = await safe_insert_performance(db, {
                'date': sdata.get('date'),
                'venue': sdata.get('venue'),
                'notes': sdata.get('notes'),
                'duration': sdata.get('duration'),
                'rating': sdata.get('rating'),
                'region': 'Carnatic'
                # Add FKs after all entities are loaded if needed
            })
            if obj:
                performance_count += 1
        await db.commit()

        # --- 9. Raga Arohana/Avarohana Audio ---
        # For each raga, look for arohana/avarohana audio files and seed as AudioSample
        for region, raga_path in [("Carnatic", os.path.join(CARNATIC_PATH, 'raga')), ("Hindustani", os.path.join(HINDUSTANI_PATH, 'raga'))]:
            if not os.path.exists(raga_path):
                continue
            for fname in os.listdir(raga_path):
                if not fname.endswith('.json'):
                    continue
                with open(os.path.join(raga_path, fname), 'r', encoding='utf-8') as f:
                    raga_data = json.load(f)
                raga_name = raga_data.get('name')
                if not raga_name:
                    continue
                # Find raga in DB
                raga_obj = await db.execute(select(Raga).where(Raga.name == raga_name))
                raga = raga_obj.scalars().first()
                if not raga:
                    continue
                # Look for audio files (e.g., arohana_{raga_name}.mp3, avarohana_{raga_name}.mp3)
                for typ in ['arohana', 'avarohana']:
                    pattern = os.path.join(raga_path, f'{typ}_{raga_name}.*')
                    for audio_file in glob.glob(pattern):
                        ext = os.path.splitext(audio_file)[1].lower()
                        if ext not in ['.mp3', '.wav', '.flac', '.ogg']:
                            continue
                        with open(audio_file, 'rb') as af:
                            audio_data = af.read()
                        audio_obj = filter_model_fields(AudioSample, {
                            'file_path': audio_file,
                            'type': typ,
                            'data': audio_data,
                            'raga_id': raga.id,
                            'region': region,
                        })
                        await db.merge(AudioSample(**audio_obj))
        print(f"Seeded: {raga_count} ragas, {composer_count} composers, {artist_count} artists, {performance_count} performances, {type_count} types, {tala_count} talas, {song_count} songs, {audio_count} audio samples.")

        # --- Foreign Key Linking: Song -> Raga, Composer (with alternates and fuzzy matching) ---
        print("[INFO] Linking foreign keys for songs (with alternates and fuzzy matching)...")
        result = await db.execute(select(Song))
        songs = result.scalars().all()
        result = await db.execute(select(Raga))
        ragas = result.scalars().all()
        result = await db.execute(select(Composer))
        composers = result.scalars().all()
        raga_lookup = build_name_lookup(ragas, main_field='name', alternate_field='alternate_names')
        composer_lookup = build_name_lookup(composers, main_field='name', alternate_field='alternate_names')

        # --- Debug prints ---
        print("Sample song raga/composer names:")
        for song in songs[:5]:
            print("Song:", getattr(song, 'title', None), "| raga:", getattr(song, 'raga', None), "| composer:", getattr(song, 'composer', None))
        print("Sample raga names in lookup:")
        for k in list(raga_lookup.keys())[:5]:
            print("Raga key:", k)
        print("Sample composer names in lookup:")
        for k in list(composer_lookup.keys())[:5]:
            print("Composer key:", k)

        linked, missing_raga, missing_composer, fuzzy_raga, fuzzy_composer = 0, 0, 0, 0, 0
        for song in songs:
            raga_name = (getattr(song, 'raw_raga_name', None) or '').strip().lower()
            composer_name = (getattr(song, 'raw_composer_name', None) or '').strip().lower()
            raga = raga_lookup.get(raga_name)
            composer = composer_lookup.get(composer_name)
            # Fuzzy match if not found
            if not raga and raga_name:
                suggestion = fuzzy_match(raga_name, raga_lookup.keys())
                if suggestion:
                    print(f"[FUZZY] Raga '{raga_name}' → '{suggestion}'")
                    raga = raga_lookup[suggestion]
                    fuzzy_raga += 1
            if not composer and composer_name:
                suggestion = fuzzy_match(composer_name, composer_lookup.keys())
                if suggestion:
                    print(f"[FUZZY] Composer '{composer_name}' → '{suggestion}'")
                    composer = composer_lookup[suggestion]
                    fuzzy_composer += 1
            # Now link if found
            if raga:
                song.raga_id = raga.id
            if composer:
                song.composer_id = composer.id
            if raga and composer:
                linked += 1
            else:
                if not raga:
                    missing_raga += 1
                if not composer:
                    missing_composer += 1
        await db.commit()
        print(f"[INFO] Linked {linked} songs. Missing raga: {missing_raga}, missing composer: {missing_composer}, fuzzy raga matches: {fuzzy_raga}, fuzzy composer matches: {fuzzy_composer}")
        # Print up to 10 sample missing links for investigation
        missing_samples = []
        for song in songs:
            if not song.raga_id or not song.composer_id:
                missing_samples.append((song.title, song.raw_raga_name, song.raw_composer_name))
            if len(missing_samples) >= 10:
                break
        if missing_samples:
            print("[DEBUG] Sample missing links (title, raw_raga_name, raw_composer_name):")
            for t, r, c in missing_samples:
                print(f"    {t} | raga: {r} | composer: {c}")
        print("[NOTE] For fuzzy matching, install rapidfuzz: pip install rapidfuzz")

if __name__ == "__main__":
    asyncio.run(seed()) 