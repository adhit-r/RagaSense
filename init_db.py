#!/usr/bin/env python3
"""
Initialize the database and import ragas.

This script will:
1. Create the PostgreSQL database if it doesn't exist
2. Create all database tables
3. Import sample raga data
"""
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

# Load environment variables
load_dotenv()

# Database configuration from environment variables
DB_USER = os.getenv('DB_USER', 'raga_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'your_password')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'raga_detector')
DATABASE_URL = os.getenv('DATABASE_URL', 
    f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def create_database():
    """Create the database if it doesn't exist."""
    # Create a connection to the default 'postgres' database
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    engine = create_engine(db_url, isolation_level='AUTOCOMMIT')
    
    # Create the database if it doesn't exist
    try:
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {'dbname': DB_NAME}
            ).scalar()
            
            if not result:
                conn.execute(text(f"CREATE DATABASE {DB_NAME} OWNER {DB_USER}"))
                print(f"Database '{DB_NAME}' created successfully with owner '{DB_USER}'.")
            else:
                print(f"Database '{DB_NAME}' already exists.")
                
            # Skip PostGIS extension as it's not required for basic functionality
            print("Skipping PostGIS extension as it's not required for basic functionality.")
            
    except Exception as e:
        print(f"Error creating database: {e}")
        return False
    finally:
        engine.dispose()
    
    return True

def create_tables():
    """Create all database tables."""
    # Import inside function to avoid circular imports
    from backend import create_app, db
    from backend.models.raga import Raga  # Import models to ensure they are registered
    
    app = create_app()
    with app.app_context():
        try:
            print("Creating database tables...")
            
            # Create all tables
            db.create_all()
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"Available tables: {tables}")
            
            print("Database tables created successfully.")
            return True
        except Exception as e:
            print(f"Error creating database tables: {e}")
            import traceback
            traceback.print_exc()
            return False

def validate_raga_data(raga_data):
    """Validate raga data before importing."""
    required_fields = ['name', 'arohana', 'avarohana', 'tradition']
    for field in required_fields:
        if field not in raga_data:
            raise ValueError(f"Missing required field: {field} in raga {raga_data.get('name', 'Unknown')}")
    
    # Validate tradition
    valid_traditions = {'Hindustani', 'Carnatic'}
    if raga_data['tradition'] not in valid_traditions:
        raise ValueError(f"Invalid tradition {raga_data['tradition']} for raga {raga_data['name']}")
    
    # Validate note sequences and structure
    valid_notes = {'Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni', 
                  'Re♭', 'Re♯', 'Ga♭', 'Ga♯', 'Ma♯', 'Dha♭', 'Dha♯', 'Ni♭', 'Ni♯'}
    
    # Check basic note sequence validity
    valid_notes.add("Sa'")  # Add upper octave Sa
    for note in raga_data.get('arohana', []) + raga_data.get('avarohana', []):
        if note not in valid_notes:
            raise ValueError(f"Invalid note {note} in raga {raga_data['name']}")
    
    # Validate swara sequence rules
    arohana = raga_data.get('arohana', [])
    avarohana = raga_data.get('avarohana', [])
    
    # For certain ragas like Puriya Dhanashri, arohana can start with Ni
    # But it must end with Sa or Sa'
    if (arohana[0] not in {'Sa', 'Ni'}) or (arohana[-1] != 'Sa' and arohana[-1] != "Sa'"):
        raise ValueError(f"Arohana must start with Sa/Ni and end with Sa/Sa' in raga {raga_data['name']}")
    if (avarohana[0] != 'Sa' and avarohana[0] != "Sa'") or avarohana[-1] != 'Sa':
        raise ValueError(f"Avarohana must start with Sa/Sa' and end with Sa in raga {raga_data['name']}")
    
    def validate_progression(sequence, direction='ascending'):
        # Convert notes to numeric degrees for easier comparison
        def get_degree(note):
            # Basic notes
            base_degrees = {'Sa': 1, 'Re': 2, 'Ga': 3, 'Ma': 4, 'Pa': 5, 'Dha': 6, 'Ni': 7}
            
            # Handle upper octave Sa specially
            if note == "Sa'":
                return 8
            
            # Remove accidentals to get base note
            base_note = note.rstrip('♯♭')
            base_degree = base_degrees[base_note]
            
            # Adjust for accidentals
            if '♭' in note:
                base_degree -= 0.1
            elif '♯' in note:
                base_degree += 0.1
                
            return base_degree
        
        def format_sequence(seq):
            return ' → '.join(seq)
            
        # Convert to degrees and validate each transition
        notes = []
        prev_degree = None
        for i, note in enumerate(sequence):
            curr_degree = get_degree(note)
            notes.append(note)
            
            # Skip validation for first note
            if prev_degree is None:
                prev_degree = curr_degree
                continue
            
            if direction == 'ascending':
                # Special cases for ascending sequence
                if note in ['Sa', "Sa'"]:  # Allow Sa or Sa' at any point
                    pass
                elif i == 1 and sequence[0] == 'Ni':  # Allow Ni to Sa transition at start
                    pass
                elif curr_degree < prev_degree:  # Must ascend or stay same
                    seq = format_sequence(notes[-2:])
                    raise ValueError(
                        f"Invalid ascending sequence in {raga_data['name']}: {seq}"
                        f"\nFull sequence: {format_sequence(sequence)}"
                    )
            else:  # descending
                # Special cases for descending sequence
                if i == 0 and note == 'Sa':  # Allow starting with Sa, and the transition from Sa to any note
                    pass
                elif i == 1 and sequence[0] == 'Sa':  # Allow any note as the second note when starting with Sa
                    pass
                elif prev_degree == get_degree("Sa'"):  # Allow any note after Sa'
                    pass
                elif note == 'Sa' and i == len(sequence) - 1:  # Allow ending with Sa
                    pass
                elif curr_degree > prev_degree:  # Must descend or stay same
                    seq = format_sequence(notes[-2:])
                    raise ValueError(
                        f"Invalid descending sequence in {raga_data['name']}: {seq}"
                        f"\nFull sequence: {format_sequence(sequence)}"
                    )
            
            prev_degree = curr_degree
    
    validate_progression(arohana, 'ascending')
    validate_progression(avarohana, 'descending')
    
    # Validate Carnatic-specific fields
    if raga_data['tradition'] == 'Carnatic':
        # Validate melakarta number and structure
        if 'melakarta_number' in raga_data and raga_data['melakarta_number'] is not None:
            if not isinstance(raga_data['melakarta_number'], int) or \
               not (1 <= raga_data['melakarta_number'] <= 72):
                raise ValueError(f"Invalid melakarta number for raga {raga_data['name']}")
            
            # Validate melakarta structure
            if len(arohana) != 8 or len(avarohana) != 8:
                raise ValueError(f"Melakarta raga {raga_data['name']} must have 7 distinct swaras")
        
        # Validate chakra
        if 'chakra' in raga_data:
            valid_chakras = {'Indu', 'Netra', 'Agni', 'Veda', 'Bana', 'Rishi'}
            if raga_data['chakra'] not in valid_chakras:
                raise ValueError(f"Invalid chakra {raga_data['chakra']} for raga {raga_data['name']}")
        
        # Validate janya relationships
        if 'janya_ragas' in raga_data:
            if not isinstance(raga_data['janya_ragas'], list):
                raise ValueError(f"Janya ragas must be a list in {raga_data['name']}")
    
    # Validate Hindustani-specific fields
    if raga_data['tradition'] == 'Hindustani':
        # Validate thaat
        if 'thaat' in raga_data:
            valid_thaats = {'Bilawal', 'Kalyan', 'Khamaj', 'Bhairav', 'Purvi', 
                           'Marwa', 'Kafi', 'Asavari', 'Bhairavi', 'Todi'}
            if raga_data['thaat'] not in valid_thaats:
                raise ValueError(f"Invalid thaat {raga_data['thaat']} for raga {raga_data['name']}")
        
        # Validate time periods
        if 'time' in raga_data:
            valid_times = {'Morning', 'Sunrise', 'Afternoon', 'Evening', 'Night', 'Late Night',
                          'First Prahar of Night', 'Second Prahar of Night', 'Third Prahar of Night',
                          'Fourth Prahar of Night'}
            for time in raga_data['time']:
                if time not in valid_times:
                    raise ValueError(f"Invalid time period {time} in raga {raga_data['name']}")
        
        # Validate vadi-samvadi relationship
        if 'vadi' in raga_data and 'samvadi' in raga_data:
            # Convert note names to scale degrees
            vadi_base = raga_data['vadi'].rstrip('♯♭')
            samvadi_base = raga_data['samvadi'].rstrip('♯♭')
            vadi_degree = {'Sa': 1, 'Re': 2, 'Ga': 3, 'Ma': 4, 
                          'Pa': 5, 'Dha': 6, 'Ni': 7}[vadi_base]
            samvadi_degree = {'Sa': 1, 'Re': 2, 'Ga': 3, 'Ma': 4, 
                            'Pa': 5, 'Dha': 6, 'Ni': 7}[samvadi_base]
            
            # Calculate interval between vadi and samvadi
            interval = (samvadi_degree - vadi_degree) % 7
            # In Indian classical music, samvadi should be a perfect fourth (3 steps)
            # or perfect fifth (4 steps) from the vadi
            if interval not in [3, 4]:
                raise ValueError(f"Invalid vadi-samvadi relationship in raga {raga_data['name']}")
    
    # Validate JSONB array fields
    array_fields = ['alternate_names', 'varjya_swaras', 'time', 'season', 
                   'rasa', 'mood', 'notable_compositions', 'janya_ragas']
    for field in array_fields:
        if field in raga_data and not isinstance(raga_data[field], list):
            raise ValueError(f"Field {field} must be an array in raga {raga_data['name']}")
    
    # Validate string fields
    string_fields = ['vadi', 'samvadi', 'jati', 'description', 'history', 
                    'carnatic_equivalent', 'hindustani_equivalent', 'janaka_raga']
    for field in string_fields:
        if field in raga_data and not isinstance(raga_data[field], str):
            raise ValueError(f"Field {field} must be a string in raga {raga_data['name']}")
    
    return True

def import_sample_data():
    """Import sample raga data from JSON file."""
    import json
    from pathlib import Path
    from backend import create_app, db
    from backend.models.raga import Raga
    
    app = create_app()
    with app.app_context():
        try:
            print("Checking if sample data needs to be imported...")

            # Check if there are any ragas in the database
            with db.session.begin():
                if db.session.query(Raga).first() is None:
                    print("No ragas found. Importing sample data...")

                    # Load sample data from JSON file
                    json_path = Path(__file__).parent / 'backend' / 'data' / 'ragas_sample.json'
                    if not json_path.exists():
                        print(f"Sample data file not found at: {json_path}")
                        return False

                    with open(json_path, 'r', encoding='utf-8') as f:
                        sample_ragas = json.load(f)

                    # First pass: Create all ragas
                    raga_objects = {}
                    for raga_data in sample_ragas:
                        # Validate data
                        validate_raga_data(raga_data)
                        
                        # Convert single values to lists for JSONB array fields
                        if 'time' in raga_data and isinstance(raga_data['time'], str):
                            raga_data['time'] = [raga_data['time']]
                            
                        # Add empty values for any missing fields
                        raga_data.setdefault('alternate_names', [])
                        raga_data.setdefault('varjya_swaras', [])
                        raga_data.setdefault('season', [])
                        raga_data.setdefault('rasa', [])
                        raga_data.setdefault('audio_features', {})
                        raga_data.setdefault('history', '')
                        
                        # Store relationships for second pass
                        related_ragas = raga_data.pop('related_ragas', [])
                        parent_name = raga_data.pop('parent_raga', None)
                        
                        # Create and add the raga to the session
                        raga = Raga(**raga_data)
                        db.session.add(raga)
                        db.session.flush()  # Get the ID
                        
                        raga_objects[raga.name] = {
                            'object': raga,
                            'related': related_ragas,
                            'parent': parent_name
                        }
                    
                    # Second pass: Set up relationships
                    for raga_name, raga_info in raga_objects.items():
                        raga = raga_info['object']
                        
                        # Set parent raga
                        if raga_info['parent']:
                            parent = raga_objects.get(raga_info['parent'])
                            if parent:
                                raga.parent_raga_id = parent['object'].id
                        
                        # Set related ragas
                        for related_name in raga_info['related']:
                            related = raga_objects.get(related_name)
                            if related:
                                raga.similar_ragas.append(related['object'])

                    # Commit all changes
                    db.session.commit()
                    print(f"Successfully imported {len(sample_ragas)} ragas from {json_path}")
                else:
                    count = db.session.query(Raga).count()
                    print(f"Database already contains {count} ragas. Skipping sample data import.")

            return True
            
        except Exception as e:
            print(f"Error importing sample data: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return False

def check_database_integrity():
    """Check database integrity and relationships."""
    from backend import create_app, db
    from backend.models.raga import Raga
    from sqlalchemy import text
    
    app = create_app()
    with app.app_context():
        try:
            print("\nChecking database integrity...")
            issues = []
            
            # 1. Check for orphaned relationships
            orphaned = db.session.execute(
                text("""
                SELECT COUNT(*) 
                FROM raga_relationships rr 
                LEFT JOIN ragas r1 ON rr.raga_id = r1.id 
                LEFT JOIN ragas r2 ON rr.related_raga_id = r2.id 
                WHERE r1.id IS NULL OR r2.id IS NULL
                """)
            ).scalar()
            
            if orphaned > 0:
                issues.append(f"Found {orphaned} orphaned relationship records")

            # 2. Check for invalid parent references
            invalid_parents = db.session.execute(
                text("""
                SELECT COUNT(*) 
                FROM ragas r 
                LEFT JOIN ragas p ON r.parent_raga_id = p.id 
                WHERE r.parent_raga_id IS NOT NULL AND p.id IS NULL
                """)
            ).scalar()
            
            if invalid_parents > 0:
                issues.append(f"Found {invalid_parents} invalid parent raga references")

            # 3. Check for circular parent relationships
            circular = db.session.execute(
                text("""
                WITH RECURSIVE ancestry AS (
                    SELECT id, parent_raga_id, 1 as depth
                    FROM ragas
                    UNION ALL
                    SELECT r.id, r.parent_raga_id, a.depth + 1
                    FROM ragas r
                    JOIN ancestry a ON r.id = a.parent_raga_id
                    WHERE a.depth < 100
                )
                SELECT COUNT(DISTINCT id) 
                FROM ancestry 
                WHERE depth >= 100
                """)
            ).scalar()
            
            if circular > 0:
                issues.append(f"Found {circular} ragas with circular parent relationships")

            # 4. Check for valid JSON in JSONB fields
            for field in ['alternate_names', 'arohana', 'avarohana', 'varjya_swaras', 
                         'time', 'season', 'rasa', 'mood', 'notable_compositions']:
                invalid = db.session.query(Raga).filter(
                    text(f"{field}::text !~ '^\\[.*\\]$'")
                ).count()
                if invalid > 0:
                    issues.append(f"Found {invalid} ragas with invalid {field} array format")

            # 5. Check for required fields
            missing_required = db.session.query(Raga).filter(
                db.or_(
                    Raga.name == None,
                    Raga.arohana == None,
                    Raga.avarohana == None
                )
            ).count()
            
            if missing_required > 0:
                issues.append(f"Found {missing_required} ragas missing required fields")

            if issues:
                print("\nDatabase integrity issues found:")
                for issue in issues:
                    print(f"- {issue}")
                return False
            
            print("Database integrity check passed successfully!")
            return True
            
        except Exception as e:
            print(f"Error checking database integrity: {e}")
            import traceback
            traceback.print_exc()
            return False

def init_db():
    """Initialize the database and import ragas."""
    print("Initializing database...")
    print(f"Using database: {DATABASE_URL}")

    try:
        # Step 1: Create database if it doesn't exist
        print("\n=== Step 1: Checking/Creating Database ===")
        if not create_database():
            print("Failed to create database.")
            return False

        # Step 2: Create tables
        print("\n=== Step 2: Creating Tables ===")
        if not create_tables():
            print("Failed to create tables.")
            return False

        # Step 3: Import sample data
        print("\n=== Step 3: Importing Sample Data ===")
        if not import_sample_data():
            print("Failed to import sample data.")
            return False
            
        print("\n=== Step 4: Checking Database Integrity ===")
        if not check_database_integrity():
            print("Database integrity check failed.")
            return False

        print("\n=== Database Initialization Complete! ===")
        print("The database has been initialized successfully!")
        return True
        
    except Exception as e:
        print(f"\n!!! Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if not init_db():
        sys.exit(1)
