from sqlalchemy import create_engine, text
from ragasense_backend.app.core.config import settings

# Use a synchronous engine for inspection
engine = create_engine(settings.DATABASE_URL.replace('asyncpg', 'psycopg2'))

tables = ['ragas', 'artists', 'audio_samples', 'composers', 'performances', 'talas', 'types', 'songs']

with engine.connect() as conn:
    print("Table row counts:")
    for t in tables:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
        print(f"{t}: {count}")