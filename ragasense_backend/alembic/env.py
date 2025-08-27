import sys
import os
from logging.config import fileConfig
from sqlalchemy import pool, create_engine
from alembic import context

# Add app to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ragasense_backend.app.core.config import settings
from ragasense_backend.app.models.raga import Raga
from ragasense_backend.app.models.artist import Artist
from ragasense_backend.app.models.performance import Performance
from ragasense_backend.app.models.audio_sample import AudioSample
from ragasense_backend.app.models.composer import Composer
from ragasense_backend.app.models.type import Type
from ragasense_backend.app.models.tala import Tala
from ragasense_backend.app.models.song import Song
from sqlalchemy import MetaData

# Combine all model metadata
metadata = MetaData()
for model in [Raga, Artist, Performance, AudioSample, Composer, Type, Tala, Song]:
    if hasattr(model, '__table__'):
        model.__table__.tometadata(metadata)

target_metadata = metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

# Use a synchronous engine for autogenerate (schema inspection)
def get_sync_url():
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg"):
        return url.replace("postgresql+asyncpg", "postgresql+psycopg2")
    return url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_sync_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = create_engine(get_sync_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
