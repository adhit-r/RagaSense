import sys
import os
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from alembic import context

# Add ragasense_backend to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ragasense_backend'))

from backend.core.config import settings
from backend.models.raga import Base as RagaBase
from backend.models.artist import Artist
from backend.models.performance import Performance
from backend.models.audio_sample import AudioSample

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = RagaBase.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    """
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    """
    from sqlalchemy.ext.asyncio import AsyncEngine
    connectable = context.config.attributes.get('connection', None)
    if connectable is None:
        connectable = RagaBase.metadata.bind
    if connectable is None:
        from sqlalchemy.ext.asyncio import create_async_engine
        connectable = create_async_engine(settings.DATABASE_URL, poolclass=pool.NullPool)

    async def do_run_migrations(connection):
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        async with context.begin_transaction():
            await context.run_migrations()

    import asyncio
    async def run():
        async with connectable.connect() as connection:
            await do_run_migrations(connection)
    asyncio.run(run())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
