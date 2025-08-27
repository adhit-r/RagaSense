# RagaSense Backend

A modern, modular FastAPI backend for raga detection, analysis, and musicological research.

## Features
- FastAPI for async, high-performance APIs
- Modular structure: models, schemas, endpoints, CRUD, ML integration
- Expanded database schema: Raga, Artist, Performance, AudioSample
- Alembic migrations for schema management
- Seed scripts for rich sample data
- Local Postgres support (no Docker required)
- Clean, maintainable codebase

## Directory Structure
```
ragasense_backend/
  app/
    api/
      endpoints/
      __init__.py
    core/
    models/
    schemas/
    crud/
    seed/
    main.py
  alembic/
  alembic.ini
  requirements.txt
  README.md
  .env
```

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your local Postgres in `.env`:**
   ```env
   DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5432/<dbname>
   ```
3. **Run Alembic migrations:**
   ```bash
   alembic upgrade head
   ```
4. **Seed the database:**
   ```bash
   python -m app.seed.seed_data
   ```
5. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Docs
Once running, visit [http://localhost:8000/docs](http://localhost:8000/docs) for interactive OpenAPI documentation.

---

For more details, see the code and comments in each module.

## Notes
- The Music-data directory (ragasense_backend/Music-data) is ignored by git to avoid tracking large datasets.

## Verifying Seeded Data
- After seeding, you can verify the data by querying the database directly or using the API (e.g., GET /ragas).
- Example: Start the server and visit http://localhost:8000/docs to try endpoints.

## Improving Async Handling in Seeding
- The seed script uses async SQLAlchemy sessions. Ensure all DB operations are awaited (e.g., use `await db.merge(obj)` if merge is async).
- If you see warnings about coroutines not awaited, update the script to await all async DB calls.
