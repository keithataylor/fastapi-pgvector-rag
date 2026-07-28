# FastAPI pgvector RAG

Python/FastAPI runtime baseline for the private RAG backend described in
`docs/project-spec.md`.

## Prerequisites

- Python 3.11 or later
- `uv`

## Local setup

Create the project virtual environment, install the declared dependencies, and
synchronise them with the committed lockfile:

```powershell
uv sync
```

## Run the API

```powershell
uv run uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health` and returns:

```json
{"status": "ok"}
```

## Slice 2 database setup

Docker Desktop is required to run the local PostgreSQL/pgvector database.

Create a local environment file and update `POSTGRES_PASSWORD` and the matching
password in `DATABASE_URL`:

```powershell
Copy-Item .env.example .env
```

Start the database and wait for it to become healthy:

```powershell
docker compose --env-file .env up -d --wait
```

Run the migration integration validation:

```powershell
$env:RUN_MIGRATION_INTEGRATION = "1"
uv run --env-file .env pytest -m migration_integration -rs
uv run --env-file .env alembic current
```

Stop the database when finished:

```powershell
docker compose --env-file .env down
```

The Compose API service and full README workflow are added in later slices.

## Validation

Run the tests:

```powershell
uv run pytest
```

Run linting:

```powershell
uv run ruff check .
```

Check formatting:

```powershell
uv run ruff format --check .
```
