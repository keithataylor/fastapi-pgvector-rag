# FastAPI pgvector RAG

Python/FastAPI runtime baseline for the private RAG backend described in
`docs/project-spec.md`.

## Prerequisites

- Python 3.12
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
