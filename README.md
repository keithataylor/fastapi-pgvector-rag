# FastAPI pgvector RAG

A private FastAPI retrieval-augmented generation backend using PostgreSQL,
pgvector, LangChain, and OpenAI. It ingests PDF and UTF-8 text documents and
returns grounded answers plus the source chunks used.

## Prerequisites

- Docker Desktop with Docker Compose
- Python 3.11 or later and [uv](https://docs.astral.sh/uv/)
- An OpenAI API key and compatible chat model only to ingest documents or call
  a provider-backed `/chat` endpoint

## Configure a fresh clone

Install the locked development environment:

```powershell
uv sync
```

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

Set a non-default `POSTGRES_PASSWORD` in `.env` and use the same password in
`DATABASE_URL` and `MIGRATION_TEST_DATABASE_URL`. The latter must name the
separate disposable `rag_migration_test` database. Set `OPENAI_API_KEY` and
`CHAT_MODEL` only for real ingestion and grounded-chat use.

Embeddings are fixed to OpenAI `text-embedding-3-small` at 1536 dimensions.
Changing either requires a schema migration, re-embedding stored chunks, and
rebuilding the vector index.

## Start the stack

Run these commands in order. Do not initially start all services before the
database migrations have been applied.

```powershell
docker compose --env-file .env build api
docker compose --env-file .env up -d --wait db
docker compose --env-file .env run --rm api alembic upgrade head
docker compose --env-file .env up -d --wait api
Invoke-RestMethod http://127.0.0.1:8000/health
```

The final command returns `@{status=ok}`. The API container reads the committed
`documents/` directory through a read-only `/app/documents` mount.

## Ingest the supplied documents

With `OPENAI_API_KEY` and `CHAT_MODEL` configured in `.env`, ingest the safe
fictional samples:

```powershell
docker compose --env-file .env run --rm api python -m app.ingest
```

The first run reports both documents as `ingested`. Run the same command again;
it reports both as `skipped`, proving checksum-based duplicate handling.

`OpenAIEmbedder` validates each provider result against the fixed 1536-dimension
contract before ingestion persists a document or chunk. A provider response
with the wrong dimension aborts that document's ingestion.

## Ask a grounded question

```powershell
$body = @{ question = "How much notice is required to cancel the Northstar service?"; top_k = 2 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType "application/json" -Body $body
```

The answer should state that 30 days' written notice is required. Every source
in `sources` contains `document_id`, `filename`, `page_number`, `chunk_index`,
and `text`; the PDF source has page number `1` and the text source has no page
number.

Without provider configuration, a valid `/chat` request returns the controlled
`503` response `{"detail":"Service configuration is unavailable."}`.

## Validate

Run the ordinary automated and quality checks:

```powershell
$env:RUN_MIGRATION_INTEGRATION = "1"
uv run --env-file .env pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
Remove-Item Env:RUN_MIGRATION_INTEGRATION
```

Automated tests use mocked or fake provider integrations and do not make paid
OpenAI requests. The migration integration suite validates upgrade, downgrade,
and re-upgrade lifecycle behaviour; those rollback operations are not part of
normal operator setup.

## Final real-provider acceptance validation

Before delivery, a developer performs this one-time validation with their own
temporary OpenAI API key and configured chat model: ingest the committed samples
with real `text-embedding-3-small`, confirm the 1536-dimension validation occurs
before persistence, re-ingest to confirm skips, and call `/chat` to verify the
grounded answer and all five source fields (`document_id`, `filename`,
`page_number`, `chunk_index`, and `text`). Do not print, record, commit, or
place credentials in generated files; remove them from the local environment
afterwards. Clients provide their own credentials only when operating or
deploying the service.

## Shutdown and cleanup

Stop the stack while retaining the database volume:

```powershell
docker compose --env-file .env down
```

To permanently delete local PostgreSQL data, use the destructive command below:

```powershell
docker compose --env-file .env down -v
```


## Proprietary Notice

Copyright © 2026 Keith A. Taylor. All rights reserved. This repository contains proprietary software developed independently before any client engagement and is provided solely for evaluation and demonstration purposes. No permission is granted to copy, modify, distribute, publish, sublicense, or use the software in whole or in part without the copyright owner’s prior written consent.