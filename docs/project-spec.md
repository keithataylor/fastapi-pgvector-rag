# Project Specification

## 1. Project objective

Build a complete, runnable FastAPI retrieval-augmented generation backend that:

* ingests PDF and UTF-8 text documents;
* chunks and embeds their content;
* stores document chunks and embeddings in PostgreSQL using pgvector;
* retrieves relevant chunks through cosine similarity;
* sends the retrieved context to an LLM through LangChain;
* exposes a `/chat` endpoint returning both the answer and the source chunks used.

The service must run locally through Docker Compose and be reproducible from a fresh clone using the README.

## 2. Required technology

* Python 3.11 or later
* FastAPI
* PostgreSQL
* pgvector
* LangChain
* OpenAI `text-embedding-3-small` embeddings at 1536 dimensions
* OpenAI-compatible chat model
* Docker and Docker Compose
* Pydantic request and response models
* Environment-variable configuration

## 3. Functional requirements

### Document ingestion

Provide a command-line ingestion script that:

* reads supported files from a configured folder;
* supports `.pdf` and `.txt` files;
* extracts text from each document;
* splits the text into chunks;
* generates an embedding for each chunk;
* stores document metadata, chunk text and embeddings in PostgreSQL;
* can be run repeatedly without creating duplicate documents or chunks.

### Chat endpoint

Provide:

```http
POST /chat
```

The endpoint must:

1. validate the incoming question;
2. generate an embedding for the question;
3. query pgvector using cosine similarity;
4. retrieve the top matching chunks;
5. pass the question and retrieved context to an LLM through LangChain;
6. return a structured JSON response containing:

   * the generated answer;
   * the source chunks used to generate it.

### Configuration

Configuration must be read from environment variables, including:

* database connection URL;
* OpenAI API key;
* chat model;
* document folder;
* chunking settings;
* retrieval `top_k`.

No secrets may be committed to Git.

The embedding model and embedding dimension are fixed project-schema decisions:
`text-embedding-3-small` at 1536 dimensions. They must not be freely changed
through runtime configuration. Changing either requires a schema migration,
re-embedding all stored chunks, and rebuilding the vector index.

## 4. Working assumptions

These assumptions stand in for client clarification and may be revised before implementation.

* The service is single-tenant.
* Authentication is not required.
* PDFs must contain extractable text; OCR is not required.
* Ingestion runs synchronously through a command-line script.
* `/chat` is stateless; conversation history is not required.
* Duplicate detection is based on document content rather than filename alone.
* Re-ingesting an unchanged document must not create duplicate records.
* A changed document is treated as a new document version unless a later requirement defines replacement behaviour.
* Source chunks should include enough metadata to identify the original document and chunk location.
* Local Docker Compose delivery is required; cloud deployment is not part of this project.
* Basic automated tests are required even though the original brief does not state them explicitly.
* Embeddings use the fixed `text-embedding-3-small` model at 1536 dimensions.
* Cosine similarity uses an HNSW index with the pgvector cosine operator class.
* A future embedding-model or embedding-dimension change requires a schema migration,
  re-embedding all stored chunks, and rebuilding the vector index.

## 5. Proposed API contract

### Request

```json
{
  "question": "What is the cancellation policy?",
  "top_k": 5
}
```

### Response

```json
{
  "answer": "The agreement may be cancelled with 30 days' written notice.",
  "sources": [
    {
      "document_id": "uuid",
      "filename": "agreement.pdf",
      "page_number": 7,
      "chunk_index": 18,
      "text": "Either party may terminate..."
    }
  ]
}
```

`top_k` determines how many of the most relevant chunks are retrieved.

## 6. Proposed architecture

The system will contain:

* a FastAPI application;
* a command-line document-ingestion script;
* document extraction and chunking services;
* LangChain-based embedding and LLM integrations using the fixed
  `text-embedding-3-small` / 1536-dimension embedding baseline;
* SQLAlchemy models and database access;
* Alembic migrations;
* PostgreSQL with the pgvector extension;
* an HNSW pgvector cosine index and a vector similarity retrieval service;
* a Docker Compose PostgreSQL/pgvector service introduced with the database
  foundation, then an API service added to complete the multi-service workflow;
* automated unit and integration tests.

### Database and migration architecture

Slice 2 establishes database environment configuration, the permanent local
PostgreSQL/pgvector Compose service, SQLAlchemy/Alembic infrastructure, and a
migration that enables the pgvector extension. Slice 3 owns the document and
chunk schema, the fixed `vector(1536)` embedding column, and the HNSW cosine
index migration. Slice 6 owns retrieval behaviour and query-plan validation.

The HNSW index uses pgvector's cosine operator class for the cosine similarity
queries required by `/chat`.

### Ingestion flow

```text
Document folder
→ PDF/text extraction
→ chunking
→ fixed `text-embedding-3-small` embedding generation (1536 dimensions)
→ PostgreSQL/pgvector persistence
```

### Chat flow

```text
Question
→ fixed `text-embedding-3-small` question embedding (1536 dimensions)
→ cosine similarity search
→ top-k chunks
→ LangChain LLM call
→ answer and source chunks
```

## 7. Data requirements

At minimum, the database must store:

### Documents

* unique identifier;
* original filename;
* content checksum;
* media type;
* ingestion timestamp.

### Chunks

* unique identifier;
* parent document identifier;
* chunk index;
* page number where available;
* chunk text;
* embedding vector stored as `vector(1536)`.

The database schema must prevent duplicate storage of unchanged document content.
It must also define an HNSW index over chunk embeddings using pgvector's cosine
operator class.

## 8. Error handling

The service must handle expected failures explicitly, including:

* unsupported file types;
* unreadable or empty documents;
* invalid UTF-8 text files;
* PDFs with no extractable text;
* missing configuration;
* database connection failures;
* embedding or LLM provider failures;
* invalid `/chat` requests;
* attempts to query an empty document store.

Errors must not expose API keys, credentials or internal stack traces through API responses.

## 9. Testing requirements

Automated tests must cover at least:

* text extraction;
* PDF extraction;
* chunking behaviour;
* duplicate-ingestion behaviour;
* persistence of documents, chunks and embeddings;
* migration upgrade, downgrade, and re-upgrade behaviour, including verification
  that the pgvector extension is enabled;
* the fixed embedding schema: a 1536-dimension vector column and its HNSW cosine
  index;
* cosine vector retrieval ordering and query-plan validation;
* `/chat` request validation;
* `/chat` response structure;
* provider failures through mocked or fake integrations.

Ordinary automated tests must not make paid OpenAI requests.

A separate manual smoke test may verify the complete real-provider workflow.
It must use the fixed embedding model and verify that provider responses conform
to the 1536-dimension schema before persistence.

## 10. Delivery requirements

The repository must contain:

* application source code;
* database migrations;
* ingestion script;
* Dockerfile;
* Docker Compose configuration;
* `.env.example`;
* sample documents;
* automated tests;
* README setup and usage instructions.

The README must document:

* prerequisites;
* environment configuration;
* Docker startup;
* database migration;
* ingestion command;
* `/chat` example request;
* test and lint commands;
* shutdown and cleanup.

The database Compose service and database-related `.env.example` settings are
introduced in Slice 2 so that migrations can be run and validated locally. The
Dockerfile, API service, and completed multi-service Compose workflow are added
in Slice 8.

## 11. Acceptance criteria

The project is complete when:

* a fresh clone can be configured and started using the README;
* `docker compose up --build` starts PostgreSQL/pgvector and the FastAPI service;
* the database migration path can be applied, reverted, and reapplied against the
  local PostgreSQL/pgvector service;
* the ingestion script processes the supplied PDF and text sample documents;
* running ingestion again does not create duplicate records;
* `/chat` returns a relevant answer for a question about the sample documents;
* `/chat` returns the source chunks used;
* configuration and secrets are loaded from environment variables;
* automated tests and linting pass;
* no placeholder implementation remains in the required execution path.

## 12. Implementation slices

1. Python and FastAPI runtime baseline
2. Database configuration; the permanent PostgreSQL/pgvector Compose service;
   database `.env.example` settings; SQLAlchemy/Alembic infrastructure; the
   initial pgvector-extension migration; and migration validation
3. Document and chunk schema; the fixed `vector(1536)` embedding column; the
   HNSW cosine index and its migrations; and persistence
4. PDF/text extraction and chunking
5. Embedding integration and idempotent ingestion
6. Cosine retrieval behaviour and query-plan validation
7. LangChain-backed `/chat`
8. API containerisation and completion of the multi-service Compose workflow
9. Remaining tests, sample documents, README and fresh-clone verification
