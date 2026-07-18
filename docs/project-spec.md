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
* OpenAI `text-embedding-3-small`
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
* embedding model;
* chat model;
* document folder;
* chunking settings;
* retrieval `top_k`.

No secrets may be committed to Git.

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
* LangChain-based embedding and LLM integrations;
* SQLAlchemy models and database access;
* Alembic migrations;
* PostgreSQL with the pgvector extension;
* a vector similarity retrieval service;
* Docker Compose services for the API and database;
* automated unit and integration tests.

### Ingestion flow

```text
Document folder
→ PDF/text extraction
→ chunking
→ embedding generation
→ PostgreSQL/pgvector persistence
```

### Chat flow

```text
Question
→ question embedding
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
* embedding vector.

The database schema must prevent duplicate storage of unchanged document content.

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
* vector retrieval ordering;
* `/chat` request validation;
* `/chat` response structure;
* provider failures through mocked or fake integrations.

Ordinary automated tests must not make paid OpenAI requests.

A separate manual smoke test may verify the complete real-provider workflow.

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

## 11. Acceptance criteria

The project is complete when:

* a fresh clone can be configured and started using the README;
* `docker compose up --build` starts PostgreSQL/pgvector and the FastAPI service;
* the ingestion script processes the supplied PDF and text sample documents;
* running ingestion again does not create duplicate records;
* `/chat` returns a relevant answer for a question about the sample documents;
* `/chat` returns the source chunks used;
* configuration and secrets are loaded from environment variables;
* automated tests and linting pass;
* no placeholder implementation remains in the required execution path.

## 12. Implementation slices

1. Python and FastAPI runtime baseline
2. PostgreSQL, pgvector and database migrations
3. Document models and persistence
4. PDF/text extraction and chunking
5. Embedding integration and idempotent ingestion
6. Cosine vector retrieval
7. LangChain-backed `/chat`
8. Docker Compose and environment configuration
9. Tests, README and fresh-clone verification
