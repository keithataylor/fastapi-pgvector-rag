# Reference Client Brief

This is the original reference brief. It is preserved unchanged as the source client request. The budget and delivery expectation are not adopted as project constraints unless explicitly stated in the approved project specification.

Summary
need a single, production-quality retrieval-augmented (RAG) API endpoint built in Python. This is a one-time task, not an ongoing role.

You will deliver one FastAPI service that ingests a set of documents into a Postgres + pgvector store, then answers questions over them through a single /chat endpoint backed by an LLM.

What you will build

1. Ingestion script
   - Load a folder of provided documents (PDF and .txt).
   - Chunk the text, generate embeddings, and upsert them into PostgreSQL using pgvector.
   - Idempotent: re-running should not create duplicates.

2. /chat endpoint (FastAPI)
   - Runs cosine similarity search over pgvector to retrieve the top-k chunks.
   - Passes the retrieved context to an LLM via LangChain and returns a structured JSON response: the answer plus the source chunks used.

3. Basics done right
   - Pydantic request/response models.
   - Config via environment variables (DB URL, model/API key).
   - A README with setup and run instructions.
   - A docker-compose that spins up Postgres + pgvector so I can run it locally in one command.

Tech stack (required)

- Python 3.11+, FastAPI
- PostgreSQL + pgvector (HNSW or IVFFlat index)
- LangChain for the retrieval + LLM call
- Embeddings: OpenAI text-embedding-3-small (or an open model such as bge-m3, your choice)
- Docker / docker-compose

Acceptance criteria
- I can clone, run docker-compose up, run the ingestion script, and hit /chat successfully within 15 minutes using the README.
- /chat returns a relevant answer plus the source chunks for a question about the sample documents.
- No hardcoded secrets; keys read from env.



To apply

- Share one relevant RAG or FastAPI sample you have built.
- Confirm the scope and $150 fixed price.
- Give a realistic delivery time (I expect 2 to 4 days).

