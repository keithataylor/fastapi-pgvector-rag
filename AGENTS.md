# Project Purpose

Build a private, complete, client-ready FastAPI RAG backend using PostgreSQL with pgvector, LangChain/OpenAI, embeddings, PDF and text ingestion, vector retrieval, Docker Compose, and automated tests.

This repository is also being used to establish a disciplined professional Codex-assisted development workflow.

The approved functional scope, assumptions, architecture, and acceptance criteria are defined in `docs/project-spec.md`.

# Working Rules

- Treat `docs/project-spec.md` as the authoritative project contract. Do not implement behaviour outside it without explicit approval.
- Inspect the current repository before making assumptions about files, structure, dependencies, or implemented behaviour.
- Keep each change limited to the explicitly approved task and acceptance criteria.
- When asked to plan or inspect, do not modify files.
- Do not introduce dependencies, architectural abstractions, or unrelated refactoring outside the approved task.
- Add or update relevant tests in the same change as implemented behaviour.
- Run the relevant tests and quality checks before reporting completion.
- Do not commit, push, merge, rebase, or rewrite Git history unless explicitly instructed.
- Never add secrets, API keys, credentials, or populated environment files to Git.
- Report files changed, checks executed, results, and any remaining risks or unverified assumptions.
- Do not claim completion unless every stated acceptance criterion has been verified.

# Validation Commands

- Synchronise the environment with `uv sync`.
- Run tests with `uv run pytest`.
- Run linting with `uv run ruff check .`.
- Check formatting with `uv run ruff format --check .`.
