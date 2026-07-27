# Project Purpose

Build a private, complete, client-ready FastAPI RAG backend using PostgreSQL with pgvector, LangChain/OpenAI, embeddings, PDF and text ingestion, vector retrieval, Docker Compose, and automated tests.

The governing project documents are:

1. `docs/client-brief.md` — the original requested requirements.
2. `docs/project-spec.md` — the current approved implementation contract, architecture, assumptions, scope, and acceptance criteria.

# Requirements Authority

* Read the relevant governing documents before planning or implementing a task.
* Treat `docs/project-spec.md` as the source of truth for implementation.
* Ensure implementation remains consistent with `docs/client-brief.md`.
* If the client brief, project specification, task prompt, existing code, or documentation conflict, stop and report the conflict before modifying files.
* Do not contradict, narrow, replace, or silently reinterpret a governing requirement.
* Present any proposed requirement change explicitly as a specification deviation, including:

  * the existing requirement;
  * the proposed replacement;
  * the reason;
  * the consequences;
  * the governing documents that would require amendment.
* Do not implement a specification deviation until it has been explicitly approved.
* Approval of a task prompt does not itself amend or override the governing documents.
* When an approved decision changes the project contract, update the relevant governing documentation in the same change or before implementation proceeds.

# Working Rules

* Inspect the current repository before making assumptions about files, structure, dependencies, configuration, or implemented behaviour.
* Keep each change limited to the explicitly approved task and its acceptance criteria.
* When asked to inspect, analyse, review, or plan, do not modify files.
* Do not implement behaviour outside the approved project scope.
* Do not introduce dependencies, architectural abstractions, compatibility restrictions, or unrelated refactoring outside the approved task.
* Add or update relevant tests in the same change as implemented behaviour.
* Run the relevant tests and quality checks before reporting completion.
* Do not commit, push, merge, rebase, or rewrite Git history unless explicitly instructed.
* Never add secrets, API keys, credentials, or populated environment files to Git.
* Report:

  * files changed;
  * implementation decisions;
  * checks executed;
  * check results;
  * requirement evidence;
  * remaining risks;
  * unverified assumptions.
* Do not claim completion unless every acceptance criterion for the task has been verified.

# Validation Commands

The following are the minimum common validation commands:

* Synchronise the environment with `uv sync`.
* Run tests with `uv run pytest`.
* Run linting with `uv run ruff check .`.
* Check formatting with `uv run ruff format --check .`.

Also run any task-specific checks required by the governing documents or affected functionality, including database, migration, Docker, integration, startup, API, security, or end-to-end checks where applicable.
