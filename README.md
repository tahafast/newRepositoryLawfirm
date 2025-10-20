# Law Firm Chatbot

Law Firm Chatbot is a modern retrieval-augmented assistant that helps legal teams search firm knowledge, analyse documents, and draft professional pleadings in minutes. The project combines a FastAPI backend, a modular retrieval pipeline, and a lightweight web UI tailored for day-to-day law practice.

## Key Features

- **Smart document ingestion** - Upload PDF, DOCX, DOC, TXT, or RTF files. Text is cleaned, chunked, and indexed in Qdrant with page-aware metadata for precise citations.
- **Contextual legal Q&A** - Retrieve the most relevant knowledge, rerank passages, and run them through the configured LLM for grounded answers with references.
- **Document generation (DocGen)** - Detects drafting intents (affidavit, writ, plaint, legal notice, bail petition, and more), gathers required facts, and produces polished HTML pleadings ready to copy.
- **Conversation memory** - Stores each exchange in SQLite (or PostgreSQL/Redis if configured) so conversations resume with prior context, titles, and document state.
- **Web experience included** - `chat.html` for the assistant, `ingest.html` for indexing, both available at `/ui/` when the API serves static assets.
- **Operational guardrails** - Adaptive retrieval strategies, intent classification, grounding checks, and extensive logging to keep outputs reliable.

## Architecture Overview

- **FastAPI service** (`app/main.py`) wires application modules, middleware, and optional static hosting.
- **RAG orchestration** (`app/modules/lawfirmchatbot/services/rag/rag_orchestrator.py`) coordinates query analysis, retrieval, reranking, and LLM prompting.
- **Document processing** (`services/ingestion`) extracts text with fallbacks, normalises Unicode, and assigns page numbers before vector indexing.
- **Vector store** uses Qdrant (cloud, docker, or embedded) via the official Python client.
- **LLM integration** leverages LangChain abstractions with provider routing (OpenAI by default) for both Q&A and document drafting models.
- **Stateful chat memory** persists conversations through SQLAlchemy and optional Redis buffering.

```
app/
  main.py
  modules/lawfirmchatbot/
    api/                  # REST endpoints and conversation APIs
    services/             # RAG, docgen, ingestion, memory helpers
    schema/               # Pydantic request/response models
core/                     # Configuration, logging, dependency wiring
frontend/                 # Static chat and ingest single-page apps
requirements.txt          # Python dependencies
docker-compose.yml        # Optional Qdrant + API runtime
```

## Getting Started

### 1. Prerequisites

- Python 3.10 or newer
- Docker (optional, required if you want to run Qdrant locally via compose)
- An API key for your chosen LLM provider (OpenAI by default)

### 2. Clone and install

```bash
git clone https://github.com/tahafast/newRepositoryLawfirm.git
cd newRepositoryLawfirm
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy env.template .env       # Windows
# cp env.template .env       # macOS/Linux
```

Edit `.env` (or define environment variables) with at least:

- `OPENAI_API_KEY` (or relevant provider keys)
- `QA_MODEL` / `DOCGEN_MODEL` choices
- Qdrant connection mode (`QDRANT_MODE`, `QDRANT_URL`, `QDRANT_API_KEY` if cloud)
- Optional: `DATABASE_URL`, `REDIS_URL`, `SERVE_FRONTEND`

### 4. Start Qdrant (local option)

```bash
docker compose up qdrant -d
```

You can also point `QDRANT_URL` at a managed instance or embed Qdrant by setting `QDRANT_MODE=embedded`.

### 5. Launch the API

```bash
uvicorn app.main:app --reload
```

By default the service listens on `http://127.0.0.1:8000`. When `SERVE_FRONTEND=1` (default), the static UI is available under:

- Chat UI: `http://127.0.0.1:8000/ui/chat.html`
- Document ingestion UI: `http://127.0.0.1:8000/ui/ingest.html`
- OpenAPI docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/healthz`

### 6. Full stack with Docker Compose (optional)

To build the API container alongside Qdrant:

```bash
docker compose up --build
```

Set your `OPENAI_API_KEY` (or other provider keys) in the environment before running the command.

## Usage Highlights

1. **Index documents**
   - Visit `/ui/ingest.html` or call `POST /api/v1/lawfirm/upload-document` with multipart form data.
   - Supported formats: PDF, DOCX, DOC, TXT, RTF. The service normalises text and stores per-page metadata.

2. **Chat with the assistant**
   - Open `/ui/chat.html` to start a conversation. The UI preserves conversation history, supports doc generation previews, and offers a one-click copy button for drafted pleadings.
   - Conversations can also be managed through the REST endpoints:
     - `POST /api/v1/lawfirm/conversations/new`
     - `GET /api/v1/lawfirm/conversations`
     - `GET /api/v1/lawfirm/conversations/{conversation_id}/messages`
     - `DELETE /api/v1/lawfirm/conversations/{conversation_id}`

3. **Document generation**
   - The system detects doc drafting intents ("generate an affidavit...") and routes the request through the DocGen manager.
   - Gathered facts are combined with precedent snippets and rendered as HTML for instant download or copy.

## API Summary

| Method | Endpoint                                             | Purpose                                      |
|--------|------------------------------------------------------|----------------------------------------------|
| POST   | `/api/v1/lawfirm/upload-document`                    | Upload one or more documents for indexing    |
| POST   | `/api/v1/lawfirm/query`                              | Submit a question to the RAG orchestrator    |
| GET    | `/api/v1/lawfirm/docs/indexed/samples?document=...`  | Preview stored vector chunks by document     |
| POST   | `/api/v1/lawfirm/conversations/new`                  | Create a new conversation                    |
| GET    | `/api/v1/lawfirm/conversations`                      | List conversations for a user                |
| GET    | `/api/v1/lawfirm/conversations/{id}/messages`        | Retrieve recent messages for a conversation  |
| DELETE | `/api/v1/lawfirm/conversations/{id}`                 | Remove a conversation and its messages       |

## Configuration Reference

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | LLM provider key used for Q&A and document drafting. |
| `QA_MODEL`, `DOCGEN_MODEL` | Model names for retrieval answers and DocGen flows. |
| `QDRANT_MODE` | `docker`, `cloud`, or `embedded` vector store mode. |
| `QDRANT_URL`, `QDRANT_API_KEY` | Connection details when using Qdrant cloud or remote. |
| `DATABASE_URL` | SQLAlchemy DSN for chat memory (SQLite by default). |
| `REDIS_URL` | Optional Redis instance for buffer/caching. |
| `SERVE_FRONTEND` | Serve the static SPA from FastAPI (`1` by default). |
| `LOG_LEVEL`, `LOG_QDRANT_HTTP` | Logging verbosity controls. |
| `WEB_SEARCH_ENABLED`, `TAVILY_API_KEY` | Toggle web fallback search support. |

Refer to `env.template` for the complete list and sensible defaults.

## Development Notes

- Run the API with `uvicorn app.main:app --reload` for auto-reload during development.
- Set `DEBUG_RAG=1` (alongside other logging flags) to inspect retrieval and LLM timing in detail.
- Unit and integration tests can be run with `python -m pytest` (see `test_intelligent_routing.py` for examples).
- The SQLite memory database (`brag_ai.sqlite`) is created automatically; delete it between sessions if you need a clean slate.

## Troubleshooting

- **No results returned** - Confirm documents were successfully ingested and the Qdrant collection exists. Use `/api/v1/lawfirm/docs/indexed/samples` to verify stored chunks.
- **LLM errors** - Make sure the API key is valid and the selected models are available to your account.
- **DocGen missing fields** - The assistant may prompt for required fields; supply the requested facts in follow-up turns.
- **Frontend not loading** - Ensure `SERVE_FRONTEND=1` or open the static files directly from the `frontend/` directory using a local server.

## License

This project is provided for internal use by the maintainers. Contact the repository owner for licensing or redistribution questions.
