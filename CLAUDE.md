# Project: AlphaMesh

AI-powered financial research assistant. A Python backend stack that orchestrates multi-agent analysis (fundamental + news), persists knowledge into a Neo4j graph + ChromaDB vector dual-store, and streams results to the frontend via Server-Sent Events over a FastAPI REST API.

---

## Tech Stack

- **Runtime**: Python 3.11+, async/await throughout
- **API**: FastAPI with lifespan-managed singletons; all endpoints under `/api/v1`
- **LLM**: Google Gemini via `langchain_google_genai` (`ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`); model names from `.env`
- **Graph DB**: Neo4j (Cypher queries); adapter in `core/memory/stores/neo4j_adapter.py`
- **Vector DB**: ChromaDB (local persistent); two collections — `news_chunks` and `entity_nodes`; adapter in `core/memory/stores/chroma_adapter.py`
- **Config**: `pydantic_settings.BaseSettings`; single `settings` singleton in `core/config.py`, reads from `.env`
- **Conversation persistence**: SQLite via `api/persistence/sqlite_adapter.py`
- **Streaming**: SSE (`text/event-stream`) via `api/sinks/sse_sink.py` and `api/routers/stream.py`

---

## Code Style

- **All new code must be async** — use `async def` and `await`, never blocking I/O in a coroutine
- Type hints on every function signature; use `from __future__ import annotations`
- `from __future__ import annotations` at the top of every file
- Prefer explicit imports over star imports
- Docstrings on public classes and non-trivial methods
- No `print()` in production paths — use `logging.getLogger(__name__)`
- Constants and config values always come from `core.config.settings`, never hardcoded

---

## Engineering Principles *(top priority — apply to every change)*

### 1 — Understand before proposing
**Before suggesting any architectural change or new component, read and understand the existing code flow end-to-end.** Trace the full call chain from request entry point to the relevant service and back. If unchanged existing code already satisfies the requirement (even partially), state that explicitly and propose augmenting it rather than replacing it.

### 2 — DRY: reuse and modify, don't duplicate
- **Search before you write.** Check `core/agents/`, `core/memory/`, `api/services/`, and `core/services.py` for existing functionality before adding anything new.
- If a utility, helper, or service *almost* fits, prefer **extending or parameterising** it over creating a parallel implementation.
- Shared logic belongs in a shared module:
  - Common agent utilities → `core/agents/utils.py`
  - Shared prompt templates → `core/agents/prompts/` or `core/memory/prompts.py`
  - Shared Pydantic models → the relevant `models/` folder in `core/agents/models/` or `core/memory/graph/models.py`
  - Cross-cutting graph helpers → `core/memory/graph/utils.py`

### 3 — Modularity and separation of concerns
Each layer has a single, clearly owned responsibility — do not blur these boundaries:

| Layer | Owns | Must NOT |
|---|---|---|
| `api/routers/` | HTTP request/response shape, validation | Contain business logic or call services directly |
| `api/services/` | API-layer orchestration (running agents, storing convos, broadcasting events) | Know about Neo4j, ChromaDB, or LLM internals |
| `api/dependencies.py` | FastAPI `Depends()` wiring | Contain logic beyond forwarding from `app.state` |
| `core/agents/` | LLM reasoning, tool calls, financial analysis | Directly import `neo4j_adapter` or `chroma_adapter` — go through `service_manager` |
| `core/memory/stores/` | Raw DB I/O (Cypher, ChromaDB queries) | Call LLMs or interpret results |
| `core/memory/graph/` | Graph-layer entities, queues, dedup | Know about HTTP or FastAPI concerns |
| `core/services.py` | Singleton wiring and lazy init | Implement any business or domain logic itself |
| `core/config.py` | Configuration only | Contain any runtime logic |

- Keep functions **small and single-purpose** — if a function is doing two logically distinct things, split it.
- New files should have a single stated responsibility in their module docstring.
- Side-effects (DB writes, network calls) must be isolated from pure logic — pure transformations (parsing, scoring, formatting) should be separately callable.

---

## Commands

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
pytest

# Run a specific test file
pytest tests/test_file.py -v

# Run tests matching a keyword
pytest -k "retriever" -v
```

---

## Repository Layout

```
AlphaMesh/
├── api/                         # FastAPI layer
│   ├── main.py                  # App factory + lifespan (startup/shutdown)
│   ├── dependencies.py          # Depends() providers (get_broadcaster, get_store, get_runner)
│   ├── routers/
│   │   ├── chat.py              # POST /api/v1/chat — submit analysis request
│   │   ├── stream.py            # GET  /api/v1/stream/{request_id} — SSE event stream
│   │   ├── conversations.py     # GET/DELETE conversation history
│   │   └── health.py            # GET /api/v1/health
│   ├── services/
│   │   ├── analysis_runner.py   # Orchestrates agent pipeline; emits events to broadcaster
│   │   ├── conversation_store.py# Persists/retrieves conversation records
│   │   └── event_broadcaster.py # Fan-out SSE events to waiting clients
│   ├── sinks/
│   │   └── sse_sink.py          # SSE response stream builder
│   ├── models/                  # Pydantic request/response models for API endpoints
│   └── persistence/
│       └── sqlite_adapter.py    # SQLite-backed ConversationAdapter
│
├── core/
│   ├── config.py                # Settings (pydantic_settings); single `settings` singleton
│   ├── services.py              # ServiceManager — lazy singleton factory for ALL services
│   ├── event_queue.py           # Async event queue shared between agents and API layer
│   ├── logger.py                # Logging helpers
│   │
│   ├── agents/
│   │   ├── base_agent.py        # Abstract base; defines agent interface
│   │   ├── orchestrator_agent.py# Top-level agent; routes tasks to sub-agents
│   │   ├── fundamental_analysis_agent.py # Fundamental/financial data analysis
│   │   ├── news_analysis_agent.py        # News ingestion + sentiment analysis
│   │   ├── financial_db.py      # yfinance / data provider wrapper
│   │   ├── financial_tools.py   # LangChain tools used by agents
│   │   ├── ticker_validation.py # Validates tickers against graph + vector store
│   │   ├── news_fetcher.py      # NewsAPI client wrapper
│   │   ├── data_prep.py         # Data cleaning / normalisation utilities
│   │   ├── utils.py             # Misc agent utilities
│   │   ├── models/              # Pydantic models for agent I/O (results, signals, etc.)
│   │   └── prompts/             # All LLM system/user prompt strings
│   │
│   └── memory/
│       ├── prompts.py           # Extraction-related prompt templates
│       ├── user_context_service.py   # Reads/writes user investment context from graph
│       ├── user_signal_writeback.py  # Writes user signals back into Neo4j
│       │
│       ├── stores/
│       │   ├── neo4j_adapter.py      # Neo4j async driver; all Cypher queries go here
│       │   └── chroma_adapter.py     # ChromaDB async wrapper; embed + upsert + query
│       │
│       ├── graph/
│       │   ├── graph_queue.py        # GraphQueueManager — durable async write queue (SQLite WAL)
│       │   ├── entity_resolver.py    # Fuzzy + semantic dedup of extracted entities
│       │   ├── nodeset_manager.py    # Bootstrap and manage default NodeSets in Neo4j
│       │   ├── relationship_extractor.py # LLM-based relationship extraction from text
│       │   ├── subgraph_service.py   # On-demand subgraph extraction + caching
│       │   ├── models.py             # Graph-layer Pydantic models (Entity, Relationship, etc.)
│       │   ├── extraction_prompts.py # Prompts for entity/relation extraction
│       │   └── utils.py             # Graph utility helpers
│       │
│       ├── ingestion/
│       │   ├── ingestor.py           # DualStoreIngestor: chunk → embed → graph + vector upsert
│       │   └── chunker.py            # ArticleChunker (size + overlap from settings)
│       │
│       └── retrieval/
│           ├── dual_store_retriever.py   # DualStoreRetriever: vector + graph retrieval, graph traversal
│           └── reranker.py              # CompositePrefilter + TwoStageReranker (Jina cross-encoder)
│
├── tests/                       # pytest test suite
├── data/                        # Runtime data (chroma_db/, conversations.db, portfolio.json)
├── .env                         # Environment variables (NEVER commit)
└── pytest.ini
```

---

## Architecture: How It Fits Together

### Startup Flow (`api/main.py` lifespan)
1. `service_manager.startup()` → initialises default NodeSets in Neo4j, starts `GraphQueueManager`
2. SQLite adapter initialised; `ConversationStore`, `EventBroadcaster`, `AnalysisRunner` created
3. All three singletons attached to `app.state`; routers access them via `Depends()` in `api/dependencies.py`

### Request Flow (chat → stream)
```
POST /api/v1/chat
    └─► AnalysisRunner.run()
            └─► OrchestratorAgent
                    ├─► FundamentalAnalysisAgent  (yfinance + LLM)
                    └─► NewsAnalysisAgent          (NewsAPI → ingest → retrieve)
                                    │
                                    ▼
                        Events emitted to EventBroadcaster
                                    │
                                    ▼
GET /api/v1/stream/{request_id}  (SSE)
    └─► sse_sink.py streams events to client
```

### Memory Pipeline (ingestion → retrieval)
```
Raw text (article/financial data)
    └─► DualStoreIngestor
            ├─► ArticleChunker          (split into ~512-token chunks)
            ├─► ChromaDBAdapter         (embed + upsert to news_chunks)
            └─► GraphQueueManager       (async queue → entity resolve → Neo4j upsert)
                    └─► EntityResolver  (fuzzy + semantic dedup, ~85% / 0.85 thresholds)

Query
    └─► DualStoreRetriever
            ├─► ChromaDBAdapter.query() (vector similarity)
            ├─► Neo4jAdapter (graph traversal, up to RETRIEVER_MAX_ITERATIONS hops)
            └─► CompositePrefilter → TwoStageReranker (Jina cross-encoder) → top-k results
```

---

## Service Singleton Pattern

`core/services.py` exports a **single** `service_manager` instance. All services are created lazily on first call and cached. **Never instantiate services directly** in agent or API code — always call via `service_manager`:

```python
from core.services import service_manager

neo4j   = service_manager.get_neo4j_adapter()
chroma  = service_manager.get_chroma_adapter()
llm     = service_manager.get_agent(temperature=0.0)
ingestor = service_manager.get_ingestor()
retriever = service_manager.get_retriever()
```

---

## Key Configuration Variables (`.env`)

| Variable | Purpose |
|---|---|
| `LLM_API_KEY` | Google Gemini API key (maps to `GOOGLE_API_KEY`) |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j connection |
| `NEO4J_DATABASE` | Neo4j database name (default: `neo4j`) |
| `CHROMA_PATH` | ChromaDB persistence directory (default: `./data/chroma_db`) |
| `JINA_API_KEY` | Jina reranker (optional; runs composite-only if unset) |
| `NEWSAPI_KEY` | NewsAPI key for news fetching |
| `LLM_MODEL` | Gemini model name (default: `gemini-2.5-flash-lite`) |
| `EMBEDDING_MODEL` | Embedding model (default: `gemini-embedding-001`) |
| `GRAPH_QUEUE_DB_PATH` | SQLite WAL path for graph task queue (default: `./data/graph_tasks.db`) |

---

## Critical Rules & Gotchas

- **NEVER commit `.env`** — it contains API keys
- **`GraphQueueManager` is stateful and async** — it must be started via `await service_manager.startup()` before any graph writes. Do not call `graph_queue.ingest()` before startup completes
- **Two Chroma collections**: `news_chunks` (article text) vs `entity_nodes` (resolved entities). Do not mix them — use `get_chroma_adapter()` for news and `get_entity_chroma_adapter()` for entities
- **`EntityResolver` does dedup** — fuzzy threshold 85.0, semantic threshold 0.85. These are tunable via `settings.EXTRACTION_FUZZY_THRESHOLD` / `EXTRACTION_SEMANTIC_THRESHOLD`
- **`TwoStageReranker`** = `CompositePrefilter` (stage 1, fast alpha/beta score) + Jina cross-encoder (stage 2, expensive). `DualStoreRetriever` uses only stage 1 (`prefilter=`); the orchestrator node uses stage 2 (`reranker=`)
- **SSE streaming**: events flow `agent → EventBroadcaster → sse_sink`. Do NOT write directly to the response in routers; use `broadcaster.publish(request_id, event)`
- **Conversation state** is in SQLite (`data/conversations.db`). Agent results that should be persisted must go through `ConversationStore`, not Neo4j
- **`data/` directory** is gitignored — all runtime state lives there

---

## Adding a New Agent

1. Subclass `BaseAgent` in `core/agents/base_agent.py`
2. Add a `get_my_agent()` method to `ServiceManager` in `core/services.py`
3. Wire it into `OrchestratorAgent` (`core/agents/orchestrator_agent.py`)
4. Prompt strings go in `core/agents/prompts/`; Pydantic result models go in `core/agents/models/`

## Adding a New API Endpoint

1. Create or extend a router in `api/routers/`
2. Use `Depends(get_broadcaster)`, `Depends(get_store)`, or `Depends(get_runner)` — never `app.state` directly
3. Register the router in `api/main.py` `create_app()`
4. Request/response Pydantic models go in `api/models/`

## Running Tests

```bash
pytest tests/ -v                  # full suite
pytest tests/ -k "ingest" -v      # filter by name
pytest tests/ --tb=short          # compact tracebacks
```

Tests requiring live Neo4j or ChromaDB are integration tests and may need `.env` populated.
