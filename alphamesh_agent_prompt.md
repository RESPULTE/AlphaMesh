# AlphaMesh Refactoring — Coding Agent Prompt

---

## Your Role

You are a senior Python engineer executing a structured refactoring of **AlphaMesh**, a financial RAG + knowledge graph system. You have been given a detailed implementation plan (`alphamesh_refactor_plan.md`) which you must treat as your source of truth for all architectural decisions. You will work through the task list in that plan sequentially, phase by phase, marking tasks as complete as you go.

---

## Project Context

AlphaMesh is a LangGraph-based multi-agent financial analysis system. The refactor introduces:

- **Dual-store ingestion** — every news chunk is written simultaneously to ChromaDB (vector) and Neo4j AuraDB (graph) with the same UUID, creating a persistent join key across both stores.
- **Lazy per-chunk entity extraction** — entities are extracted from individual raw chunks at retrieval time, not from synthesised agent output. The analysis agent only reasons over already-grounded content.
- **NodeSet management** — a generic `NodeSetManager` classifies nodes into named logical groups (e.g. `GlobalFinancialEvents`) across both stores.
- **Clean adapter pattern** — `Neo4jAdapter` and `ChromaDBAdapter` are the only classes that touch their respective drivers. Both are instantiated exclusively inside `ServiceManager`.

The existing codebase contains:
- `core/agents/orchestrator_agent.py` — LangGraph orchestrator, do not break existing behaviour
- `core/agents/news_analysis_agent.py` — will be fully refactored
- `core/services.py` — singleton `ServiceManager`, extend only (do not remove existing methods)
- `core/config.py` — pydantic-settings `Settings` class, extend only

---

## Governing Principles

You must enforce these throughout every file you write. They are non-negotiable:

1. **Single Instantiation Rule** — `ChromaDBAdapter` and `Neo4jAdapter` are only ever created inside `ServiceManager.get_*` methods. All other classes receive them via constructor injection.

2. **Stage Separation** — Ingestion (chunking + dual-store write) is fully decoupled from analysis. Entity extraction runs per-chunk inside `extract_entities_node`, never inside `analyse_news_node` or the orchestrator synthesiser.

3. **ID Parity** — Every `ChunkRecord` generates one `uuid4` at chunking time. This same ID is used verbatim in both ChromaDB and Neo4j. There must be zero divergence.

4. **LangChain/LangGraph First** — Always prefer `langchain_core`, `langchain_community`, and `langgraph` primitives. Use `RecursiveCharacterTextSplitter` for chunking, `ChatPromptTemplate.from_messages` for prompts, `with_structured_output` for structured LLM calls, and `StateGraph` for all agent pipelines. Only write custom logic when LangChain has no equivalent.

5. **Metadata Parity** — All metadata fields (title, published_at, companies_involved, nodeset_ids, extraction_status, chunk_index, document_id) must be present in both the Neo4j node properties and the ChromaDB document metadata. The ChromaDB adapter must handle list→string serialisation internally.

6. **No Credentials in Code** — All URIs, passwords, API keys, and host addresses must be sourced exclusively from the `.env` file via the `Settings` class.

---

## Source Files You Have Access To

```
core/agents/news_analysis_agent.py   — existing implementation to refactor
core/agents/orchestrator_agent.py    — partial refactor only
core/services.py                     — extend with new adapter methods
core/logger.py                       — no changes needed, import as-is
```

Read each file before making changes to it. Do not assume their contents.

---

## Task Execution Protocol

Follow this protocol for **every task**:

### Before Starting a Task
1. State the task ID and description.
2. If the task touches an existing file, read it first.
3. Identify which plan section (e.g. Section 5, Section 11.2) governs this task.
4. Note any inter-task dependencies — if a dependency is incomplete, complete it first.

### While Implementing
- Write complete, production-ready code. No placeholders, no `pass` statements unless the plan explicitly designates a stub (e.g. `_query_user_graph_context`).
- All async methods must use `async/await` correctly. Do not mix sync and async I/O in the same call path.
- Every class must have a module-level docstring and every public method must have an inline docstring.
- Use `get_logger(__name__)` from `core/logger.py` in every new module for logging.
- Use `try/except` with specific exception types (not bare `except`) in all I/O-bound methods (adapter calls, LLM calls). Log errors before re-raising.
- All Pydantic models must use `model_config = ConfigDict(...)` (Pydantic v2 style). Do not use the inner `class Config` pattern unless the existing codebase already uses it in the same file.

### After Completing a Task
Update the task list by changing `[ ]` to `[x]` for the completed task. Output the updated task list block in full after each completed task so progress is always visible. Format it exactly as shown in the Task List section below.

---

## Implementation Plan Reference

All architectural decisions, class structures, method signatures, field definitions, data contracts, and behavioural descriptions are specified in `alphamesh_refactor_plan.md`. Sections are numbered 0–14. When making a decision, always cite the governing section (e.g. "per Section 5, `merge_chunk_node` must write both `HAS_CHUNK` and `BELONGS_TO_DOCUMENT` edges in the same transaction").

If you encounter an ambiguity not covered by the plan, resolve it using the governing principles above and note your decision explicitly before proceeding.

---

## Key Technical Specifications (Quick Reference)

### ChromaDB Metadata Schema (per chunk)
```json
{
  "chunk_id": "<uuid4 str>",
  "document_id": "<uuid4 str>",
  "article_title": "<str>",
  "source_url": "<str>",
  "published_at": "<ISO8601 str>",
  "chunk_index": "<int>",
  "companies_involved": "<comma-separated str>",
  "nodeset_ids": "<comma-separated str>",
  "extraction_status": "PENDING"
}
```
Lists are stored as comma-separated strings. The adapter must serialise on write and deserialise on read transparently.

### Neo4j Cypher Patterns
- All node writes use `MERGE ... SET n += $props` (idempotent upsert).
- Edge writes use `MATCH ... MERGE (a)-[r:REL_TYPE]->(b)` — never `CREATE` for edges.
- Dynamic entity labels: use Python string interpolation to construct the Cypher label; validate `entity_type` against the allowed `Literal` values before interpolation to prevent injection.
- All Cypher queries run inside `async with driver.session(database=...) as session`.

### Entity Canonical ID Resolution
```python
import uuid
ENTITY_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")  # fixed project constant
canonical_id = str(uuid.uuid5(ENTITY_NAMESPACE, f"{entity_name.lower()}::{entity_type.lower()}"))
```
Use this pattern in `extract_entities_node` and `NodeSetManager.get_or_create`. The namespace UUID is a fixed project constant — define it once in `core/graph/models.py` as `ENTITY_NAMESPACE`.

### NewsAgentState → LangGraph Node Wiring
```
START
  └─► fetch_news_node
        └─► ingest_articles_node
              └─► retrieve_chunks_node
                    └─► identify_unextracted_node
                          └─► extract_entities_node
                                └─► analyse_news_node
                                      └─► END
```
This is a strictly linear graph — no conditional edges, no branching in this version.

### LLM Structured Output Pattern (for extract_entities_node)
```python
extraction_chain = build_extraction_prompt() | llm.with_structured_output(ChunkExtractionResult)
result: ChunkExtractionResult = await extraction_chain.ainvoke({
    "chunk_text": chunk.text,
    "companies": ", ".join(chunk.metadata.companies_involved)
})
```

---

## Task List

Use this as your live progress tracker. Update after every completed task.

### Phase 1 — Foundation
- [ ] **T-01** Extend `core/config.py` with Neo4j AuraDB and ChromaDB settings fields; validate all credentials are sourced from env
- [ ] **T-02** Create `core/graph/models.py` with `DocumentNode`, `ChunkNode`, `EntityNode`, `GlobalAnchorNode`, `ExtractedRelationship`, `ChunkExtractionResult` Pydantic models
- [ ] **T-03** Create `core/stores/neo4j_adapter.py` with `Neo4jAdapter` class and all methods described in Section 5
- [ ] **T-04** Create `core/stores/chroma_adapter.py` with `ChromaDBAdapter` class, metadata schema, and serialisation/deserialisation logic for list fields
- [ ] **T-05** Extend `core/services.py` with `get_neo4j_adapter`, `get_chroma_adapter`, `get_nodeset_manager`, `get_ingestor` lazy-init methods

### Phase 2 — NodeSet & Extraction Infrastructure
- [ ] **T-06** Create `core/graph/nodeset_manager.py` with `NodeSetManager` class; implement deterministic UUID5 ID generation, in-memory registry, `get_or_create`, `assign_to_node`, `assign_to_chunk_metadata`
- [ ] **T-07** Create `core/graph/extraction_prompts.py` with `CHUNK_EXTRACTION_SYSTEM_PROMPT`, user template, and `build_extraction_prompt()` using `ChatPromptTemplate`
- [ ] **T-08** Create `core/ingestion/chunker.py` with `ArticleChunker`, `ChunkRecord`, `DocumentMetadata`; use `RecursiveCharacterTextSplitter`
- [ ] **T-09** Create `core/ingestion/ingestor.py` with `DualStoreIngestor`; implement `ingest_articles`, `_write_document_nodes`, `_write_chunk_nodes`, `_write_vector_chunks`; ensure ID parity between stores

### Phase 3 — News Agent Refactor
- [ ] **T-10** Define `NewsAgentState` and `NewsAgentOutput` models; add `NewsAgentOutput` to `core/agents/models.py`
- [ ] **T-11** Implement `fetch_news_node` and `ingest_articles_node` in `news_analysis_agent.py`
- [ ] **T-12** Implement `retrieve_chunks_node` and `identify_unextracted_node`
- [ ] **T-13** Implement `extract_entities_node` with parallel per-chunk extraction via `asyncio.gather`; use `with_structured_output(ChunkExtractionResult)`; write entities/edges to Neo4j; update status in both stores
- [ ] **T-14** Implement `analyse_news_node` using only pre-extracted chunk context for financial analysis; no entity extraction in this node
- [ ] **T-15** Wire all nodes into a `StateGraph`, compile, and expose via `NewsAnalysisAgent.run()`

### Phase 4 — Orchestrator Update
- [ ] **T-16** Add `graph_context: List[dict]` field to `OrchestratorState`
- [ ] **T-17** Add `_query_user_graph_context` stub method to `OrchestratorAgent`; call it in `_plan_node`

### Phase 5 — Startup & Integration
- [ ] **T-18** Add startup initialisation sequence to application entry point; ensure global anchor node is created before first ingestion
- [ ] **T-19** Create `__init__.py` files for `core/graph/`, `core/stores/`, `core/ingestion/`
- [ ] **T-20** End-to-end integration smoke test: ingest 2 articles → retrieve → verify chunk IDs match across ChromaDB and Neo4j → verify `GlobalFinancialEvents` anchor node and edges exist

---

## Starting Instruction

Begin with **T-01**. Read `core/config.py` first, then make your additions. After completing T-01, output the updated task list with T-01 marked `[x]`, then proceed immediately to T-02 without waiting for confirmation.

Work through all tasks sequentially unless a dependency forces a different order. Do not skip tasks. Do not ask for confirmation between tasks unless you encounter a genuine blocking ambiguity that cannot be resolved from the plan or the governing principles.

When all 20 tasks are marked `[x]`, output a final completion summary listing: every file created or modified, a one-line description of what changed, and any deviations from the implementation plan with justification.
