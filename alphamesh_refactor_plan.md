# AlphaMesh Refactoring — Implementation Plan & Task List

> **Audience:** LLM coding agent  
> **Purpose:** Structured guide for refactoring AlphaMesh to support dual-store (ChromaDB + Neo4j AuraDB) ingestion, lazy per-chunk entity extraction, NodeSet management, and a clean two-stage news analysis pipeline.  
> **Convention:** No full code is provided — only structure, class/method signatures, data contracts, and behavioural descriptions.

---

## 0. Guiding Principles

1. **Single Instantiation Rule** — `ChromaDBAdapter` and `Neo4jAdapter` are only ever created inside `ServiceManager`. All other modules receive them via dependency injection.
2. **Stage Separation** — Ingestion (chunking + dual-store write) is fully decoupled from analysis (agent reasoning). Entity extraction happens at the chunk level during ingestion, never inside the analysis agent.
3. **ID Parity** — Every chunk gets one canonical `uuid4` ID that is the same in both ChromaDB and Neo4j. This is the join key.
4. **NodeSet as Metadata Contract** — NodeSets are not graph traversal constructs; they are metadata classifiers written to both stores, expressing which global entity a node belongs to.
5. **LangChain/LangGraph First** — Prefer `langchain_community`, `langchain_core`, and `langgraph` primitives. Only write custom classes when LangChain has no equivalent.

---

## 1. Target File Structure

```
core/
├── agents/
│   ├── base_agent.py               (existing — no change)
│   ├── fundamental_analysis_agent.py (existing — no change)
│   ├── models.py                   (extend with NewsAgentOutput)
│   ├── news_analysis_agent.py      (full refactor)
│   └── orchestrator_agent.py       (partial refactor — add graph query stub)
│
├── graph/
│   ├── __init__.py
│   ├── models.py                   (NEW — custom Pydantic graph node/edge models)
│   ├── nodeset_manager.py          (NEW — NodeSetManager class)
│   └── extraction_prompts.py       (NEW — system/user prompts for entity extraction)
│
├── stores/
│   ├── __init__.py
│   ├── neo4j_adapter.py            (NEW — Neo4jAdapter class)
│   └── chroma_adapter.py           (NEW — ChromaDBAdapter class)
│
├── ingestion/
│   ├── __init__.py
│   ├── chunker.py                  (NEW — ArticleChunker class)
│   └── ingestor.py                 (NEW — DualStoreIngestor class)
│
├── memory/
│   ├── conversation_writeback.py   (existing — no change)
│   ├── memory_system.py            (existing — no change)
│   └── prompts.py                  (existing — no change)
│
├── config.py                       (extend — add Neo4j + Chroma settings)
├── services.py                     (extend — register adapters and ingestor)
└── logger.py                       (existing — no change)
```

---

## 2. Config Extension (`core/config.py`)

**What to add to the existing `Settings` / `BaseSettings` class:**

```
NEO4J_URI: str
NEO4J_USERNAME: str
NEO4J_PASSWORD: str
NEO4J_DATABASE: str = "neo4j"

CHROMA_HOST: str = "localhost"
CHROMA_PORT: int = 8000
CHROMA_COLLECTION_NEWS: str = "news_chunks"

CHUNK_SIZE: int = 512          # tokens
CHUNK_OVERLAP: int = 64        # tokens

EXTRACTION_BATCH_SIZE: int = 6
EXTRACTION_MAX_CONCURRENCY: int = 3
```

All values must be sourced from the `.env` file via `pydantic-settings`. No defaults for credentials.

---

## 3. Graph Models (`core/graph/models.py`)

Define Pydantic models that represent the Neo4j graph schema. These are **not** ORM models — they are data contracts used by the adapter and ingestor.

### 3.1 Node Models

**`DocumentNode`**
- Fields: `id: str`, `title: str`, `source_url: str`, `published_at: datetime`, `ingested_at: datetime`, `companies_involved: List[str]`, `nodeset_ids: List[str]`
- No `full_text` field — the document node is a structural anchor only.
- Label in Neo4j: `Document`

**`ChunkNode`**
- Fields: `id: str` (same UUID as ChromaDB document ID), `text: str`, `chunk_index: int`, `document_id: str`, `companies_involved: List[str]`, `nodeset_ids: List[str]`, `extraction_status: Literal["PENDING", "EXTRACTED"]`
- Label in Neo4j: `Chunk`

**`EntityNode`**
- Fields: `id: str` (UUID5 canonical), `name: str`, `entity_type: Literal["Company", "Person", "MacroIndicator", "Event", "GeoPoliticalRegion", "Instrument"]`, `aliases: List[str]`, `nodeset_ids: List[str]`
- Label in Neo4j: dynamic, matches `entity_type`

**`GlobalAnchorNode`**
- Fields: `id: str` (deterministic, e.g. `uuid5(NAMESPACE, "GlobalFinancialEvents")`), `name: str = "Global Financial Events"`, `description: str`
- Label: `GlobalAnchor`
- This node must be created once at system initialisation (idempotent `MERGE`).

### 3.2 Edge Models

**`HAS_CHUNK`** — `(Document)-[:HAS_CHUNK]->(Chunk)`  
**`BELONGS_TO_DOCUMENT`** — `(Chunk)-[:BELONGS_TO_DOCUMENT]->(Document)` (reverse, for fast traversal)  
**`MENTIONS_ENTITY`** — `(Chunk)-[:MENTIONS_ENTITY {confidence: float}]->(Entity)`  
**`ANCHORED_TO`** — `(Document)-[:ANCHORED_TO]->(GlobalAnchor)`  
**`RELATED_TO`** — `(Entity)-[:RELATED_TO {relationship_type: str, source_chunk_id: str}]->(Entity)`  

### 3.3 Extracted Relationship Model (used during extraction only)

**`ExtractedRelationship`**
- Fields: `source_entity_local_id: str`, `target_entity_local_id: str`, `relationship_type: str`, `confidence: float`
- This is an intermediate Pydantic model used between the LLM extraction call and graph write — it is not persisted directly.

**`ChunkExtractionResult`**
- Fields: `chunk_id: str`, `entities: List[EntityNode]`, `relationships: List[ExtractedRelationship]`

**`BatchExtractionResult`**
- Fields: `results: List[ChunkExtractionResult]`

---

## 4. NodeSet Manager (`core/graph/nodeset_manager.py`)

### Class: `NodeSetManager`

**Purpose:** Manages the creation, registration, and lookup of NodeSets. A NodeSet is a named logical group (e.g. `"GlobalFinancialEvents"`, `"TechSector"`, `"TSMC_Articles"`) that acts as a metadata classifier for both graph nodes and vector store documents. The manager is generic — it handles any NodeSet type, not just financial ones.

**Constructor:** Receives `neo4j_adapter: Neo4jAdapter` via injection. Does not instantiate the adapter itself.

**State:** Maintains an in-memory registry `Dict[str, str]` mapping `nodeset_name → nodeset_id` (UUID5 deterministic). On first call to any method, it syncs the registry against the graph via a `MERGE` query.

**Methods:**

`get_or_create(name: str, description: str = "") -> str`
- Deterministically computes `nodeset_id = uuid5(NAMESPACE, name)`
- Issues a `MERGE` on the Neo4j `NodeSet` label using `nodeset_id` as the key
- Updates in-memory registry
- Returns `nodeset_id`

`get_id(name: str) -> Optional[str]`
- Returns the ID from registry if exists, else `None`

`assign_to_node(node_id: str, node_label: str, nodeset_id: str) -> None`
- Writes a `BELONGS_TO_NODESET` edge from the specified node to the NodeSet node

`assign_to_chunk_metadata(chunk_metadata: dict, nodeset_id: str) -> dict`
- Appends `nodeset_id` to the `nodeset_ids` list in a ChromaDB metadata dict
- Returns the updated dict (pure function, no I/O)

`get_global_financial_events_id() -> str`
- Convenience wrapper: calls `get_or_create("GlobalFinancialEvents", "...")`
- Called during system init to guarantee the global anchor always exists

---

## 5. Neo4j Adapter (`core/stores/neo4j_adapter.py`)

### Class: `Neo4jAdapter`

**Purpose:** All Neo4j I/O passes through this class. Uses the official `neo4j` Python driver (async variant). No Cognee dependency for this path.

**Constructor:** Receives `uri`, `username`, `password`, `database` from config. Initialises an `AsyncDriver` lazily on first use.

**Methods:**

`async merge_document_node(node: DocumentNode) -> None`
- Executes `MERGE (d:Document {id: $id}) SET d += $props`

`async merge_chunk_node(node: ChunkNode) -> None`
- Executes `MERGE (c:Chunk {id: $id}) SET c += $props`
- Also issues `MERGE (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c)` and reverse edge in the same transaction

`async merge_entity_node(node: EntityNode) -> None`
- Dynamic label: `MERGE (e:{entity_type} {id: $id}) SET e += $props`
- Also merges into the base `Entity` label for cross-type traversal

`async merge_relationship(source_id: str, target_id: str, rel_type: str, props: dict) -> None`
- Generic edge writer used by both extraction and writeback

`async get_chunk_extraction_status(chunk_ids: List[str]) -> Dict[str, str]`
- Returns `{chunk_id: extraction_status}` for a batch of chunk IDs
- Used by the news agent to identify which chunks need extraction

`async update_chunk_extraction_status(chunk_id: str, status: str) -> None`

`async merge_nodeset_node(nodeset_id: str, name: str, description: str) -> None`

`async anchor_document_to_global(document_id: str, global_anchor_id: str) -> None`
- Issues `MERGE (d:Document {id: $doc_id})-[:ANCHORED_TO]->(g:GlobalAnchor {id: $anchor_id})`

`async close() -> None`

---

## 6. ChromaDB Adapter (`core/stores/chroma_adapter.py`)

### Class: `ChromaDBAdapter`

**Purpose:** All ChromaDB I/O passes through this class. Uses `chromadb` Python client. Collection is accessed lazily (created if not exists).

**Constructor:** Receives `host`, `port`, `collection_name` from config. Initialises `chromadb.AsyncHttpClient` (or `chromadb.HttpClient` if async client is unavailable in installed version — check `chromadb` version).

**Methods:**

`get_or_create_collection(collection_name: str) -> Collection`
- Called lazily on first access

`async upsert_chunks(chunk_ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: List[dict]) -> None`
- Batch upsert. IDs must match the ChunkNode IDs in Neo4j.

`async query(query_embedding: List[float], n_results: int, where: Optional[dict] = None) -> QueryResult`
- Standard vector similarity query. Supports metadata filtering via `where`.

`async get_by_ids(ids: List[str]) -> GetResult`

`async delete_by_ids(ids: List[str]) -> None`

**Metadata schema for each chunk (written at upsert time):**

```json
{
  "chunk_id": "<uuid4>",
  "document_id": "<uuid4>",
  "article_title": "<str>",
  "source_url": "<str>",
  "published_at": "<ISO8601 str>",
  "chunk_index": "<int>",
  "companies_involved": "<comma-separated str>",
  "nodeset_ids": "<comma-separated str>",
  "extraction_status": "PENDING"
}
```

Note: ChromaDB metadata values must be primitive types. Lists are serialised as comma-separated strings. The adapter handles serialisation/deserialisation internally.

---

## 7. ServiceManager Extension (`core/services.py`)

Extend the existing `ServiceManager` class with the following additions. **Do not change existing methods.**

**New private attributes:**
```
_neo4j_adapter: Optional[Neo4jAdapter] = None
_chroma_adapter: Optional[ChromaDBAdapter] = None
_nodeset_manager: Optional[NodeSetManager] = None
_dual_store_ingestor: Optional[DualStoreIngestor] = None
```

**New methods (lazy-init pattern, matching existing style):**

`get_neo4j_adapter() -> Neo4jAdapter`
- Reads `settings.NEO4J_*` values
- Returns singleton `Neo4jAdapter`

`get_chroma_adapter() -> ChromaDBAdapter`
- Reads `settings.CHROMA_*` values
- Returns singleton `ChromaDBAdapter`

`get_nodeset_manager() -> NodeSetManager`
- Depends on `get_neo4j_adapter()`
- Returns singleton `NodeSetManager`

`get_ingestor() -> DualStoreIngestor`
- Depends on `get_neo4j_adapter()`, `get_chroma_adapter()`, `get_nodeset_manager()`, `get_embedding_func()`
- Returns singleton `DualStoreIngestor`

---

## 8. Article Chunker (`core/ingestion/chunker.py`)

### Class: `ArticleChunker`

**Purpose:** Converts a raw news article dict (from NewsAPI) into a list of `ChunkRecord` objects ready for dual-store ingestion.

**Dependencies:** `langchain_text_splitters.RecursiveCharacterTextSplitter` (LangChain-native).

**`ChunkRecord` (dataclass or Pydantic model):**
```
document_id: str           # uuid4, shared across all chunks of this article
chunk_id: str              # uuid4, unique per chunk
chunk_index: int
text: str
article_title: str
source_url: str
published_at: datetime
companies_involved: List[str]  # pre-extracted from NewsAPI response metadata
```

**Constructor:** Receives `chunk_size: int`, `chunk_overlap: int`.

**Methods:**

`chunk_article(article: dict, companies_involved: List[str]) -> Tuple[DocumentMetadata, List[ChunkRecord]]`
- `article` is a raw NewsAPI article dict (keys: `title`, `url`, `publishedAt`, `content`, `description`)
- Generates one `document_id` (uuid4) for the whole article
- Generates one `chunk_id` (uuid4) per chunk
- Concatenates `title + description + content` before splitting (in that order, separated by `\n\n`)
- Uses `RecursiveCharacterTextSplitter` with `chunk_size` and `chunk_overlap`
- Returns a `DocumentMetadata` object (for the document node) and a list of `ChunkRecord`

`DocumentMetadata` (inline Pydantic model):
```
document_id: str
title: str
source_url: str
published_at: datetime
companies_involved: List[str]
```

---

## 9. Dual-Store Ingestor (`core/ingestion/ingestor.py`)

### Class: `DualStoreIngestor`

**Purpose:** Orchestrates the full ingestion pipeline for a batch of news articles. This is the only class that writes to both stores simultaneously.

**Constructor:** Receives `neo4j_adapter`, `chroma_adapter`, `nodeset_manager`, `embedding_func`, `chunker`.

**Methods:**

`async ingest_articles(articles: List[dict], companies_involved: List[str]) -> List[str]`
- Top-level entry point called by `NewsAnalysisAgent`
- Returns list of all `chunk_id`s ingested
- Pipeline steps (in order):
  1. Call `nodeset_manager.get_global_financial_events_id()` to get anchor ID
  2. For each article, call `chunker.chunk_article(...)` to get `DocumentMetadata` + `List[ChunkRecord]`
  3. Call `_write_document_nodes(...)` for all document metadata
  4. Call `_write_chunk_nodes(...)` for all chunk records
  5. Call `_write_vector_chunks(...)` for all chunk records
  6. Return all chunk IDs

`async _write_document_nodes(docs: List[DocumentMetadata], global_anchor_id: str) -> None`
- For each doc: `neo4j_adapter.merge_document_node(DocumentNode(...))`
- For each doc: `neo4j_adapter.anchor_document_to_global(doc.document_id, global_anchor_id)`
- For each doc: `nodeset_manager.assign_to_node(doc.document_id, "Document", global_anchor_id)`

`async _write_chunk_nodes(chunks: List[ChunkRecord], global_anchor_id: str) -> None`
- For each chunk: `neo4j_adapter.merge_chunk_node(ChunkNode(..., extraction_status="PENDING"))`
- Links chunk to its document via `HAS_CHUNK` / `BELONGS_TO_DOCUMENT` edges
- Sets `nodeset_ids` to include `global_anchor_id`

`async _write_vector_chunks(chunks: List[ChunkRecord]) -> None`
- Computes embeddings in batch using `embedding_func.aembed_documents([c.text for c in chunks])`
- Constructs metadata dicts per chunk (matches the ChromaDB metadata schema above)
- Calls `chroma_adapter.upsert_chunks(...)`

---

## 10. Extraction Prompts (`core/graph/extraction_prompts.py`)

This module holds all prompt templates for entity extraction. No logic — only string constants and `ChatPromptTemplate` builders.

**Constants:**

`CHUNK_EXTRACTION_SYSTEM_PROMPT: str`
- Instructs the LLM to extract entities and relationships from a single news chunk only
- Specifies allowed entity types: `Company`, `Person`, `MacroIndicator`, `Event`, `GeoPoliticalRegion`, `Instrument`
- Specifies output must be a JSON object matching `ChunkExtractionResult` schema
- Explicitly states: "Do not infer relationships across multiple articles. Only extract what is directly stated in the provided text."
- Includes schema with `local_id` referencing pattern for edges (avoids hallucinated entity IDs)

`CHUNK_EXTRACTION_USER_TEMPLATE: str`
- Template: `"Extract entities and relationships from the following news chunk:\n\n{chunk_text}\n\nKnown companies involved: {companies}"`

`build_extraction_prompt() -> ChatPromptTemplate`
- Returns a `ChatPromptTemplate` from `CHUNK_EXTRACTION_SYSTEM_PROMPT` and `CHUNK_EXTRACTION_USER_TEMPLATE`
- Uses `langchain_core.prompts.ChatPromptTemplate.from_messages`

---

## 11. News Analysis Agent Refactor (`core/agents/news_analysis_agent.py`)

This is the most substantial change. The agent is restructured into a `StateGraph` (LangGraph) with explicit nodes.

### 11.1 State Model: `NewsAgentState`

```
query: str
ticker: str
start_date: datetime
end_date: datetime
raw_articles: List[dict]              # from NewsAPI
chunk_ids: List[str]                  # after ingestion
retrieved_chunks: List[ChunkResult]   # after vector retrieval
unextracted_chunk_ids: List[str]      # subset needing extraction
extraction_results: List[ChunkExtractionResult]
analysis: Optional[str]
sources: List[CitedSource]
entities_enriched: List[EntityNode]   # for orchestrator writeback
```

### 11.2 LangGraph Nodes

**`fetch_news_node`**
- Calls `service_manager.get_news_api().get_everything(...)` with `ticker`, date range
- Filters out articles with no content
- Extracts `companies_involved` from article metadata (a list of ticker/company names related to the query)
- Writes to state: `raw_articles`

**`ingest_articles_node`**
- Calls `service_manager.get_ingestor().ingest_articles(raw_articles, companies_involved)`
- Writes to state: `chunk_ids`

**`retrieve_chunks_node`**
- Embeds `query` using `service_manager.get_embedding_func()`
- Calls `chroma_adapter.query(query_embedding, n_results=20, where={"ticker": ticker})`
- Deserialises results into `List[ChunkResult]` (a lightweight dataclass: `chunk_id`, `text`, `metadata`, `score`)
- Writes to state: `retrieved_chunks`

**`identify_unextracted_node`**
- Calls `neo4j_adapter.get_chunk_extraction_status([c.chunk_id for c in retrieved_chunks])`
- Filters to those with status `"PENDING"`
- Writes to state: `unextracted_chunk_ids`

**`extract_entities_node`**
> Uses a batched extraction strategy. Unextracted chunks are grouped into batches of up to `EXTRACTION_BATCH_SIZE` chunks (default 6, configurable via `config.py`). Each batch is a single LLM call using `with_structured_output(BatchExtractionResult)`. The prompt includes all chunks in the batch with clear `[CHUNK_ID: ...]` delimiters so the LLM can echo back the correct `chunk_id` in each result. Batches are processed with `asyncio.gather` gated by a module-level `asyncio.Semaphore(EXTRACTION_MAX_CONCURRENCY)` (default 3). After each batch call, each `ChunkExtractionResult` is individually committed to Neo4j and both store statuses are updated.

**`analyse_news_node`**
- Builds context string from ALL `retrieved_chunks` (both previously extracted and newly extracted)
- Calls the synthesis LLM with a financial analysis system prompt (not extraction — analysis only)
- The agent's job here is: "given these grounded chunks, produce a financial news analysis"
- Does NOT produce entities or relationships — those are already in the graph
- Writes to state: `analysis`, `sources`

### 11.3 Graph Wiring

```
START → fetch_news → ingest_articles → retrieve_chunks → 
identify_unextracted → extract_entities → analyse_news → END
```

### 11.4 Agent Class: `NewsAnalysisAgent(AbstractAgent)`

- Wraps the `StateGraph` and exposes `run(input: BaseAgentInput) -> NewsAgentOutput`
- Constructs initial `NewsAgentState` from `BaseAgentInput`
- Returns `NewsAgentOutput` (see below)

### 11.5 `NewsAgentOutput(BaseAgentOutput)` (in `models.py`)

```
analysis: str
sources: List[CitedSource]
entities_enriched: List[EntityNode]
```

Add `get_llm_context_str()` method returning formatted `analysis` + source citations.

---

## 12. Orchestrator Agent Partial Refactor (`core/agents/orchestrator_agent.py`)

### 12.1 Add Graph Context Query Stub

Add a new method `_query_user_graph_context` to `OrchestratorAgent`:

```python
async def _query_user_graph_context(
    self, 
    query: str, 
    conversation_id: Optional[str]
) -> List[dict]:
    """
    Future iteration: retrieve user-specific graph context ranked by recency.
    Queries Neo4j for entities and relationships related to the user's 
    conversation history, ordered by `ingested_at` descending.
    
    Returns: List of entity/relationship dicts (empty list until implemented)
    """
    # TODO: Implement in future iteration
    return []
```

Call this stub at the start of `_plan_node` and include its output (empty for now) in a reserved `graph_context` field on `OrchestratorState`.

### 12.2 `OrchestratorState` Extension

Add field:
```
graph_context: List[dict] = Field(default_factory=list)
```

### 12.3 No Other Changes

The rest of `orchestrator_agent.py` remains as-is. The `_synthesize_node` already handles `entities_enriched` from agent outputs correctly.

---

## 13. Initialisation & Startup Sequence

In whichever startup module initialises the application (e.g., `main.py` or `app.py`):

1. `service_manager.get_neo4j_adapter()` — verifies connection
2. `service_manager.get_nodeset_manager().get_global_financial_events_id()` — creates global anchor node if not exists (idempotent)
3. `service_manager.get_chroma_adapter()` — verifies/creates collection
4. `service_manager.get_ingestor()` — warms up ingestor dependencies

This guarantees the global anchor node exists before any article ingestion.

---

## 14. Data Flow Summary (End-to-End)

```
User Query
    │
    ▼
OrchestratorAgent._plan_node
    │  (calls _query_user_graph_context stub — returns [])
    ▼
NewsAnalysisAgent (LangGraph pipeline)
    │
    ├─► fetch_news_node        → raw NewsAPI articles
    │
    ├─► ingest_articles_node   → DualStoreIngestor
    │       ├── DocumentNode  (Neo4j, no full text)
    │       │       └── :ANCHORED_TO → GlobalAnchorNode
    │       ├── ChunkNode[]   (Neo4j, status=PENDING, nodeset_ids=[global_anchor_id])
    │       └── ChromaDB upsert[] (same IDs, metadata with nodeset_ids)
    │
    ├─► retrieve_chunks_node   → vector similarity search in ChromaDB
    │
    ├─► identify_unextracted   → neo4j batch status check
    │
    ├─► extract_entities_node  → per-chunk LLM extraction (PENDING chunks only)
    │       ├── EntityNode[]  (Neo4j, UUID5 canonical)
    │       ├── MENTIONS_ENTITY edges
    │       ├── RELATED_TO edges
    │       ├── ChunkNode status → EXTRACTED (Neo4j + ChromaDB)
    │       └── entities_enriched → state
    │
    └─► analyse_news_node      → financial analysis from grounded chunks
            └── analysis, sources → NewsAgentOutput
    │
    ▼
OrchestratorAgent._synthesize_node
    │  (receives NewsAgentOutput.entities_enriched for writeback)
    └─► run_conversation_writeback (async, non-blocking)
```

---

## Task List

### Phase 1 — Foundation

- [ ] **T-01** Extend `core/config.py` with Neo4j AuraDB and ChromaDB settings fields; validate all credentials are sourced from env
- [ ] **T-02** Create `core/graph/models.py` with `DocumentNode`, `ChunkNode`, `EntityNode`, `GlobalAnchorNode`, `ExtractedRelationship`, `ChunkExtractionResult` Pydantic models
- [ ] **T-03** Create `core/stores/neo4j_adapter.py` with `Neo4jAdapter` class and all methods described in section 5
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
- [ ] **T-13** Implement `extract_entities_node` with parallel per-chunk extraction using `asyncio.gather`; use `with_structured_output(ChunkExtractionResult)` for LLM calls; write entities/edges to Neo4j; update status in both stores
- [ ] **T-14** Implement `analyse_news_node` that uses only pre-extracted chunk context for financial analysis (no entity extraction in this node)
- [ ] **T-15** Wire all nodes into a `StateGraph`, compile, and expose via `NewsAnalysisAgent.run()`

### Phase 4 — Orchestrator Update

- [ ] **T-16** Add `graph_context: List[dict]` field to `OrchestratorState`
- [ ] **T-17** Add `_query_user_graph_context` stub method to `OrchestratorAgent` with docstring describing future Neo4j recency query; call it in `_plan_node`

### Phase 5 — Startup & Integration

- [ ] **T-18** Add startup initialisation sequence (global anchor node creation, collection warm-up) to application entry point
- [ ] **T-19** Create `__init__.py` files for `core/graph/`, `core/stores/`, `core/ingestion/`
- [ ] **T-20** End-to-end integration test: ingest 2 articles → retrieve → confirm chunk IDs match in both ChromaDB and Neo4j → confirm `GlobalFinancialEvents` anchor node exists with correct edges

---

*End of Implementation Plan*
