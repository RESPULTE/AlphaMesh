# AlphaMesh Pipeline Implementation Plan v2
## Refined for Existing Codebase Integration

---

## 0. What Changes vs. What Stays

### DO NOT TOUCH
- `graph_models.py` — all DataPoint schemas are correct and reused as-is
- `entity_merger.py` — reused by write-back for post-write dedup
- `financial_retriever.py` — retrieval layer is correct, untouched
- `nodeset_manager.py` — NodeSet lifecycle management is correct
- `exceptions.py` — exception hierarchy is correct
- `financial_db.py` — SQLite financial store is correct
- `fundamental_analysis_agent.py` — agent logic is correct; minor output extension only
- `news_analysis_agent.py` — agent logic is correct; minor output extension only
- `base_agent.py` — abstract base is correct

### MODIFY
- `pipeline_tasks.py` — add `build_lean_document_pipeline()`, add `summarise_chunks_lean()` task
- `memory_system.py` — add `ingest_document_lean()`, add `write_back_conversation_entities()`, deprecate `ingest_conversation()` (replaced by write-back)
- `orchestrator_agent.py` — modify `_synthesize_node` to emit CoT `<relationships>` block + fire write-back
- `models.py` — extend `BaseAgentOutput` to carry enriched entity DataPoints
- `prompts.py` — add `LEAN_SUMMARY_SYSTEM_PROMPT` and `SYNTHESISER_WRITEBACK_SYSTEM_PROMPT`

### ADD
- `core/memory/conversation_writeback.py` — new module handling async graph write-back from synthesizer output

---

## 1. Models Extension (`core/agents/models.py`)

Extend `BaseAgentOutput` to carry enriched entity DataPoints. This is the contract between downstream agents and the write-back system.

```python
# core/agents/models.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Optional

from cognee.infrastructure.engine import DataPoint
from pydantic import BaseModel, Field


class BaseAgentInput(BaseModel):
    """
    The unified input schema shared by the Orchestrator and all Sub-Agents.
    UNCHANGED — do not modify fields.
    """
    query: str = Field(description="The original user query for context.")
    vector_query: str = Field(
        description="The query optimized for vector store retrieval."
    )
    ticker: Optional[str] = Field(description="The stock ticker symbol (e.g., AAPL).")
    metrics: Optional[List[str]] = Field(default_factory=list)
    start_date: Optional[datetime] = Field(default=None)
    end_date: Optional[datetime] = Field(default=None)


class BaseAgentOutput(BaseModel, ABC):
    """
    Abstract base for agent outputs.

    CHANGE: Added `entities_enriched` field.
    Each agent MUST populate this with the DataPoint objects it has resolved
    during its run. These are passed to the write-back system after synthesis.

    Minimum requirement per agent:
      - FundamentalAnalysisAgent: one Company DataPoint for the analysed ticker
      - NewsAnalysisAgent: one Company DataPoint + any FinancialEvent DataPoints
        that were prominent in the retrieved articles
    """
    agent_name: str = Field(
        description="The name of the agent that produced this output."
    )
    analysis: str = Field(
        description="The detailed analysis or primary output of the agent."
    )
    entities_enriched: List[Any] = Field(
        default_factory=list,
        description=(
            "List of enriched DataPoint objects resolved by this agent. "
            "Used by the write-back system to populate the knowledge graph. "
            "Must use graph_models.py classes (Company, FinancialEvent, etc.)."
        ),
    )

    @abstractmethod
    def get_llm_context_str(self) -> str:
        raise NotImplementedError
```

---

## 2. Agent Entity Emission

Each agent must populate `entities_enriched` at the end of its run. This is the minimal enrichment — agents do not perform graph extraction; they only emit the primary entities they already know about.

### `fundamental_analysis_agent.py` — modify `_generate_analysis` node

At the end of `_generate_analysis`, construct a `Company` DataPoint from the ticker and analysis. The agent already knows the company it analysed.

```python
# Inside _generate_analysis, after generating analysis text:

from core.memory.graph_models import Company
from core.memory.nodeset_manager import get_canonical_id  # uuid5 from pipeline_tasks.py

def _build_company_entity(ticker: str, analysis_text: str) -> Company:
    """
    Build a minimal enriched Company DataPoint from what the fundamental agent knows.
    The description is seeded from the analysis — this is the enrichment.
    """
    # Extract first sentence of analysis as description (terse but informative)
    first_sentence = analysis_text.split(".")[0].strip() + "." if analysis_text else ""
    return Company(
        id=get_canonical_id(ticker.upper()),   # reuse existing uuid5 helper
        ticker=ticker.upper(),
        name=ticker.upper(),                    # orchestrator plan may not have full name
        description=first_sentence,
        sector="",                              # downstream assign_nodesets will resolve
        enriched=True,                          # flag for dedup pipeline
    )

# In _generate_analysis return:
return FundamentalAnalysisOutput(
    financial_data=state.financial_data,
    analysis=response.content,
    entities_enriched=[_build_company_entity(state.ticker, response.content)],
)
```

### `news_analysis_agent.py` — modify `_generate_analysis` node

At the end of `_generate_analysis`, emit a `Company` DataPoint and any `FinancialEvent` DataPoints that appeared in the news context. The news agent already has article titles and content — event extraction is cheap at this stage (no extra LLM call needed for just the company).

```python
# Inside _generate_analysis, after generating analysis text:

from core.memory.graph_models import Company, FinancialEvent
from core.memory.nodeset_manager import get_canonical_id
from datetime import datetime

def _build_entities_from_news(
    ticker: str, sources: list, analysis_text: str
) -> list:
    """
    Build minimal enriched DataPoints from what the news agent already retrieved.
    Company is always emitted. No extra LLM call — events are not extracted here;
    that is the synthesiser's job via the CoT <relationships> block.
    """
    entities = []

    company = Company(
        id=get_canonical_id(ticker.upper()),
        ticker=ticker.upper(),
        name=ticker.upper(),
        description=f"Company in focus for news analysis: {ticker.upper()}",
        sector="",
        enriched=True,
    )
    entities.append(company)
    return entities

# In _generate_analysis return:
return NewsAnalysisOutput(
    analysis=retval.content,
    sources=state.news_context,
    entities_enriched=_build_entities_from_news(
        state.ticker, state.news_context, retval.content
    ),
)
```

---

## 3. Synthesiser CoT Change (`core/agents/orchestrator_agent.py`)

This is the most significant change. The `_synthesize_node` is modified to:
1. Emit a `<relationships>` block as a CoT reasoning step BEFORE the user-facing response
2. Parse the relationships block
3. Fire write-back asynchronously AFTER returning the response to the user

### Add to `prompts.py`

```python
# core/memory/prompts.py — ADD these two prompts

SYNTHESISER_WRITEBACK_SYSTEM_PROMPT = """\
You are a Senior Financial Analyst and Knowledge Graph Architect.

Your task has TWO mandatory parts, in this exact order:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — RELATIONSHIP REASONING (do this FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing the user response, reason about the relationships between
the entities surfaced in the agent findings. This is your thinking step.

Output a <relationships> block as a JSON array. Each entry must be:
{
  "from_name": "<entity name>",
  "from_type": "Company | FinancialEvent | FinancialConcept | Sector",
  "relation": "<RELATION_TYPE>",
  "to_name": "<entity name>",
  "to_type": "Company | FinancialEvent | FinancialConcept | Sector",
  "confidence": "high | low"
}

Allowed RELATION_TYPE values (use exact strings):
  AFFECTS | CAUSED_BY | INCREASES | DECREASES | CORRELATED_WITH |
  MITIGATES | EXPOSES_TO | REPORTED_BY | COMPETES_WITH | ACQUIRED_BY

CONFIDENCE rules:
  "high" = explicitly stated in agent findings with specific evidence
  "low"  = inferred or implied without direct evidence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — USER RESPONSE (do this SECOND, using Part 1 as your foundation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a cohesive narrative financial analysis grounded in the agent findings.
Use numeric in-text citations like [1], [2] when referencing news sources.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT FORMAT (strictly):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<relationships>
[...json array or empty array []...]
</relationships>
<response>
...your narrative financial analysis for the user...
</response>

Do not output anything outside these two blocks.
""".strip()


LEAN_SUMMARY_SYSTEM_PROMPT = """\
Extract financial facts. Output 1-2 sentences only.
Include: company/ticker, metric or topic, value or direction, time period.
No preamble. No filler. If no financial fact is present, output exactly: NO_FINANCIAL_DATA
""".strip()
```

### Modify `_synthesize_node` in `orchestrator_agent.py`

```python
# core/agents/orchestrator_agent.py

import asyncio
import re
import json
from core.memory.prompts import SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
from core.memory.conversation_writeback import run_conversation_writeback

# Modify OrchestratorState to carry write-back payload
class OrchestratorState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    plan: Optional[OrchestratorPlan] = None
    agent_outputs: Dict[str, BaseAgentOutput] = Field(default_factory=dict)
    final_response: Optional[FinalResponse] = None
    # NEW: write-back payload populated by _synthesize_node
    writeback_relationships: List[dict] = Field(default_factory=list)
    writeback_entities: List[Any] = Field(default_factory=list)
    conversation_id: Optional[str] = None  # passed in from caller


async def _synthesize_node(self, state: OrchestratorState) -> Dict[str, Any]:
    """
    MODIFIED: Emits CoT <relationships> block before the user-facing response.
    The relationships block serves as explicit reasoning grounding the analysis.
    Write-back is fired asynchronously after this node completes.
    """
    context_parts = []
    fundamental_df = None
    news_sources = []
    all_enriched_entities = []

    for name, output in state.agent_outputs.items():
        context_parts.append(output.get_llm_context_str())
        if name == "fundamentals_agent":
            fundamental_df = getattr(output, "financial_data", None)
        if name == "news_agent":
            news_sources = getattr(output, "sources", [])
        # Collect enriched entities from ALL agents
        all_enriched_entities.extend(getattr(output, "entities_enriched", []))

    # Use the CoT synthesiser prompt instead of the old system prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYNTHESISER_WRITEBACK_SYSTEM_PROMPT + "\n\nAgent Findings:\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Produce the relationship reasoning and final analysis."),
        ]
    )

    chain = prompt | self._llm
    response = await chain.ainvoke(
        {"history": state.messages, "context": "\n\n".join(context_parts)}
    )
    raw = response.content.strip()

    # --- Parse <relationships> block (fault-tolerant) ---
    relationships = []
    rel_match = re.search(r"<relationships>(.*?)</relationships>", raw, re.DOTALL)
    if rel_match:
        try:
            relationships = json.loads(rel_match.group(1).strip())
            if not isinstance(relationships, list):
                relationships = []
        except json.JSONDecodeError:
            relationships = []   # malformed JSON must NOT break user response

    # --- Parse <response> block ---
    resp_match = re.search(r"<response>(.*?)</response>", raw, re.DOTALL)
    user_response = resp_match.group(1).strip() if resp_match else raw

    # --- Fire write-back asynchronously (non-blocking) ---
    # This runs after the node returns — user never waits for it
    if state.conversation_id:
        asyncio.create_task(
            run_conversation_writeback(
                relationships=relationships,
                enriched_entities=all_enriched_entities,
                user_email=None,          # pass from caller if multi-tenant needed
                conversation_id=state.conversation_id,
            )
        )

    return {
        "final_response": FinalResponse(
            summary=user_response,
            fundamental_data=fundamental_df,
            sources=news_sources,
        ),
        "writeback_relationships": relationships,
        "writeback_entities": all_enriched_entities,
    }


# Modify OrchestratorAgent.run() to accept conversation_id
async def run(
    self, messages: List[BaseMessage], conversation_id: Optional[str] = None
) -> FinalResponse:
    """Entry point accepting a list of LangChain messages."""
    initial_state = OrchestratorState(
        messages=messages,
        conversation_id=conversation_id,
    )
    final_state = await self._graph.ainvoke(initial_state)

    if final_state.get("final_response"):
        return final_state["final_response"]

    return FinalResponse(
        summary=final_state["plan"].final_answer or "I couldn't process that request."
    )
```

---

## 4. Conversation Write-Back (`core/memory/conversation_writeback.py`)

New module. Called fire-and-forget from `_synthesize_node`. Responsible for writing enriched entities and synthesiser-derived relationships into the knowledge graph.

```python
"""
core/memory/conversation_writeback.py

Async write-back of enriched entities and relationships to the knowledge graph.
Called fire-and-forget from OrchestratorAgent._synthesize_node.

Write order (strict):
  1. Resolve NodeSets for enriched entities (via assign_nodesets logic)
  2. Dedup enriched entities against existing graph nodes
  3. Write enriched entities via add_data_points
  4. Resolve relationship endpoints against written entities
  5. Write EntityRelationship DataPoints

A failure at any step is logged but NEVER propagated — this is a background task.
The user response has already been delivered before this runs.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, List, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.tasks.storage import add_data_points

from core.memory.entity_merger import find_and_merge_candidates
from core.memory.graph_models import (
    Company,
    FinancialConcept,
    FinancialEvent,
    FinancialKnowledgeGraph,
)
from core.memory.nodeset_manager import (
    GLOBAL_FINANCIAL_EVENT_NODESET,
    GLOBAL_FINANCIAL_WISDOM_NODESET,
    GLOBAL_NODESET_NAME,
    get_or_create_nodeset,
    get_or_create_global_nodeset,
    get_or_create_user_nodeset,
    get_user_nodeset_name,
)
from core.memory.pipeline_tasks import get_canonical_id

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EntityRelationship DataPoint — written as a graph edge
# ---------------------------------------------------------------------------

from cognee.infrastructure.engine import DataPoint
from pydantic import Field


class EntityRelationship(DataPoint):
    """
    A directed relationship between two named entities.
    Written to the graph by the write-back system.
    canonical ID: uuid5 of "{from_name}::{relation}::{to_name}"
    """
    __tablename__ = "entity_relationship"

    from_name: str
    from_type: str
    relation_type: str
    to_name: str
    to_type: str
    confidence: str = "low"
    source_conversation_id: str = ""
    metadata: dict = {"index_fields": ["from_name", "relation_type", "to_name"]}


def _relationship_id(from_name: str, relation: str, to_name: str) -> uuid.UUID:
    key = f"{from_name.upper()}::{relation}::{to_name.upper()}"
    return uuid.uuid5(uuid.NAMESPACE_DNS, key)


# ---------------------------------------------------------------------------
# NodeSet resolution for enriched entities
# ---------------------------------------------------------------------------

async def _resolve_entity_nodeset(entity: Any) -> None:
    """
    Assign belongs_to_set on an enriched entity using the same logic as
    assign_nodesets() in pipeline_tasks.py, but without requiring a
    DocumentChunk wrapper.

    This is a simplified version that handles the entity types produced
    by downstream agents during conversation write-back.
    """
    entity_type = type(entity).__name__

    try:
        entity.belongs_to_set = getattr(entity, "belongs_to_set", []) or []

        if entity_type == "FinancialConcept":
            ns = await get_or_create_nodeset(GLOBAL_FINANCIAL_WISDOM_NODESET)
            entity.belongs_to_set.append(ns)

        elif entity_type == "FinancialEvent":
            ns = await get_or_create_nodeset(GLOBAL_FINANCIAL_EVENT_NODESET)
            entity.belongs_to_set.append(ns)

        elif entity_type == "Company":
            global_ns = await get_or_create_global_nodeset()
            entity.belongs_to_set.append(global_ns)
            # If sector is known, also resolve sector NodeSet
            if getattr(entity, "sector", None):
                try:
                    sector_ns = await get_or_create_nodeset(entity.sector)
                    entity.belongs_to_set.append(sector_ns)
                except Exception:
                    pass  # sector resolution failure is non-fatal

    except Exception as exc:
        logger.warning(
            "write_back: failed to resolve NodeSet for %s '%s': %s",
            entity_type,
            getattr(entity, "name", getattr(entity, "ticker", "?")),
            exc,
        )


# ---------------------------------------------------------------------------
# Entity deduplication check
# ---------------------------------------------------------------------------

async def _should_write_entity(entity: Any) -> bool:
    """
    Check if entity should be written or skipped.
    Mirrors the dedup logic from graph_extraction._resolve_entity_pool but
    operates against the live graph rather than a batch.

    Returns True if entity should be written (new or unenriched stub exists).
    Returns False if an already-enriched node with same canonical ID exists.
    """
    from cognee.infrastructure.databases.relational import get_relational_engine

    canonical_id = str(getattr(entity, "id", None) or "")
    if not canonical_id:
        return True

    table = getattr(entity, "__tablename__", None)
    if not table:
        return True

    try:
        engine = get_relational_engine()
        # Check if an enriched node already exists with this ID
        result = await engine.fetch_one(
            f"SELECT id FROM {table} WHERE id = :cid LIMIT 1",
            {"cid": canonical_id},
        )
        # If node doesn't exist → write it
        return result is None
    except Exception:
        # If check fails, write anyway — add_data_points handles upserts
        return True


# ---------------------------------------------------------------------------
# Relationship DataPoint construction
# ---------------------------------------------------------------------------

def _build_relationship_datapoints(
    relationships: List[dict],
    conversation_id: str,
) -> List[EntityRelationship]:
    """
    Convert the synthesiser's <relationships> JSON array into
    EntityRelationship DataPoints ready for add_data_points.

    Skips malformed entries silently — never raises.
    """
    datapoints = []
    for rel in relationships:
        try:
            from_name = rel["from_name"]
            relation = rel["relation"]
            to_name = rel["to_name"]
        except KeyError:
            logger.debug("write_back: skipping malformed relationship: %s", rel)
            continue

        datapoints.append(
            EntityRelationship(
                id=_relationship_id(from_name, relation, to_name),
                from_name=from_name,
                from_type=rel.get("from_type", "unknown"),
                relation_type=relation,
                to_name=to_name,
                to_type=rel.get("to_type", "unknown"),
                confidence=rel.get("confidence", "low"),
                source_conversation_id=conversation_id,
            )
        )

    return datapoints


# ---------------------------------------------------------------------------
# Main write-back entry point
# ---------------------------------------------------------------------------

async def run_conversation_writeback(
    relationships: List[dict],
    enriched_entities: List[Any],
    conversation_id: str,
    user_email: Optional[str] = None,
) -> None:
    """
    Write enriched entities and synthesiser-derived relationships to the graph.

    Called fire-and-forget from OrchestratorAgent._synthesize_node.
    All exceptions are caught and logged — this function NEVER raises.

    Write order:
      1. Resolve NodeSets for all enriched entities
      2. Filter to entities that need writing (dedup check)
      3. Write enriched entities via add_data_points
      4. Run entity merger on written entities (APOC fuzzy + semantic dedup)
      5. Build and write EntityRelationship DataPoints

    Args:
        relationships:      Parsed <relationships> JSON list from synthesiser.
        enriched_entities:  DataPoint objects from all downstream agents.
        conversation_id:    Unique ID for this conversation turn (for traceability).
        user_email:         If provided, user-specific entities get USER NodeSet.
    """
    try:
        if not enriched_entities and not relationships:
            logger.debug("write_back [%s]: nothing to write.", conversation_id)
            return

        # --- Step 1: Resolve NodeSets ---
        for entity in enriched_entities:
            await _resolve_entity_nodeset(entity)

        # If user_email provided, also tag user-specific entities
        if user_email:
            from core.memory.graph_models import UserInvestmentInterest, UserLearningInterest
            _, user_ns = await get_or_create_user_nodeset(user_email)
            for entity in enriched_entities:
                if isinstance(entity, (UserInvestmentInterest, UserLearningInterest)):
                    entity.belongs_to_set = getattr(entity, "belongs_to_set", []) or []
                    entity.belongs_to_set.append(user_ns)

        # --- Step 2: Filter to entities that need writing ---
        to_write = []
        for entity in enriched_entities:
            if await _should_write_entity(entity):
                to_write.append(entity)
            else:
                logger.debug(
                    "write_back [%s]: skipping already-enriched entity %s '%s'.",
                    conversation_id,
                    type(entity).__name__,
                    getattr(entity, "name", getattr(entity, "ticker", "?")),
                )

        # --- Step 3: Write enriched entities ---
        if to_write:
            await add_data_points(to_write)
            logger.info(
                "write_back [%s]: wrote %d enriched entities.",
                conversation_id,
                len(to_write),
            )

        # --- Step 4: Run entity merger on written entities ---
        # This catches any near-duplicates introduced by the write-back
        # and merges them using the existing APOC fuzzy + semantic pipeline.
        if to_write:
            try:
                graph_engine = await get_graph_engine()
                await find_and_merge_candidates(graph_engine, to_write)
            except Exception as merge_exc:
                # Merger failure is non-fatal — entities are written, just not merged
                logger.warning(
                    "write_back [%s]: entity merger failed (non-fatal): %s",
                    conversation_id,
                    merge_exc,
                )

        # --- Step 5: Build and write relationship DataPoints ---
        if relationships:
            rel_datapoints = _build_relationship_datapoints(
                relationships, conversation_id
            )
            if rel_datapoints:
                await add_data_points(rel_datapoints)
                logger.info(
                    "write_back [%s]: wrote %d relationship edges.",
                    conversation_id,
                    len(rel_datapoints),
                )

    except Exception as exc:
        # CRITICAL: This function must NEVER raise — it is fire-and-forget
        logger.error(
            "write_back [%s]: unhandled error (user response unaffected): %s",
            conversation_id,
            exc,
            exc_info=True,
        )
```

---

## 5. Lean Document Pipeline (`core/memory/pipeline_tasks.py`)

Add `build_lean_document_pipeline()` alongside the existing `build_financial_pipeline()`. The existing pipeline is kept for backward compatibility but is no longer called for standard document ingestion.

```python
# core/memory/pipeline_tasks.py — ADD these functions

from core.memory.prompts import LEAN_SUMMARY_SYSTEM_PROMPT


async def summarise_chunks_lean(
    data_chunks: List[DocumentChunk],
) -> List[DocumentChunk]:
    """
    Lean summarisation pipeline task.

    Generates a terse 1-2 sentence financial summary for each chunk
    that has been classified into a meaningful section.

    DIFFERENCES from the old summarize_text cognee task:
      - Uses LEAN_SUMMARY_SYSTEM_PROMPT (domain-specific, forces terse output)
      - Hard cap of 80 output tokens per chunk
      - Skips unclassified chunks (section is None or chunk < 120 chars)
      - Stores summary text back on chunk.text_summary (not a new DataPoint)
      - Returns the SAME chunk list — does not add extra DataPoints

    This keeps the vector store summary index populated for
    SearchType.SUMMARIES retrieval, without spinning up the full
    LLM-based cognee summarise_text pipeline.
    """
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    for chunk in data_chunks:
        # Skip front matter / table of contents / trivially short chunks
        if not chunk.text or len(chunk.text) < 120:
            continue

        try:
            summary = await LLMGateway.acreate(
                messages=[
                    {"role": "system", "content": LEAN_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk.text[:2000]},  # cap input too
                ],
                max_tokens=80,
                temperature=0.0,
            )
            summary_text = summary.choices[0].message.content.strip()

            if summary_text == "NO_FINANCIAL_DATA":
                continue

            # Store on the chunk for add_data_points to index
            chunk.text_summary = summary_text

        except Exception as exc:
            logger.warning(
                "summarise_chunks_lean: failed for chunk %s: %s", chunk.id, exc
            )
            continue

    return data_chunks


async def build_lean_document_pipeline(
    chunks_per_batch: int = 100,
    chunk_size: Optional[int] = None,
    include_summaries: bool = True,
) -> list[Task]:
    """
    Build a token-efficient ingestion pipeline for financial documents.

    DIFFERENCE from build_financial_pipeline:
      - Removes extract_financial_graph (eliminates 2 LLM calls per chunk)
      - Removes assign_nodesets (no entities extracted, nothing to assign)
      - Removes merge_entities (nothing to merge)
      - Optionally adds summarise_chunks_lean (1 LLM call per chunk, 80 tokens out)
      - Graph construction is deferred to conversation write-back

    Task order:
      1. classify_documents
      2. extract_chunks_from_documents
      3. summarise_chunks_lean        ← optional; 1 LLM call/chunk, 80 tokens max
      4. add_data_points              ← embeds chunks (and summaries if present)

    Args:
        chunks_per_batch:   Batch size for add_data_points.
        chunk_size:         Max tokens per chunk (auto if None).
        include_summaries:  If True, runs lean summarisation. Default True.
                            Set False to skip all LLM calls entirely (pure embed).

    Returns:
        List of Task objects in execution order.
    """
    cognify_config = get_cognify_config()
    embed_triplets = cognify_config.triplet_embedding

    tasks = [
        Task(classify_documents),
        Task(
            extract_chunks_from_documents,
            max_chunk_size=chunk_size,
            chunker=TextChunker,
        ),
    ]

    if include_summaries:
        tasks.append(Task(summarise_chunks_lean))

    tasks.append(
        Task(
            add_data_points_with_custom_edges,
            embed_triplets=embed_triplets,
            task_config={"batch_size": chunks_per_batch},
        )
    )

    logger.info(
        "Built lean document pipeline with %d tasks (summaries=%s).",
        len(tasks),
        include_summaries,
    )
    return tasks
```

---

## 6. Memory System Changes (`core/memory/memory_system.py`)

### Add `ingest_document_lean()`

New method for ingesting financial documents without graph extraction. Replaces calling `ingest_financial_report()` + `cognify()` for documents.

```python
# core/memory/memory_system.py — ADD method to FinancialMemorySystem

async def ingest_document_lean(
    self,
    ticker: str,
    report_type: str,
    content: str,
    period: Optional[str] = None,
    include_summaries: bool = True,
    is_global: bool = True,
) -> Any:
    """
    Ingest a financial document using the lean pipeline (no graph extraction).

    REPLACES the pattern of: ingest_financial_report() → cognify()
    for standard document ingestion use cases.

    The lean pipeline:
      - Chunks the document
      - Optionally generates terse financial summaries (80 tokens/chunk)
      - Embeds and indexes chunks for vector retrieval
      - Does NOT extract entities or build graph edges

    Graph construction happens lazily via conversation write-back as
    users query the system and the synthesiser extracts relationships.

    Args:
        ticker:            Stock ticker (e.g. "AAPL").
        report_type:       "10-K", "10-Q", "8-K", "annual", "quarterly".
        content:           Full text of the report.
        period:            Reporting period string (e.g. "Q3 2024"). Optional.
        include_summaries: If True, runs lean per-chunk summarisation.
        is_global:         True for public SEC filings (default).

    Returns:
        PipelineRunInfo from run_custom_pipeline.

    Raises:
        IngestionError:   If cognee.add() fails.
        MemorySystemError: If the pipeline fails.
    """
    self._require_initialized()

    if not ticker or not content:
        raise ValueError("ticker and content are required.")

    header = f"FINANCIAL REPORT\nTICKER: {ticker.upper()}\nTYPE: {report_type}\n"
    if period:
        header += f"PERIOD: {period}\n"
    text = header + "\n" + content

    node_set = [GLOBAL_NODESET_NAME] if is_global else None
    await self._add_to_cognee(text, node_set=node_set)

    logger.info(
        "Running lean pipeline for %s %s (%d chars, summaries=%s).",
        report_type, ticker.upper(), len(text), include_summaries,
    )

    tasks = await build_lean_document_pipeline(include_summaries=include_summaries)

    try:
        result = await run_custom_pipeline(
            tasks=tasks,
            dataset=DATASET_NAME,
            pipeline_name="lean_document_pipeline",
            incremental_loading=True,
        )
        logger.info("Lean document pipeline completed for %s %s.", report_type, ticker)
        return result
    except Exception as exc:
        raise MemorySystemError(f"Lean document pipeline failed: {exc}") from exc
```

### Deprecate `ingest_conversation()`

`ingest_conversation()` currently runs full cognify on conversation text, which is the expensive LLM-per-chunk path. With write-back from the synthesiser, conversation insights flow into the graph directly as typed DataPoints — no need to cognify raw conversation text.

```python
# core/memory/memory_system.py — REPLACE ingest_conversation body

async def ingest_conversation(
    self,
    user_email: str,
    messages: List[dict],
) -> None:
    """
    DEPRECATED: Conversation insights now flow into the knowledge graph via
    OrchestratorAgent._synthesize_node → run_conversation_writeback.

    Kept for backward compatibility. In the new architecture this method
    is a no-op — calling it will log a warning and return immediately.

    To write conversation entities to the graph, call run_conversation_writeback()
    from the synthesiser after each conversation turn.
    """
    logger.warning(
        "ingest_conversation() is deprecated. Conversation insights are now "
        "written to the graph via run_conversation_writeback() from the synthesiser. "
        "This call has no effect."
    )
    return
```

---

## 7. `__init__.py` Update (`core/memory/__init__.py`)

Export the new write-back module and `EntityRelationship` DataPoint.

```python
# core/memory/__init__.py — ADD to imports and __all__

from core.memory.conversation_writeback import (
    EntityRelationship,
    run_conversation_writeback,
)

# Add to __all__:
"EntityRelationship",
"run_conversation_writeback",
```

---

## 8. Integration Contract

### Data flow — document ingestion (Pipeline A)

```
SEC filing / news text
        │
        ▼
FinancialMemorySystem.ingest_document_lean()
        │  cognee.add(text, node_set=[GLOBAL_NODESET_NAME])
        ▼
build_lean_document_pipeline()
  ├── classify_documents
  ├── extract_chunks_from_documents    ← TextChunker (unchanged)
  ├── summarise_chunks_lean            ← 1 LLM call/chunk, max_tokens=80
  └── add_data_points                  ← embed chunks + summaries → vector store
        │
        ▼
DocumentChunk nodes in relational DB
Chunk embeddings in vector store (SearchType.CHUNKS)
Summary embeddings in vector store (SearchType.SUMMARIES)
NO graph edges written yet
```

### Data flow — conversation (Pipeline B)

```
User message
        │
        ▼
OrchestratorAgent.run(messages, conversation_id)
        │
        ├── _plan_node
        │     LLM → OrchestratorPlan (ticker, agents_required, dates)
        │
        ├── _execute_node (parallel)
        │     ├── FundamentalAnalysisAgent.run()
        │     │     returns FundamentalAnalysisOutput
        │     │       .analysis: str
        │     │       .financial_data: DataFrame
        │     │       .entities_enriched: [Company(ticker, description)]
        │     │
        │     └── NewsAnalysisAgent.run()
        │           returns NewsAnalysisOutput
        │             .analysis: str
        │             .sources: List[CitedSource]
        │             .entities_enriched: [Company(ticker)]
        │
        └── _synthesize_node
              ├── LLM call with SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
              │     input:  agent findings (get_llm_context_str()) + chat history
              │     output: <relationships>[...json...]</relationships>
              │             <response>...narrative analysis...</response>
              │
              ├── parse <relationships> → List[dict]  (fault-tolerant)
              ├── parse <response> → user_response: str
              │
              ├── return FinalResponse(summary=user_response, ...)  ← USER GETS THIS
              │
              └── asyncio.create_task(run_conversation_writeback(...))  ← FIRE & FORGET
                        │
                        ├── resolve NodeSets for enriched entities
                        ├── dedup check against graph
                        ├── add_data_points(enriched_entities)
                        ├── find_and_merge_candidates()  ← APOC fuzzy + semantic
                        └── add_data_points(EntityRelationship DataPoints)
```

### What each pipeline reads and writes

| | Reads | Writes |
|---|---|---|
| Pipeline A (lean ingest) | Raw text files | DocumentChunk nodes, chunk/summary embeddings in vector store |
| Pipeline B (conversation) | DocumentChunk embeddings (SearchType.CHUNKS/SUMMARIES), graph traversals | Company/FinancialEvent/FinancialConcept nodes, EntityRelationship edges |
| Write-back | Enriched entities from agents, relationships from synthesiser | Graph nodes via add_data_points, dedup via entity_merger |

### Invariants that must hold

1. **`ingest_document_lean()` never extracts graph entities.** The lean pipeline ends at `add_data_points` with chunks only. No `extract_financial_graph`, no `assign_nodesets`.

2. **Write-back is always fire-and-forget.** `run_conversation_writeback` is always wrapped in `asyncio.create_task()`. It contains a top-level try/except that logs errors but never raises. A write-back failure must never surface to the user.

3. **Relationship DataPoints are written after entities.** Step 5 of `run_conversation_writeback` runs only after Steps 1–4 complete. Edges point to nodes that already exist.

4. **Entity NodeSet resolution reuses existing `nodeset_manager` functions.** `_resolve_entity_nodeset()` in `conversation_writeback.py` calls the same `get_or_create_nodeset()` functions used by `assign_nodesets()` in the pipeline. NodeSet IDs are deterministic and idempotent.

5. **`ingest_conversation()` is a no-op.** Any existing call sites that call it must be updated to rely on write-back instead. The method logs a deprecation warning.

6. **`entities_enriched` on `BaseAgentOutput` defaults to empty list.** Agents that do not populate it will not contribute entities to write-back — this is safe. The synthesiser's relationship extraction still runs; it just won't have entity stubs to anchor against.

7. **`build_financial_pipeline()` is preserved untouched.** The two-pass extraction pipeline remains available for use cases that require full graph extraction from documents (e.g., a bulk re-indexing job). It is not called during normal document ingestion in the new architecture.

---

## 9. LLM Call Budget (Revised)

| Stage | LLM calls | Tokens in (approx) | Tokens out (approx) |
|---|---|---|---|
| Lean chunk summarisation | 1 per non-trivial chunk | ~600 | 80 (hard cap) |
| Orchestrator planner | 1 per turn | ~800 | 300 |
| FundamentalAnalysisAgent (decomposer) | 0–1 per turn | ~2000 | 400 |
| FundamentalAnalysisAgent (analyst) | 1 per turn | ~3000 | 600 |
| NewsAnalysisAgent (execute + generate) | 1–3 per turn | ~2000 | 600 |
| Synthesiser | 1 per turn | ~4000 | 1200 |
| Write-back | **0** | — | — |
| ~~Full cognify (extract_financial_graph)~~ | ~~2 per chunk~~ | ~~eliminated for docs~~ | ~~eliminated~~ |
| ~~ingest_conversation + cognify~~ | ~~2 per chunk~~ | ~~eliminated~~ | ~~eliminated~~ |
