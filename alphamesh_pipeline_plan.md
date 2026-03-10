# AlphaMesh Pipeline Implementation Plan

## Overview

Two pipelines share a single graph/vector/relational storage layer:

- **Pipeline A — Document Ingestion**: Offline, zero LLM graph-extraction calls. Writes chunks and summaries to vector + relational stores only. Graph edges are NOT created here.
- **Pipeline B — User Conversation**: Online, user-facing. Retrieves from Pipeline A's outputs. Builds the knowledge graph incrementally through a post-synthesis write-back step.

Both pipelines share the same `DataPoint` schemas defined in `shared/models.py`. Any node written by either pipeline must use these schemas — this is the integration contract.

---

## Directory Structure

```
alphamesh/
├── shared/
│   ├── models.py              # All DataPoint schemas — shared by both pipelines
│   ├── canonical.py           # Canonical ID generation logic
│   └── storage.py             # Storage layer config (vector, graph, relational)
│
├── ingestion/
│   ├── pipeline.py            # Pipeline A entry point
│   ├── chunker.py             # Custom section-aware chunker for SEC filings
│   ├── summariser.py          # Custom summarisation task (domain-specific prompt)
│   └── loader.py              # Document loader — accepts metadata alongside file
│
├── conversation/
│   ├── pipeline.py            # Pipeline B entry point
│   ├── orchestrator.py        # Intent parsing + entity extraction + agent routing
│   ├── agents/
│   │   ├── base.py            # Abstract base agent
│   │   ├── news_agent.py      # Financial news retrieval + reasoning
│   │   └── financial_agent.py # 10-K / financial analysis retrieval + reasoning
│   ├── synthesiser.py         # CoT relationship block + user response generation
│   └── writeback.py           # Async graph write-back task
│
└── dedup/
    └── pipeline.py            # Deduplication pipeline (run before edge creation)
```

---

## Part 1 — Shared Data Models (`shared/models.py`)

All entities across both pipelines must be defined here. Do not define DataPoint subclasses anywhere else.

```python
from cognee.infrastructure.engine import DataPoint
from typing import Optional

# ── Core financial entities ──────────────────────────────────────────────────

class Company(DataPoint):
    """
    Stub created by orchestrator. Enriched by downstream agents.
    canonical_id format: "company::{ticker.upper()}"
    """
    __tablename__ = "company"
    canonical_id: str
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    description: Optional[str] = None          # enriched by financial_agent
    enriched: bool = False                     # set True after downstream agent writes
    metadata: dict = {"index_fields": ["ticker", "name", "description"]}


class FinancialEvent(DataPoint):
    """
    Macro or micro event affecting markets.
    canonical_id format: "event::{slug}" where slug is a normalised lowercase string
    e.g. "event::fed-rate-hike-2024-11"
    """
    __tablename__ = "financial_event"
    canonical_id: str
    name: str
    event_type: str                            # "macro", "earnings", "merger", "regulatory"
    event_date: Optional[str] = None
    description: Optional[str] = None
    enriched: bool = False
    metadata: dict = {"index_fields": ["name", "event_type", "description"]}


class FinancialConcept(DataPoint):
    """
    e.g. "discount rate", "yield curve inversion", "P/E ratio"
    canonical_id format: "concept::{slug}"
    """
    __tablename__ = "financial_concept"
    canonical_id: str
    name: str
    definition: Optional[str] = None
    enriched: bool = False
    metadata: dict = {"index_fields": ["name", "definition"]}


class DocumentChunk(DataPoint):
    """
    Produced exclusively by Pipeline A (document ingestion).
    Never created by conversation pipeline.
    canonical_id format: "chunk::{doc_id}::{chunk_index}"
    """
    __tablename__ = "document_chunk"
    canonical_id: str
    content: str
    doc_id: str                                # stable ID for the source document
    ticker: str
    doc_type: str                              # "10-K", "10-Q", "8-K", "news"
    section: Optional[str] = None             # e.g. "Risk Factors", "MD&A"
    fiscal_year: Optional[int] = None
    filing_date: Optional[str] = None
    chunk_index: int
    metadata: dict = {"index_fields": ["content", "ticker", "section"]}


class ChunkSummary(DataPoint):
    """
    Terse LLM-generated summary of a DocumentChunk.
    canonical_id format: "summary::{chunk.canonical_id}"
    Enables SearchType.SUMMARIES retrieval.
    """
    __tablename__ = "chunk_summary"
    canonical_id: str
    summary: str
    source_chunk_id: str                      # canonical_id of parent DocumentChunk
    ticker: str
    section: Optional[str] = None
    fiscal_year: Optional[int] = None
    metadata: dict = {"index_fields": ["summary", "ticker"]}


# ── Relationship model ────────────────────────────────────────────────────────

class EntityRelationship(DataPoint):
    """
    Produced exclusively by Pipeline B write-back.
    Represents a directed edge between two entities.
    canonical_id format: "rel::{from_id}::{relation_type}::{to_id}"
    """
    __tablename__ = "entity_relationship"
    canonical_id: str
    from_id: str                              # canonical_id of source entity
    from_type: str                            # DataPoint __tablename__ of source
    relation_type: str                        # e.g. "AFFECTED_BY", "REPORTED_BY", "INCREASES"
    to_id: str                                # canonical_id of target entity
    to_type: str                              # DataPoint __tablename__ of target
    confidence: str                           # "high" | "low"
    source_conversation_id: str              # for traceability
    metadata: dict = {"index_fields": ["relation_type", "from_id", "to_id"]}
```

---

## Part 2 — Canonical ID Generation (`shared/canonical.py`)

Canonical IDs are the deduplication key. Both pipelines must use these functions — never construct canonical IDs inline.

```python
import hashlib
import re

def company_id(ticker: str) -> str:
    return f"company::{ticker.strip().upper()}"

def event_id(name: str, date: str = None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if date:
        slug = f"{slug}::{date}"
    return f"event::{slug}"

def concept_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"concept::{slug}"

def chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"chunk::{doc_id}::{chunk_index}"

def summary_id(chunk_canonical_id: str) -> str:
    return f"summary::{chunk_canonical_id}"

def relationship_id(from_id: str, relation_type: str, to_id: str) -> str:
    return f"rel::{from_id}::{relation_type}::{to_id}"

def doc_id(file_path: str, ticker: str, doc_type: str, fiscal_year: int = None) -> str:
    """Stable document ID based on content identity, not file path."""
    key = f"{ticker}::{doc_type}::{fiscal_year or 'na'}::{file_path}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
```

---

## Part 3 — Pipeline A: Document Ingestion (`ingestion/pipeline.py`)

### Entry Point

```python
import asyncio
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.data.methods import load_or_create_datasets
from cognee.modules.users.methods import get_default_user
from cognee.low_level import setup
from cognee.tasks.storage import add_data_points

from ingestion.chunker import section_aware_chunk
from ingestion.summariser import summarise_chunks
from ingestion.loader import load_documents_with_metadata

async def run_ingestion_pipeline(
    file_paths: list[str],
    ticker: str,
    doc_type: str,                  # "10-K" | "10-Q" | "8-K" | "news"
    fiscal_year: int = None,
    filing_date: str = None,
):
    """
    Entry point for document ingestion. Call this once per document batch.
    Does NOT create graph edges. No LLM graph-extraction calls.
    LLM is used ONLY in summarise_chunks with a terse domain-specific prompt.
    """
    await setup()
    user = await get_default_user()
    datasets = await load_or_create_datasets(["alphamesh_documents"], [], user)
    dataset_id = datasets[0].id

    doc_metadata = {
        "ticker": ticker,
        "doc_type": doc_type,
        "fiscal_year": fiscal_year,
        "filing_date": filing_date,
    }

    pipeline = run_tasks(
        tasks=[
            Task(load_documents_with_metadata, metadata=doc_metadata),
            Task(section_aware_chunk),
            Task(summarise_chunks),
            Task(add_data_points),
        ],
        dataset_id=dataset_id,
        data=file_paths,
        user=user,
        pipeline_name="document_ingestion",
    )

    async for status in pipeline:
        print(f"[ingestion] {status}")
```

### Section-Aware Chunker (`ingestion/chunker.py`)

```python
import re
from shared.models import DocumentChunk
from shared.canonical import chunk_id, doc_id as make_doc_id

# SEC 10-K section header patterns
SEC_SECTION_PATTERNS = [
    (r"item\s+1[^a-z]", "Business"),
    (r"item\s+1a", "Risk Factors"),
    (r"item\s+1b", "Unresolved Staff Comments"),
    (r"item\s+2[^a-z]", "Properties"),
    (r"item\s+3[^a-z]", "Legal Proceedings"),
    (r"item\s+7[^a-z]", "MD&A"),
    (r"item\s+7a", "Quantitative Disclosures"),
    (r"item\s+8[^a-z]", "Financial Statements"),
    (r"item\s+9[^a-z]", "Controls and Procedures"),
]

MAX_CHUNK_CHARS = 1200      # target ~300 tokens per chunk at avg 4 chars/token

def detect_section(text: str) -> str | None:
    lower = text[:120].lower()
    for pattern, label in SEC_SECTION_PATTERNS:
        if re.search(pattern, lower):
            return label
    return None

def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split on paragraph boundaries first. If a paragraph exceeds max_chars,
    split on sentence boundaries. Preserves all characters (invertible).
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) <= max_chars:
                        buf += (" " if buf else "") + sent
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent
                if buf:
                    chunks.append(buf)
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks

async def section_aware_chunk(documents: list[dict]) -> list[DocumentChunk]:
    """
    Input:  list of dicts from load_documents_with_metadata
            Each dict: { "text": str, "file_path": str, "metadata": dict }
    Output: list of DocumentChunk DataPoints
    """
    datapoints = []
    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]
        stable_doc_id = make_doc_id(
            doc["file_path"], meta["ticker"],
            meta["doc_type"], meta.get("fiscal_year")
        )
        lines = text.split("\n")
        current_section = None
        section_buffer = ""
        section_chunks = []

        for line in lines:
            detected = detect_section(line)
            if detected:
                if section_buffer.strip():
                    section_chunks.append((current_section, section_buffer))
                current_section = detected
                section_buffer = line
            else:
                section_buffer += "\n" + line

        if section_buffer.strip():
            section_chunks.append((current_section, section_buffer))

        chunk_idx = 0
        for section_label, section_text in section_chunks:
            for chunk_text in split_into_chunks(section_text):
                datapoints.append(DocumentChunk(
                    canonical_id=chunk_id(stable_doc_id, chunk_idx),
                    content=chunk_text.strip(),
                    doc_id=stable_doc_id,
                    ticker=meta["ticker"],
                    doc_type=meta["doc_type"],
                    section=section_label,
                    fiscal_year=meta.get("fiscal_year"),
                    filing_date=meta.get("filing_date"),
                    chunk_index=chunk_idx,
                ))
                chunk_idx += 1

    return datapoints
```

### Summarisation Task (`ingestion/summariser.py`)

```python
from shared.models import DocumentChunk, ChunkSummary
from shared.canonical import summary_id
from cognee.infrastructure.llm.get_llm_client import get_llm_client

# Intentionally terse. No filler. Forces short output = fewer tokens.
FINANCIAL_SUMMARY_SYSTEM_PROMPT = """
You extract financial facts. Output 1-2 sentences only.
Include: company/ticker, metric or topic, value or direction, time period.
No preamble. No filler. If no financial fact is present, output: "No financial data."
""".strip()

async def summarise_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk | ChunkSummary]:
    """
    Generates a ChunkSummary for each DocumentChunk.
    Skips chunks where section is None (front matter, table of contents etc.)
    Returns both the original chunks AND the summaries so add_data_points
    receives everything in one pass.
    """
    llm = get_llm_client()
    output = list(chunks)    # preserve original chunks in output

    for chunk in chunks:
        if chunk.section is None:
            continue                         # skip unclassified sections
        if len(chunk.content) < 120:
            continue                         # skip trivially short chunks

        response = await llm.acreate(
            messages=[
                {"role": "system", "content": FINANCIAL_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": chunk.content},
            ],
            max_tokens=80,                   # hard cap — summaries must be terse
            temperature=0.0,
        )
        summary_text = response.choices[0].message.content.strip()

        if summary_text == "No financial data.":
            continue

        output.append(ChunkSummary(
            canonical_id=summary_id(chunk.canonical_id),
            summary=summary_text,
            source_chunk_id=chunk.canonical_id,
            ticker=chunk.ticker,
            section=chunk.section,
            fiscal_year=chunk.fiscal_year,
        ))

    return output
```

---

## Part 4 — Pipeline B: User Conversation (`conversation/pipeline.py`)

### Entry Point

```python
import asyncio
from conversation.orchestrator import parse_intent_and_route
from conversation.agents.news_agent import NewsAgent
from conversation.agents.financial_agent import FinancialAgent
from conversation.synthesiser import synthesise
from conversation.writeback import run_writeback

async def run_conversation_pipeline(
    user_message: str,
    conversation_id: str,
    conversation_history: list[dict],       # full prior turns for context
) -> str:
    """
    Returns the user-facing response string.
    Write-back to graph is fire-and-forget — does not block this return.
    """
    # Step 1: orchestrator
    intent = await parse_intent_and_route(user_message, conversation_history)
    # intent schema: {
    #     "goal": str,
    #     "entities": [{"canonical_id": str, "type": str, "name": str}],
    #     "agents_required": ["news_agent", "financial_agent"],  # subset
    #     "time_context": str | None
    # }

    # Step 2: run required downstream agents (can run in parallel)
    agent_map = {
        "news_agent": NewsAgent(),
        "financial_agent": FinancialAgent(),
    }
    agent_tasks = [
        agent_map[name].run(intent)
        for name in intent["agents_required"]
        if name in agent_map
    ]
    agent_outputs = await asyncio.gather(*agent_tasks)
    # agent_outputs: list of AgentOutput dicts
    # AgentOutput schema: {
    #     "agent": str,
    #     "findings": str,          # natural language findings
    #     "entities_enriched": list[DataPoint],  # enriched DataPoints for write-back
    #     "sources": list[str],     # chunk canonical_ids used
    # }

    # Step 3: synthesiser — CoT relationship extraction + user response
    synthesis = await synthesise(
        user_message=user_message,
        intent=intent,
        agent_outputs=agent_outputs,
        conversation_id=conversation_id,
    )
    # synthesis schema: {
    #     "response": str,                        # user-facing text
    #     "relationships": list[dict],            # parsed from <relationships> block
    #     "all_enriched_entities": list[DataPoint]
    # }

    # Step 4: fire-and-forget write-back (non-blocking)
    asyncio.create_task(run_writeback(
        relationships=synthesis["relationships"],
        enriched_entities=synthesis["all_enriched_entities"],
        conversation_id=conversation_id,
    ))

    return synthesis["response"]
```

### Orchestrator (`conversation/orchestrator.py`)

```python
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from shared.canonical import company_id, event_id, concept_id
import json

ORCHESTRATOR_SYSTEM_PROMPT = """
You are a financial query router. Extract structured intent from the user message.
Return ONLY valid JSON. No preamble. No markdown.

JSON schema:
{
  "goal": "<one sentence describing what the user wants>",
  "entities": [
    {"canonical_id": "<use format company::TICKER or event::slug or concept::slug>",
     "type": "company|event|concept",
     "name": "<as mentioned by user>"}
  ],
  "agents_required": ["news_agent", "financial_agent"],
  "time_context": "<fiscal year, quarter, date range, or null>"
}

agents_required rules:
- Always include "financial_agent" for any question about financials, filings, earnings
- Include "news_agent" for questions about recent events, market reactions, macro factors
- Include both if the question involves impact of external events on a company
""".strip()

async def parse_intent_and_route(
    user_message: str,
    conversation_history: list[dict],
) -> dict:
    llm = get_llm_client()
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        *conversation_history[-6:],          # last 3 turns for context, not full history
        {"role": "user", "content": user_message},
    ]
    response = await llm.acreate(
        messages=messages,
        max_tokens=300,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    intent = json.loads(raw)

    # Normalise canonical IDs using shared functions
    for entity in intent.get("entities", []):
        if entity["type"] == "company":
            ticker = entity["name"].upper()
            entity["canonical_id"] = company_id(ticker)
        elif entity["type"] == "event":
            entity["canonical_id"] = event_id(entity["name"])
        elif entity["type"] == "concept":
            entity["canonical_id"] = concept_id(entity["name"])

    return intent
```

### Base Agent (`conversation/agents/base.py`)

```python
from abc import ABC, abstractmethod
from cognee.modules.search.methods import search
from cognee.modules.search.types import SearchType

class BaseAgent(ABC):
    name: str = "base"
    search_type: SearchType = SearchType.CHUNKS

    async def retrieve(
        self,
        query: str,
        filters: dict = None,          # e.g. {"ticker": "AAPL", "doc_type": "10-K"}
        top_k: int = 8,
    ) -> list[dict]:
        """
        Retrieves relevant chunks or summaries from the vector store.
        filters are applied post-retrieval if cognee does not support native metadata filtering.
        """
        results = await search(
            query_text=query,
            search_type=self.search_type,
            top_k=top_k,
        )
        if filters:
            results = [
                r for r in results
                if all(getattr(r, k, None) == v for k, v in filters.items())
            ]
        return results

    @abstractmethod
    async def run(self, intent: dict) -> dict:
        """
        Must return AgentOutput dict:
        {
            "agent": str,
            "findings": str,
            "entities_enriched": list[DataPoint],
            "sources": list[str],
        }
        """
        pass
```

### Financial Agent (`conversation/agents/financial_agent.py`)

```python
from cognee.infrastructure.llm.get_llm_client import get_llm_client
from cognee.modules.search.types import SearchType
from shared.models import Company
from shared.canonical import company_id
from conversation.agents.base import BaseAgent

FINANCIAL_AGENT_SYSTEM_PROMPT = """
You are a financial analyst. You will receive excerpts from SEC filings.
Analyse them and produce a concise, factual answer to the user's question.
Cite specific metrics, values, and time periods where available.
Be terse. No filler.
""".strip()

class FinancialAgent(BaseAgent):
    name = "financial_agent"
    search_type = SearchType.CHUNKS

    async def run(self, intent: dict) -> dict:
        llm = get_llm_client()
        company_entities = [e for e in intent["entities"] if e["type"] == "company"]
        enriched_entities = []

        all_chunks = []
        for entity in company_entities:
            ticker = entity["name"].upper()
            chunks = await self.retrieve(
                query=intent["goal"],
                filters={"ticker": ticker, "doc_type": "10-K"},
                top_k=8,
            )
            all_chunks.extend(chunks)

            # Enrich company entity
            enriched_entities.append(Company(
                canonical_id=entity["canonical_id"],
                ticker=ticker,
                name=entity["name"],
                description=f"Company analysed for: {intent['goal']}",
                enriched=True,
            ))

        context = "\n\n---\n\n".join(c.content for c in all_chunks)
        response = await llm.acreate(
            messages=[
                {"role": "system", "content": FINANCIAL_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {intent['goal']}\n\nContext:\n{context}"},
            ],
            max_tokens=600,
            temperature=0.1,
        )
        findings = response.choices[0].message.content.strip()

        return {
            "agent": self.name,
            "findings": findings,
            "entities_enriched": enriched_entities,
            "sources": [c.canonical_id for c in all_chunks],
        }
```

### Synthesiser (`conversation/synthesiser.py`)

```python
from cognee.infrastructure.llm.get_llm_client import get_llm_client
import json, re

SYNTHESISER_SYSTEM_PROMPT = """
You are a financial synthesis assistant. You receive findings from multiple specialist agents.

YOUR TASK:
1. First, output a <relationships> block as JSON array identifying entity relationships
   found in the findings. This is your reasoning step — commit to each relationship
   explicitly before writing your response.
2. Then, output a <response> block with your final answer to the user.

RELATIONSHIPS FORMAT:
Each relationship must be:
{"from": "<canonical_id>", "from_name": "<name>", "relation": "<RELATION_TYPE>",
 "to": "<canonical_id>", "to_name": "<name>", "confidence": "high|low"}

Relation types: AFFECTS, CAUSED_BY, REPORTED_BY, INCREASES, DECREASES,
                CORRELATED_WITH, MITIGATES, EXPOSES_TO

CONFIDENCE:
- "high" = explicitly stated in findings with a value or direct attribution
- "low"  = inferred from context without explicit statement

OUTPUT FORMAT (strictly):
<relationships>
[...json array...]
</relationships>
<response>
...your answer to the user...
</response>

Do not output anything outside these two blocks.
""".strip()

async def synthesise(
    user_message: str,
    intent: dict,
    agent_outputs: list[dict],
    conversation_id: str,
) -> dict:
    llm = get_llm_client()

    findings_block = ""
    all_enriched = []
    for output in agent_outputs:
        findings_block += f"\n\n### {output['agent'].upper()} FINDINGS\n{output['findings']}"
        all_enriched.extend(output.get("entities_enriched", []))

    # Provide known entity canonical IDs to ground relationship extraction
    entity_ref = json.dumps([
        {"canonical_id": e["canonical_id"], "name": e["name"], "type": e["type"]}
        for e in intent["entities"]
    ], indent=2)

    user_content = (
        f"USER QUESTION: {user_message}\n\n"
        f"KNOWN ENTITIES:\n{entity_ref}\n"
        f"{findings_block}"
    )

    response = await llm.acreate(
        messages=[
            {"role": "system", "content": SYNTHESISER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1200,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()

    # Parse relationships block — fault-tolerant
    relationships = []
    rel_match = re.search(r"<relationships>(.*?)</relationships>", raw, re.DOTALL)
    if rel_match:
        try:
            relationships = json.loads(rel_match.group(1).strip())
        except json.JSONDecodeError:
            relationships = []     # malformed JSON must not break user response

    # Parse response block
    resp_match = re.search(r"<response>(.*?)</response>", raw, re.DOTALL)
    user_response = resp_match.group(1).strip() if resp_match else raw

    return {
        "response": user_response,
        "relationships": relationships,
        "all_enriched_entities": all_enriched,
        "conversation_id": conversation_id,
    }
```

### Write-Back Task (`conversation/writeback.py`)

```python
from shared.models import EntityRelationship
from shared.canonical import relationship_id
from cognee.tasks.storage import add_data_points
from dedup.pipeline import run_dedup

async def run_writeback(
    relationships: list[dict],
    enriched_entities: list,
    conversation_id: str,
):
    """
    Runs asynchronously after synthesiser returns user response.
    Steps:
    1. Dedup enriched entities against existing nodes
    2. Write enriched entities (upsert via canonical_id)
    3. Construct EntityRelationship DataPoints from relationship list
    4. Write relationship DataPoints

    Relationships are written AFTER entities — never before.
    A failure here must not propagate to the caller (fire-and-forget).
    """
    try:
        if not relationships and not enriched_entities:
            return

        # Step 1 + 2: dedup then write entities
        deduped_entities = await run_dedup(enriched_entities)
        if deduped_entities:
            await add_data_points(deduped_entities)

        # Step 3: construct relationship DataPoints
        rel_datapoints = []
        for rel in relationships:
            try:
                rel_datapoints.append(EntityRelationship(
                    canonical_id=relationship_id(
                        rel["from"], rel["relation"], rel["to"]
                    ),
                    from_id=rel["from"],
                    from_type=rel.get("from_type", "unknown"),
                    relation_type=rel["relation"],
                    to_id=rel["to"],
                    to_type=rel.get("to_type", "unknown"),
                    confidence=rel.get("confidence", "low"),
                    source_conversation_id=conversation_id,
                ))
            except KeyError:
                continue

        # Step 4: write relationships
        if rel_datapoints:
            await add_data_points(rel_datapoints)

    except Exception as e:
        # Log but never raise — write-back must not affect user response
        print(f"[writeback] error in conversation {conversation_id}: {e}")
```

---

## Part 5 — Deduplication Pipeline (`dedup/pipeline.py`)

```python
from cognee.infrastructure.databases.relational import get_relational_engine

async def run_dedup(entities: list) -> list:
    """
    Checks each entity's canonical_id against the relational DB.
    - canonical_id not found         → include as new write
    - canonical_id found, enriched=False → include as upsert (upgrade stub)
    - canonical_id found, enriched=True  → skip (already enriched, do not overwrite)

    Returns filtered list of entities to write.
    """
    engine = get_relational_engine()
    to_write = []

    for entity in entities:
        canonical_id = getattr(entity, "canonical_id", None)
        if not canonical_id:
            to_write.append(entity)
            continue

        table = entity.__tablename__
        existing = await engine.fetch_one(
            f"SELECT enriched FROM {table} WHERE canonical_id = :cid",
            {"cid": canonical_id}
        )

        if existing is None:
            to_write.append(entity)                   # new node
        elif not existing["enriched"]:
            to_write.append(entity)                   # upgrade stub to enriched
        # else: already enriched — skip

    return to_write
```

---

## Part 6 — Integration Contract

### What each pipeline reads and writes

| Pipeline | Reads from | Writes to |
|---|---|---|
| A (ingestion) | Raw files on disk | `document_chunk`, `chunk_summary` tables + vector index |
| B (conversation) | `document_chunk` + `chunk_summary` vector indices | `company`, `financial_event`, `financial_concept`, `entity_relationship` tables + graph |

### Invariants that must hold

1. **Schema contract**: Any node written by either pipeline must use a class from `shared/models.py`. No DataPoint subclasses elsewhere.

2. **Canonical ID contract**: All canonical IDs must be generated via `shared/canonical.py` functions. Never construct them inline. This is the deduplication key.

3. **Write ordering in Pipeline B**: Dedup must complete before `add_data_points` is called. The strict order is: `run_dedup` → `add_data_points(entities)` → `add_data_points(relationships)`. Relations are always written after nodes.

4. **Write-back is non-blocking**: `run_writeback` is always called via `asyncio.create_task(...)`, never awaited directly. A failure in write-back must never surface to the user — the try/except in `writeback.py` is mandatory.

5. **Graph is not queried during ingestion**: Pipeline A never calls `cognee.search()`. Pipeline B never calls ingestion pipeline tasks.

6. **LLM calls per pipeline**:
   - Pipeline A: 1 LLM call per non-trivial chunk, `max_tokens=80` (summarisation only)
   - Pipeline B: 1 orchestrator + 1 per agent + 1 synthesiser per user turn
   - Write-back: 0 LLM calls

---

## LLM Call Budget Summary

| Stage | LLM calls | Approx tokens in | Approx tokens out |
|---|---|---|---|
| Ingestion summarisation | 1 per chunk | ~400 | 80 (hard cap) |
| Orchestrator | 1 per turn | ~500 | 300 |
| Financial agent | 1 per invocation | ~2000 | 600 |
| News agent | 1 per invocation | ~2000 | 600 |
| Synthesiser | 1 per turn | ~3000 | 1200 |
| Write-back | **0** | — | — |
| ~~Cognee default graph extraction~~ | ~~1 per chunk~~ | ~~eliminated~~ | ~~eliminated~~ |
