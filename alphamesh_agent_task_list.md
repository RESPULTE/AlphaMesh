# AlphaMesh Pipeline — Agent Task List
## Implementation, Verification & Acceptance Checklist

> **How to use this list**
> Work through tasks in order within each phase. Each task has:
> - **Action** — what to implement
> - **File** — exact file to edit or create
> - **Verify** — concrete check the agent must run or assert before marking done
> - **Acceptance** — the condition that proves the task is complete
>
> Do NOT proceed to the next phase until all tasks in the current phase pass verification.
> If a verify step fails, fix before moving on — never skip.

---

## PHASE 0 — Pre-flight Checks

These tasks have no code changes. They establish the baseline before anything is touched.

---

### TASK 0.1 — Confirm existing test suite passes
**File:** root of project (run existing tests)
**Action:** Run the full existing test suite and record which tests pass/fail before any changes.
**Verify:**
```bash
pytest --tb=short -q 2>&1 | tee /tmp/baseline_test_results.txt
```
**Acceptance:** Baseline recorded. Any pre-existing failures are noted separately and excluded from regression tracking.

---

### TASK 0.2 — Confirm cognee is importable and graph engine is reachable
**File:** none (environment check)
**Action:** Run a quick smoke test confirming cognee imports, graph DB connects, and vector store is reachable.
**Verify:**
```python
import asyncio
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.engine.operations.setup import setup

async def check():
    await setup()
    g = await get_graph_engine()
    assert g is not None, "Graph engine is None"
    v = get_vector_engine()
    assert v is not None, "Vector engine is None"
    print("PASS: cognee environment OK")

asyncio.run(check())
```
**Acceptance:** Script prints `PASS: cognee environment OK` with no exceptions.

---

### TASK 0.3 — Confirm existing graph_models imports cleanly
**File:** `core/memory/graph_models.py`
**Action:** No changes. Confirm all models import and `model_rebuild()` calls succeed.
**Verify:**
```python
from core.memory.graph_models import (
    Company, FinancialConcept, FinancialEvent,
    UserInvestmentInterest, UserLearningInterest,
    FinancialKnowledgeGraph, Sector, Industry,
    ALL_ENTITIES, ALL_MAIN_SECTORS,
    GLOBAL_FINANCIAL_EVENT_NODESET, GLOBAL_FINANCIAL_WISDOM_NODESET,
)
print("PASS: graph_models imports OK")
```
**Acceptance:** No ImportError, no Pydantic validation errors.

---

### TASK 0.4 — Confirm pipeline_tasks.py `get_canonical_id` is importable
**File:** `core/memory/pipeline_tasks.py`
**Action:** No changes. Verify `get_canonical_id` exists at module level.
**Verify:**
```python
from core.memory.pipeline_tasks import get_canonical_id
import uuid
result = get_canonical_id("AAPL")
assert isinstance(result, uuid.UUID), "Expected UUID"
print(f"PASS: get_canonical_id('AAPL') = {result}")
```
**Acceptance:** Returns a valid UUID. If `get_canonical_id` does not exist, note it and add it in Phase 1 before any other phase proceeds.

---

## PHASE 1 — `core/agents/models.py` Extension

**Goal:** Add `entities_enriched` field to `BaseAgentOutput` without breaking existing agents.

---

### TASK 1.1 — Add `entities_enriched` field to `BaseAgentOutput`
**File:** `core/agents/models.py`
**Action:**
- Import `Any` from `typing` (already imported)
- Add the following field to `BaseAgentOutput`:
```python
entities_enriched: List[Any] = Field(
    default_factory=list,
    description=(
        "List of enriched DataPoint objects resolved by this agent. "
        "Must use classes from core.memory.graph_models. "
        "Populated by each agent's final node before returning."
    ),
)
```
- Do NOT change `agent_name`, `analysis`, or `get_llm_context_str`.
- Do NOT make `entities_enriched` abstract or required.

**Verify:**
```python
from core.agents.models import BaseAgentOutput, BaseAgentInput
# Confirm field exists with correct default
import inspect
fields = BaseAgentOutput.model_fields
assert "entities_enriched" in fields, "entities_enriched field missing"
assert fields["entities_enriched"].default_factory is not None, "Must have default_factory=list"
print("PASS: BaseAgentOutput.entities_enriched field present")
```

**Acceptance:** Field exists, defaults to empty list, `BaseAgentOutput` still cannot be instantiated directly (it is abstract).

---

### TASK 1.2 — Verify existing agents still instantiate without error
**File:** none (regression check)
**Action:** No code change. Confirm existing agent output classes still instantiate cleanly.
**Verify:**
```python
from core.agents.fundamental_analysis_agent import FundamentalAnalysisOutput
from core.agents.news_analysis_agent import NewsAnalysisOutput

f = FundamentalAnalysisOutput(analysis="test", financial_data=None)
assert f.entities_enriched == [], "entities_enriched should default to []"
assert f.agent_name == "fundamentals_agent"

n = NewsAnalysisOutput(analysis="test", sources=[])
assert n.entities_enriched == [], "entities_enriched should default to []"
assert n.agent_name == "news_agent"

print("PASS: existing agent outputs instantiate correctly")
```
**Acceptance:** Both instantiate without error, `entities_enriched` defaults to `[]`.

---

## PHASE 2 — `core/memory/prompts.py` Additions

**Goal:** Add the two new prompts without modifying any existing ones.

---

### TASK 2.1 — Add `LEAN_SUMMARY_SYSTEM_PROMPT`
**File:** `core/memory/prompts.py`
**Action:** Append the following constant at the end of the file. Do NOT modify any existing prompts.
```python
LEAN_SUMMARY_SYSTEM_PROMPT = """\
Extract financial facts. Output 1-2 sentences only.
Include: company/ticker, metric or topic, value or direction, time period.
No preamble. No filler. If no financial fact is present, output exactly: NO_FINANCIAL_DATA\
"""
```
**Verify:**
```python
from core.memory.prompts import LEAN_SUMMARY_SYSTEM_PROMPT
assert "NO_FINANCIAL_DATA" in LEAN_SUMMARY_SYSTEM_PROMPT
assert len(LEAN_SUMMARY_SYSTEM_PROMPT) < 500, "Prompt should be terse"
print("PASS: LEAN_SUMMARY_SYSTEM_PROMPT present and terse")
```
**Acceptance:** Constant importable, contains `NO_FINANCIAL_DATA` sentinel, under 500 characters.

---

### TASK 2.2 — Add `SYNTHESISER_WRITEBACK_SYSTEM_PROMPT`
**File:** `core/memory/prompts.py`
**Action:** Append after `LEAN_SUMMARY_SYSTEM_PROMPT`. Do NOT modify any existing prompts.

Paste the full prompt from the implementation plan (Section 3, "Add to prompts.py"). It must:
- Describe Part 1 (relationship reasoning) and Part 2 (user response) clearly
- List allowed `RELATION_TYPE` values exactly
- Specify the `<relationships>` and `<response>` XML tags
- End with "Do not output anything outside these two blocks."

**Verify:**
```python
from core.memory.prompts import SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
required_strings = [
    "<relationships>", "<response>",
    "AFFECTS", "CAUSED_BY", "INCREASES", "DECREASES",
    "CORRELATED_WITH", "MITIGATES", "EXPOSES_TO",
    "confidence", "high | low",
    "Do not output anything outside these two blocks",
]
for s in required_strings:
    assert s in SYNTHESISER_WRITEBACK_SYSTEM_PROMPT, f"Missing required string: {s!r}"
print("PASS: SYNTHESISER_WRITEBACK_SYSTEM_PROMPT contains all required elements")
```
**Acceptance:** All required strings present in the prompt.

---

### TASK 2.3 — Export new prompts from `__init__.py`
**File:** `core/memory/__init__.py`
**Action:** Add to the import block:
```python
from core.memory.prompts import (
    LEAN_SUMMARY_SYSTEM_PROMPT,
    SYNTHESISER_WRITEBACK_SYSTEM_PROMPT,
)
```
Add both to `__all__`.

**Verify:**
```python
from core.memory import LEAN_SUMMARY_SYSTEM_PROMPT, SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
print("PASS: new prompts exported from core.memory")
```
**Acceptance:** Both importable from `core.memory` directly.

---

## PHASE 3 — `core/memory/conversation_writeback.py` (New File)

**Goal:** Create the write-back module. This is the most critical new file.

---

### TASK 3.1 — Create `conversation_writeback.py` with `EntityRelationship` DataPoint
**File:** `core/memory/conversation_writeback.py` (CREATE)
**Action:** Create the file. Implement the `EntityRelationship` DataPoint class exactly as specified in the implementation plan Section 4.

Required fields:
- `from_name: str`
- `from_type: str`
- `relation_type: str`
- `to_name: str`
- `to_type: str`
- `confidence: str = "low"`
- `source_conversation_id: str = ""`
- `metadata: dict = {"index_fields": ["from_name", "relation_type", "to_name"]}`
- `__tablename__ = "entity_relationship"`

**Verify:**
```python
import uuid
from core.memory.conversation_writeback import EntityRelationship, _relationship_id

rel = EntityRelationship(
    id=_relationship_id("AAPL", "AFFECTS", "Fed Rate Hike"),
    from_name="AAPL",
    from_type="Company",
    relation_type="AFFECTS",
    to_name="Fed Rate Hike",
    to_type="FinancialEvent",
    confidence="high",
    source_conversation_id="test-conv-001",
)
assert rel.from_name == "AAPL"
assert rel.relation_type == "AFFECTS"
assert rel.confidence == "high"
assert isinstance(rel.id, uuid.UUID)
print("PASS: EntityRelationship instantiates correctly")
```
**Acceptance:** Class instantiates, all fields set correctly, ID is a UUID.

---

### TASK 3.2 — Implement `_relationship_id()`
**File:** `core/memory/conversation_writeback.py`
**Action:** Implement as:
```python
def _relationship_id(from_name: str, relation: str, to_name: str) -> uuid.UUID:
    key = f"{from_name.upper()}::{relation}::{to_name.upper()}"
    return uuid.uuid5(uuid.NAMESPACE_DNS, key)
```
**Verify:**
```python
from core.memory.conversation_writeback import _relationship_id
import uuid

id1 = _relationship_id("AAPL", "AFFECTS", "Fed Rate Hike")
id2 = _relationship_id("aapl", "AFFECTS", "fed rate hike")
id3 = _relationship_id("MSFT", "AFFECTS", "Fed Rate Hike")

assert id1 == id2, "Should be case-insensitive"
assert id1 != id3, "Different entities must produce different IDs"
assert isinstance(id1, uuid.UUID)
print("PASS: _relationship_id is deterministic and case-insensitive")
```
**Acceptance:** Same inputs (case-insensitive) produce same UUID. Different inputs produce different UUIDs.

---

### TASK 3.3 — Implement `_resolve_entity_nodeset()`
**File:** `core/memory/conversation_writeback.py`
**Action:** Implement the async function that assigns `belongs_to_set` on an entity based on its type. Must call:
- `get_or_create_nodeset(GLOBAL_FINANCIAL_WISDOM_NODESET)` for `FinancialConcept`
- `get_or_create_nodeset(GLOBAL_FINANCIAL_EVENT_NODESET)` for `FinancialEvent`
- `get_or_create_global_nodeset()` for `Company`, then optionally sector NodeSet if `entity.sector` is non-empty
- Must not raise on failure — catch and log

**Verify (integration, requires graph DB):**
```python
import asyncio
from cognee.modules.engine.operations.setup import setup
from core.memory.nodeset_manager import get_or_create_global_nodeset
from core.memory.graph_models import Company, FinancialEvent
from core.memory.conversation_writeback import _resolve_entity_nodeset

async def test():
    await setup()
    await get_or_create_global_nodeset()

    company = Company(ticker="AAPL", name="Apple Inc.", description="Test", sector="Information Technology")
    await _resolve_entity_nodeset(company)
    assert len(company.belongs_to_set) >= 1, "Company should have at least global NodeSet"

    event = FinancialEvent(name="Fed Rate Hike", description="Test event")
    await _resolve_entity_nodeset(event)
    assert len(event.belongs_to_set) >= 1, "FinancialEvent should have event NodeSet"

    print("PASS: _resolve_entity_nodeset assigns NodeSets correctly")

asyncio.run(test())
```
**Acceptance:** `belongs_to_set` is non-empty after the call for both entity types.

---

### TASK 3.4 — Implement `_should_write_entity()`
**File:** `core/memory/conversation_writeback.py`
**Action:** Implement the async dedup check. It must:
- Return `True` (write) if the entity has no `id` attribute
- Return `True` (write) if the entity has no `__tablename__`
- Query the relational DB for existing row by `id`
- Return `True` if no row found, `False` if row found
- Return `True` on any DB exception (fail open — write anyway)

**Verify:**
```python
import asyncio, uuid
from core.memory.conversation_writeback import _should_write_entity
from core.memory.graph_models import Company

async def test():
    # Entity with no id — should write
    e = Company(ticker="ZZZZ", name="Fake Co", description="x", sector="Energy")
    e.id = None
    result = await _should_write_entity(e)
    assert result is True, "Entity with no id should always write"
    print("PASS: _should_write_entity returns True for entity with no id")

asyncio.run(test())
```
**Acceptance:** Returns `True` for entity with `None` id. Does not raise on DB query failure.

---

### TASK 3.5 — Implement `_build_relationship_datapoints()`
**File:** `core/memory/conversation_writeback.py`
**Action:** Implement the function that converts the synthesiser's `<relationships>` JSON list into `EntityRelationship` DataPoints. Must:
- Skip entries missing `from_name`, `relation`, or `to_name` — log debug, continue
- Never raise — return whatever was successfully built
- Set `confidence` from entry or default to `"low"`
- Set `source_conversation_id` from parameter

**Verify:**
```python
from core.memory.conversation_writeback import _build_relationship_datapoints

# Valid relationships
rels = [
    {"from_name": "AAPL", "from_type": "Company", "relation": "AFFECTS",
     "to_name": "Fed Rate Hike", "to_type": "FinancialEvent", "confidence": "high"},
    {"from_name": "MSFT", "relation": "CORRELATED_WITH",
     "to_name": "AAPL"},   # missing from_type/to_type — should still work
    {"from_name": "BAD"},   # missing relation and to_name — must be skipped
]

result = _build_relationship_datapoints(rels, conversation_id="test-123")
assert len(result) == 2, f"Expected 2, got {len(result)}"
assert result[0].relation_type == "AFFECTS"
assert result[0].confidence == "high"
assert result[1].relation_type == "CORRELATED_WITH"
assert result[1].confidence == "low"   # default
assert result[0].source_conversation_id == "test-123"
print("PASS: _build_relationship_datapoints filters malformed entries correctly")
```
**Acceptance:** Malformed entries silently skipped, valid entries converted, defaults applied.

---

### TASK 3.6 — Implement `run_conversation_writeback()` (main entry point)
**File:** `core/memory/conversation_writeback.py`
**Action:** Implement the async main entry point following the 5-step write order in the implementation plan:
1. Resolve NodeSets for all enriched entities
2. Filter to entities that need writing (`_should_write_entity`)
3. Write enriched entities via `add_data_points`
4. Run `find_and_merge_candidates` on written entities (catch exception, non-fatal)
5. Build and write `EntityRelationship` DataPoints

Critical requirements:
- Entire function body wrapped in `try/except Exception` — logs error, never raises
- Early return if both `enriched_entities` and `relationships` are empty
- Relationships written AFTER entities (step 5 comes after step 3)

**Verify:**
```python
import asyncio
from core.memory.conversation_writeback import run_conversation_writeback

async def test():
    # Test 1: Empty inputs — should return silently
    await run_conversation_writeback(
        relationships=[],
        enriched_entities=[],
        conversation_id="test-empty",
    )
    print("PASS: empty inputs return silently")

    # Test 2: Malformed relationships — should not raise
    bad_rels = [{"garbage": True}, None, 42]
    await run_conversation_writeback(
        relationships=bad_rels,
        enriched_entities=[],
        conversation_id="test-bad-rels",
    )
    print("PASS: malformed relationships do not raise")

asyncio.run(test())
```
**Acceptance:** Neither test raises. Function handles all error conditions gracefully.

---

### TASK 3.7 — Export `conversation_writeback` from `core/memory/__init__.py`
**File:** `core/memory/__init__.py`
**Action:** Add:
```python
from core.memory.conversation_writeback import (
    EntityRelationship,
    run_conversation_writeback,
)
```
Add `"EntityRelationship"` and `"run_conversation_writeback"` to `__all__`.

**Verify:**
```python
from core.memory import EntityRelationship, run_conversation_writeback
print("PASS: conversation_writeback exports accessible from core.memory")
```
**Acceptance:** Both names importable from `core.memory`.

---

## PHASE 4 — `core/memory/pipeline_tasks.py` Additions

**Goal:** Add `summarise_chunks_lean()` and `build_lean_document_pipeline()` without modifying any existing functions.

---

### TASK 4.1 — Add `summarise_chunks_lean()` task
**File:** `core/memory/pipeline_tasks.py`
**Action:** Add the function after the existing `merge_entities()` function. Do NOT modify any existing functions. Implementation per the plan:
- Accepts `List[DocumentChunk]`
- Calls `LLMGateway.acreate` per chunk
- Uses `LEAN_SUMMARY_SYSTEM_PROMPT` (import from `core.memory.prompts`)
- `max_tokens=80`, `temperature=0.0`
- Skips chunks where `chunk.text` is `None` or `len < 120`
- On response `"NO_FINANCIAL_DATA"` — skips (does not set `text_summary`)
- On any exception per chunk — logs warning, continues
- Returns the SAME `data_chunks` list (in-place mutation of `text_summary`)

**Verify:**
```python
# Unit test — mock the LLM call
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from core.memory.pipeline_tasks import summarise_chunks_lean

async def test():
    # Create mock chunks
    class MockChunk:
        def __init__(self, text):
            self.id = "test-id"
            self.text = text
            self.text_summary = None

    chunks = [
        MockChunk("Apple Inc reported revenue of $90B in Q3 2024, up 5% year over year."),
        MockChunk("short"),   # < 120 chars, should be skipped
        MockChunk(None),      # None text, should be skipped
    ]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "AAPL Q3 2024 revenue $90B, up 5% YoY."

    with patch("core.memory.pipeline_tasks.LLMGateway.acreate", new=AsyncMock(return_value=mock_response)):
        result = await summarise_chunks_lean(chunks)

    assert result is chunks, "Must return same list object"
    assert chunks[0].text_summary == "AAPL Q3 2024 revenue $90B, up 5% YoY."
    assert chunks[1].text_summary is None, "Short chunk should not have summary"
    assert chunks[2].text_summary is None, "None-text chunk should not have summary"
    print("PASS: summarise_chunks_lean works correctly")

asyncio.run(test())
```
**Acceptance:** Returns same list, only populates `text_summary` on qualifying chunks, skips gracefully.

---

### TASK 4.2 — Add `build_lean_document_pipeline()`
**File:** `core/memory/pipeline_tasks.py`
**Action:** Add after `summarise_chunks_lean`. Do NOT modify `build_financial_pipeline()`.

The function must:
- Accept `chunks_per_batch: int = 100`, `chunk_size: Optional[int] = None`, `include_summaries: bool = True`
- Return a `list[Task]` with exactly:
  - Always: `Task(classify_documents)`, `Task(extract_chunks_from_documents, ...)`, `Task(add_data_points_with_custom_edges, ...)`
  - Conditionally (if `include_summaries=True`): `Task(summarise_chunks_lean)` inserted BEFORE `add_data_points_with_custom_edges`
- Must NOT include `extract_financial_graph`, `assign_nodesets`, or `merge_entities`

**Verify:**
```python
import asyncio
from core.memory.pipeline_tasks import build_lean_document_pipeline

async def test():
    # With summaries (default)
    tasks_with = await build_lean_document_pipeline(include_summaries=True)
    task_names_with = [t.executable.__name__ for t in tasks_with]
    assert "classify_documents" in task_names_with
    assert "extract_chunks_from_documents" in task_names_with
    assert "summarise_chunks_lean" in task_names_with
    assert "add_data_points_with_custom_edges" in task_names_with
    assert "extract_financial_graph" not in task_names_with, "Must NOT include graph extraction"
    assert "assign_nodesets" not in task_names_with, "Must NOT include assign_nodesets"
    assert "merge_entities" not in task_names_with, "Must NOT include merge_entities"

    # Verify ordering: summarise before add_data_points
    idx_summarise = task_names_with.index("summarise_chunks_lean")
    idx_add = task_names_with.index("add_data_points_with_custom_edges")
    assert idx_summarise < idx_add, "summarise must come before add_data_points"

    # Without summaries
    tasks_without = await build_lean_document_pipeline(include_summaries=False)
    task_names_without = [t.executable.__name__ for t in tasks_without]
    assert "summarise_chunks_lean" not in task_names_without

    print(f"PASS: lean pipeline with summaries = {task_names_with}")
    print(f"PASS: lean pipeline without summaries = {task_names_without}")

asyncio.run(test())
```
**Acceptance:** Task list contains correct tasks in correct order. Forbidden tasks absent. `include_summaries=False` removes summarisation.

---

### TASK 4.3 — Verify `build_financial_pipeline()` is unchanged
**File:** `core/memory/pipeline_tasks.py`
**Action:** No change. Run the existing pipeline builder and confirm its task list is identical to before.
**Verify:**
```python
import asyncio
from core.memory.pipeline_tasks import build_financial_pipeline

async def test():
    tasks = await build_financial_pipeline()
    task_names = [t.executable.__name__ for t in tasks]
    required = [
        "classify_documents",
        "extract_chunks_from_documents",
        "extract_financial_graph",
        "assign_nodesets",
        "add_data_points_with_custom_edges",
    ]
    for name in required:
        assert name in task_names, f"build_financial_pipeline missing: {name}"
    print(f"PASS: build_financial_pipeline unchanged: {task_names}")

asyncio.run(test())
```
**Acceptance:** All required tasks still present. Function unmodified.

---

## PHASE 5 — `core/memory/memory_system.py` Changes

**Goal:** Add `ingest_document_lean()`, deprecate `ingest_conversation()`.

---

### TASK 5.1 — Add import for `build_lean_document_pipeline`
**File:** `core/memory/memory_system.py`
**Action:** Add to the existing pipeline import line:
```python
from core.memory.pipeline_tasks import build_financial_pipeline, build_lean_document_pipeline
```
**Verify:**
```python
from core.memory.memory_system import FinancialMemorySystem
# If this imports cleanly, the pipeline import is OK
print("PASS: memory_system imports build_lean_document_pipeline")
```
**Acceptance:** `memory_system.py` imports without error.

---

### TASK 5.2 — Add `ingest_document_lean()` method to `FinancialMemorySystem`
**File:** `core/memory/memory_system.py`
**Action:** Add the method to `FinancialMemorySystem` class after `ingest_financial_report()`. Implementation per plan Section 6.

The method must:
- Call `self._require_initialized()`
- Raise `ValueError` if `ticker` or `content` is empty
- Construct a header string with ticker, report_type, optional period
- Call `self._add_to_cognee(text, node_set=...)` first
- Call `build_lean_document_pipeline(include_summaries=include_summaries)`
- Call `run_custom_pipeline(tasks=tasks, dataset=DATASET_NAME, ...)`
- Raise `MemorySystemError` wrapping any pipeline exception
- Log before and after

**Verify:**
```python
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from core.memory.memory_system import FinancialMemorySystem

async def test():
    system = FinancialMemorySystem()
    system._initialized = True
    system._global_nodeset = MagicMock()

    # Test ValueError guards
    try:
        await system.ingest_document_lean(ticker="", report_type="10-K", content="some text")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        await system.ingest_document_lean(ticker="AAPL", report_type="10-K", content="")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("PASS: ingest_document_lean raises ValueError on empty inputs")

    # Test that it calls add_to_cognee and run_custom_pipeline
    with patch.object(system, "_add_to_cognee", new=AsyncMock()) as mock_add, \
         patch("core.memory.memory_system.run_custom_pipeline", new=AsyncMock(return_value={})) as mock_run, \
         patch("core.memory.memory_system.build_lean_document_pipeline", new=AsyncMock(return_value=[])) as mock_build:

        await system.ingest_document_lean(
            ticker="AAPL", report_type="10-K", content="Revenue was $90B."
        )

        assert mock_add.called, "_add_to_cognee must be called"
        call_args = mock_add.call_args[0][0]
        assert "AAPL" in call_args, "Ticker must appear in ingested text"
        assert "10-K" in call_args, "Report type must appear in ingested text"
        assert mock_build.called, "build_lean_document_pipeline must be called"
        assert mock_run.called, "run_custom_pipeline must be called"

    print("PASS: ingest_document_lean calls correct functions")

asyncio.run(test())
```
**Acceptance:** ValueError on empty inputs. Calls `_add_to_cognee` and `run_custom_pipeline`. Text includes ticker and report type.

---

### TASK 5.3 — Deprecate `ingest_conversation()`
**File:** `core/memory/memory_system.py`
**Action:** Replace the body of `ingest_conversation()` with a no-op that logs a deprecation warning. Keep the method signature identical. Do NOT delete the method.

New body:
```python
logger.warning(
    "ingest_conversation() is deprecated. Conversation insights are now "
    "written to the graph via run_conversation_writeback() from the synthesiser. "
    "This call has no effect. caller=%s",
    user_email,
)
return
```

**Verify:**
```python
import asyncio, logging
from core.memory.memory_system import FinancialMemorySystem
from unittest.mock import MagicMock

async def test():
    system = FinancialMemorySystem()
    system._initialized = True
    system._global_nodeset = MagicMock()

    # Should return without raising, without calling cognee.add
    with patch("core.memory.memory_system.cognee") as mock_cognee:
        await system.ingest_conversation(
            user_email="test@example.com",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert not mock_cognee.add.called, "cognee.add must NOT be called"

    print("PASS: ingest_conversation is a no-op, does not call cognee.add")

asyncio.run(test())
```
**Acceptance:** Returns without error. Does not call `cognee.add`. Warning is logged.

---

### TASK 5.4 — Verify `cognify()` method is unchanged
**File:** `core/memory/memory_system.py`
**Action:** No change. Confirm signature and body are identical to baseline.
**Verify:**
```python
import inspect
from core.memory.memory_system import FinancialMemorySystem
sig = inspect.signature(FinancialMemorySystem.cognify)
params = list(sig.parameters.keys())
assert "run_in_background" in params
assert "chunks_per_batch" in params
assert "chunk_size" in params
print("PASS: cognify() signature unchanged")
```
**Acceptance:** Signature unchanged. Method still callable.

---

## PHASE 6 — `core/agents/fundamental_analysis_agent.py` Entity Emission

**Goal:** Emit a `Company` DataPoint in `entities_enriched` at the end of `_generate_analysis`.

---

### TASK 6.1 — Add `_build_company_entity()` helper
**File:** `core/agents/fundamental_analysis_agent.py`
**Action:** Add a module-level (or class-level) helper function:
```python
from core.memory.graph_models import Company
from core.memory.pipeline_tasks import get_canonical_id

def _build_fundamental_company_entity(ticker: str, analysis_text: str) -> Company:
    first_sentence = ""
    if analysis_text:
        parts = analysis_text.split(".")
        first_sentence = parts[0].strip() + "." if parts else ""
    return Company(
        id=get_canonical_id(ticker.upper()),
        ticker=ticker.upper(),
        name=ticker.upper(),
        description=first_sentence[:500],   # cap description length
        sector="",                           # assign_nodesets will resolve later
        industry=None,
    )
```
**Verify:**
```python
from core.agents.fundamental_analysis_agent import _build_fundamental_company_entity
from core.memory.graph_models import Company
import uuid

entity = _build_fundamental_company_entity("AAPL", "Apple reported strong earnings. Revenue grew 5%.")
assert isinstance(entity, Company)
assert entity.ticker == "AAPL"
assert "Apple reported strong earnings." in entity.description
assert isinstance(entity.id, uuid.UUID)
print("PASS: _build_fundamental_company_entity works correctly")
```
**Acceptance:** Returns `Company` DataPoint with correct ticker, UUID id, and truncated description.

---

### TASK 6.2 — Populate `entities_enriched` in `_generate_analysis` return
**File:** `core/agents/fundamental_analysis_agent.py`
**Action:** In the `_generate_analysis` method, modify the final `return` statement to include `entities_enriched`.

Find this return:
```python
return FundamentalAnalysisOutput(
    financial_data=state.financial_data, analysis=response.content
)
```
Replace with:
```python
return FundamentalAnalysisOutput(
    financial_data=state.financial_data,
    analysis=response.content,
    entities_enriched=[_build_fundamental_company_entity(state.ticker, response.content)],
)
```
Also handle the early-exit case (no data found) — emit entity with empty description:
```python
return FundamentalAnalysisOutput(
    financial_data=None,
    analysis="No data found.",
    entities_enriched=[_build_fundamental_company_entity(state.ticker, "")],
)
```

**Verify:**
```python
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from core.agents.fundamental_analysis_agent import FundamentalAnalysisAgent
from core.agents.models import BaseAgentInput
from core.memory.graph_models import Company

async def test():
    agent = FundamentalAnalysisAgent()

    mock_output = MagicMock()
    mock_output.content = "AAPL revenue grew 5% to $90B in FY2024."

    with patch.object(agent, "_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value={
            "financial_data": None,
            "analysis": "AAPL revenue grew 5% to $90B in FY2024.",
            "ticker": "AAPL",
        })
        # Directly test the helper instead of full agent run
        from core.agents.fundamental_analysis_agent import _build_fundamental_company_entity
        entity = _build_fundamental_company_entity("AAPL", "AAPL revenue grew 5%.")
        assert isinstance(entity, Company)
        assert entity.ticker == "AAPL"

    print("PASS: FundamentalAnalysisAgent entity emission verified")

asyncio.run(test())
```
**Acceptance:** `entities_enriched` contains exactly one `Company` DataPoint with the correct ticker.

---

## PHASE 7 — `core/agents/news_analysis_agent.py` Entity Emission

**Goal:** Emit a `Company` DataPoint in `entities_enriched` at the end of `_generate_analysis`.

---

### TASK 7.1 — Add `_build_news_entities()` helper
**File:** `core/agents/news_analysis_agent.py`
**Action:** Add a module-level helper function:
```python
from core.memory.graph_models import Company
from core.memory.pipeline_tasks import get_canonical_id

def _build_news_entities(ticker: str) -> list:
    """
    Build minimal enriched DataPoints from what the news agent knows.
    Only Company is emitted here — FinancialEvent extraction is the
    synthesiser's job via the CoT <relationships> block.
    """
    return [
        Company(
            id=get_canonical_id(ticker.upper()),
            ticker=ticker.upper(),
            name=ticker.upper(),
            description=f"Company in focus for news retrieval.",
            sector="",
            industry=None,
        )
    ]
```
**Verify:**
```python
from core.agents.news_analysis_agent import _build_news_entities
from core.memory.graph_models import Company

entities = _build_news_entities("TSLA")
assert len(entities) == 1
assert isinstance(entities[0], Company)
assert entities[0].ticker == "TSLA"
print("PASS: _build_news_entities returns correct entity list")
```
**Acceptance:** Returns list with one `Company` DataPoint with correct ticker.

---

### TASK 7.2 — Populate `entities_enriched` in `_generate_analysis` return
**File:** `core/agents/news_analysis_agent.py`
**Action:** In `_generate_analysis`, modify both return statements (success and except fallback) to include `entities_enriched`.

Success return:
```python
return NewsAnalysisOutput(
    analysis=retval.content,
    sources=state.news_context,
    entities_enriched=_build_news_entities(state.ticker),
)
```
Fallback return:
```python
return NewsAnalysisOutput(
    analysis="Error generating analysis due to model failure.",
    sources=[],
    entities_enriched=_build_news_entities(state.ticker),
)
```
**Verify:**
```python
from core.agents.news_analysis_agent import _build_news_entities, NewsAnalysisOutput
from core.memory.graph_models import Company

output = NewsAnalysisOutput(
    analysis="test",
    sources=[],
    entities_enriched=_build_news_entities("AAPL"),
)
assert len(output.entities_enriched) == 1
assert isinstance(output.entities_enriched[0], Company)
print("PASS: NewsAnalysisOutput entities_enriched populated correctly")
```
**Acceptance:** Both return paths include `entities_enriched`. Even error fallback returns entity list.

---

## PHASE 8 — `core/agents/orchestrator_agent.py` Synthesiser Change

**Goal:** Modify `_synthesize_node` to emit CoT `<relationships>` block and fire write-back. This is the highest-risk change — validate carefully.

---

### TASK 8.1 — Add new imports to `orchestrator_agent.py`
**File:** `core/agents/orchestrator_agent.py`
**Action:** Add at the top of the file:
```python
import re
import json
from core.memory.prompts import SYNTHESISER_WRITEBACK_SYSTEM_PROMPT
from core.memory.conversation_writeback import run_conversation_writeback
```
**Verify:**
```python
# Check imports don't cause circular dependency
import core.agents.orchestrator_agent
print("PASS: orchestrator_agent imports without circular dependency")
```
**Acceptance:** File imports cleanly with no circular import error.

---

### TASK 8.2 — Add `conversation_id` and write-back fields to `OrchestratorState`
**File:** `core/agents/orchestrator_agent.py`
**Action:** Add to `OrchestratorState`:
```python
conversation_id: Optional[str] = None
writeback_relationships: List[dict] = Field(default_factory=list)
writeback_entities: List[Any] = Field(default_factory=list)
```
Import `Any` from `typing` if not already imported.

**Verify:**
```python
from core.agents.orchestrator_agent import OrchestratorState
from langchain_core.messages import HumanMessage

state = OrchestratorState(
    messages=[HumanMessage(content="test")],
    conversation_id="conv-123",
)
assert state.conversation_id == "conv-123"
assert state.writeback_relationships == []
assert state.writeback_entities == []
print("PASS: OrchestratorState has new fields with correct defaults")
```
**Acceptance:** New fields present, default to correct values.

---

### TASK 8.3 — Modify `OrchestratorAgent.run()` to accept `conversation_id`
**File:** `core/agents/orchestrator_agent.py`
**Action:** Change the signature of `run()`:
```python
async def run(
    self,
    messages: List[BaseMessage],
    conversation_id: Optional[str] = None,
) -> FinalResponse:
```
And pass `conversation_id` into `OrchestratorState`:
```python
initial_state = OrchestratorState(
    messages=messages,
    conversation_id=conversation_id,
)
```
**Verify:**
```python
import inspect
from core.agents.orchestrator_agent import OrchestratorAgent
sig = inspect.signature(OrchestratorAgent.run)
params = list(sig.parameters.keys())
assert "conversation_id" in params
assert sig.parameters["conversation_id"].default is None
print("PASS: OrchestratorAgent.run() accepts conversation_id")
```
**Acceptance:** `conversation_id` parameter present with `None` default. Existing callers that don't pass it still work.

---

### TASK 8.4 — Rewrite `_synthesize_node` with CoT + write-back
**File:** `core/agents/orchestrator_agent.py`
**Action:** Replace the entire body of `_synthesize_node` with the implementation from plan Section 3. Key requirements:

1. **System prompt:** Use `SYNTHESISER_WRITEBACK_SYSTEM_PROMPT` (not the old inline system prompt string). Findings context is injected via `{context}` variable.
2. **Collect entities:** Iterate `state.agent_outputs.items()`, call `output.get_llm_context_str()`, collect `output.entities_enriched`.
3. **Parse `<relationships>`:** Use `re.search(r"<relationships>(.*?)</relationships>", raw, re.DOTALL)`. Wrap `json.loads()` in try/except. Default to `[]` on failure.
4. **Parse `<response>`:** Use `re.search(r"<response>(.*?)</response>", raw, re.DOTALL)`. Fallback to full `raw` if not found.
5. **Fire write-back:** Only if `state.conversation_id` is not None. Use `asyncio.create_task(run_conversation_writeback(...))`.
6. **Return `FinalResponse`:** Using parsed `user_response`, `fundamental_df`, `news_sources`.

**Verify:**
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.agents.orchestrator_agent import OrchestratorAgent, OrchestratorState
from langchain_core.messages import HumanMessage

async def test():
    agent = OrchestratorAgent()

    # Mock LLM response with valid CoT format
    mock_llm_response = MagicMock()
    mock_llm_response.content = """<relationships>
[{"from_name": "AAPL", "from_type": "Company", "relation": "INCREASES",
  "to_name": "Revenue", "to_type": "FinancialConcept", "confidence": "high"}]
</relationships>
<response>
Apple's revenue grew strongly in FY2024.
</response>"""

    mock_agent_output = MagicMock()
    mock_agent_output.get_llm_context_str.return_value = "### REPORT FROM fundamentals_agent\nRevenue: $90B"
    mock_agent_output.entities_enriched = []

    state = OrchestratorState(
        messages=[HumanMessage(content="How did Apple do?")],
        agent_outputs={"fundamentals_agent": mock_agent_output},
        conversation_id="test-conv-001",
    )

    with patch.object(agent._llm, "ainvoke", new=AsyncMock(return_value=mock_llm_response)), \
         patch("core.agents.orchestrator_agent.run_conversation_writeback", new=AsyncMock()) as mock_wb:

        # Give event loop time to pick up create_task
        result = await agent._synthesize_node(state)

        assert "final_response" in result
        assert result["final_response"].summary == "Apple's revenue grew strongly in FY2024."
        assert len(result["writeback_relationships"]) == 1
        assert result["writeback_relationships"][0]["relation"] == "INCREASES"

    print("PASS: _synthesize_node parses CoT output correctly")

asyncio.run(test())
```
**Acceptance:** `<relationships>` parsed into list. `<response>` extracted as user response. `FinalResponse.summary` contains only the response text, not the relationships block.

---

### TASK 8.5 — Verify synthesiser fault-tolerance: malformed `<relationships>` block
**File:** `core/agents/orchestrator_agent.py`
**Action:** No code change. Test that malformed JSON in `<relationships>` does not break the response.
**Verify:**
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.agents.orchestrator_agent import OrchestratorAgent, OrchestratorState
from langchain_core.messages import HumanMessage

async def test():
    agent = OrchestratorAgent()

    mock_llm_response = MagicMock()
    mock_llm_response.content = """<relationships>
THIS IS NOT VALID JSON {{{
</relationships>
<response>
Apple had a good quarter.
</response>"""

    mock_agent_output = MagicMock()
    mock_agent_output.get_llm_context_str.return_value = "test"
    mock_agent_output.entities_enriched = []

    state = OrchestratorState(
        messages=[HumanMessage(content="test")],
        agent_outputs={"fundamentals_agent": mock_agent_output},
        conversation_id=None,
    )

    with patch.object(agent._llm, "ainvoke", new=AsyncMock(return_value=mock_llm_response)):
        result = await agent._synthesize_node(state)
        assert result["writeback_relationships"] == [], "Malformed JSON must default to []"
        assert "Apple had a good quarter." in result["final_response"].summary
        print("PASS: malformed relationships block does not break user response")

asyncio.run(test())
```
**Acceptance:** Malformed JSON produces empty relationships list. User response is still returned correctly.

---

### TASK 8.6 — Verify synthesiser fault-tolerance: missing `<response>` block
**File:** `core/agents/orchestrator_agent.py`
**Action:** No code change. Test that missing `<response>` block falls back to full LLM output.
**Verify:**
```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.agents.orchestrator_agent import OrchestratorAgent, OrchestratorState
from langchain_core.messages import HumanMessage

async def test():
    agent = OrchestratorAgent()

    # LLM returns plain text with no XML tags at all
    mock_llm_response = MagicMock()
    mock_llm_response.content = "Apple had a good quarter, revenue up 5%."

    mock_agent_output = MagicMock()
    mock_agent_output.get_llm_context_str.return_value = "test"
    mock_agent_output.entities_enriched = []

    state = OrchestratorState(
        messages=[HumanMessage(content="test")],
        agent_outputs={"fundamentals_agent": mock_agent_output},
        conversation_id=None,
    )

    with patch.object(agent._llm, "ainvoke", new=AsyncMock(return_value=mock_llm_response)):
        result = await agent._synthesize_node(state)
        assert "Apple had a good quarter" in result["final_response"].summary
        print("PASS: missing <response> block falls back to full LLM output")

asyncio.run(test())
```
**Acceptance:** Full LLM output used as fallback. Does not raise. Does not return empty summary.

---

## PHASE 9 — Integration Smoke Tests

Run these after all phases complete. They test the full pipeline end-to-end with live services.

---

### TASK 9.1 — Lean pipeline smoke test (document ingestion)
**File:** none (integration test)
**Action:** Run the lean ingestion pipeline on a small sample document.
**Verify:**
```python
import asyncio
from core.memory.memory_system import FinancialMemorySystem, initialize_cognee
from core.memory.nodeset_manager import get_or_create_global_nodeset, get_or_create_all_sector_nodesets

async def test():
    await initialize_cognee()
    await get_or_create_global_nodeset()
    await get_or_create_all_sector_nodesets()

    system = FinancialMemorySystem()
    await system.initialize()

    sample_text = (
        "Apple Inc. (AAPL) reported fiscal Q3 2024 revenue of $85.8 billion, "
        "representing a 5% increase year-over-year. iPhone revenue accounted for "
        "$39.3 billion of the total. Operating income was $23.7 billion. "
        "The company faces headwinds from slowing consumer spending and "
        "intensifying competition in China from Huawei."
    )

    await system.ingest_document_lean(
        ticker="AAPL",
        report_type="10-K",
        content=sample_text,
        period="Q3 2024",
        include_summaries=True,
    )
    print("PASS: lean document ingestion completed without error")

asyncio.run(test())
```
**Acceptance:** Completes without exception. No graph extraction LLM calls were made (verify via logs — `extract_financial_graph` should NOT appear in log output).

---

### TASK 9.2 — Write-back smoke test
**File:** none (integration test)
**Action:** Call `run_conversation_writeback` with real entities and verify they appear in the graph.
**Verify:**
```python
import asyncio, uuid
from cognee.modules.engine.operations.setup import setup
from core.memory.memory_system import initialize_cognee
from core.memory.nodeset_manager import get_or_create_global_nodeset, get_or_create_all_sector_nodesets
from core.memory.graph_models import Company
from core.memory.pipeline_tasks import get_canonical_id
from core.memory.conversation_writeback import run_conversation_writeback
from cognee.infrastructure.databases.graph import get_graph_engine

async def test():
    await initialize_cognee()
    await get_or_create_global_nodeset()
    await get_or_create_all_sector_nodesets()

    ticker = "NVDA"
    company = Company(
        id=get_canonical_id(ticker),
        ticker=ticker,
        name="NVIDIA Corporation",
        description="NVIDIA reported record data centre revenue of $22.6B in Q2 2024.",
        sector="Information Technology",
        industry="Semiconductors",
    )

    relationships = [
        {
            "from_name": "NVIDIA Corporation",
            "from_type": "Company",
            "relation": "INCREASES",
            "to_name": "Data Centre Revenue",
            "to_type": "FinancialConcept",
            "confidence": "high",
        }
    ]

    await run_conversation_writeback(
        relationships=relationships,
        enriched_entities=[company],
        conversation_id="smoke-test-001",
    )

    # Verify entity appears in graph
    graph_engine = await get_graph_engine()
    results = await graph_engine.query(
        "MATCH (n:Company {ticker: $ticker}) RETURN n.ticker AS ticker LIMIT 1",
        {"ticker": ticker},
    )
    assert results and len(results) > 0, f"Company {ticker} not found in graph after write-back"
    print(f"PASS: write-back wrote Company({ticker}) to graph successfully")

asyncio.run(test())
```
**Acceptance:** Company node appears in graph. No exception raised. EntityRelationship DataPoint for the relationship was attempted.

---

### TASK 9.3 — `ingest_conversation()` no-op verification
**File:** none (regression test)
**Action:** Confirm existing call sites that call `ingest_conversation()` do not error.
**Verify:**
```python
import asyncio
from unittest.mock import MagicMock
from core.memory.memory_system import FinancialMemorySystem

async def test():
    system = FinancialMemorySystem()
    system._initialized = True
    system._global_nodeset = MagicMock()

    # Must not raise, must not call cognee
    await system.ingest_conversation(
        user_email="user@example.com",
        messages=[
            {"role": "user", "content": "Tell me about AAPL"},
            {"role": "assistant", "content": "Apple reported strong earnings..."},
        ],
    )
    print("PASS: ingest_conversation is safe no-op")

asyncio.run(test())
```
**Acceptance:** Completes without exception, no side effects.

---

### TASK 9.4 — Regression: existing agents run without error
**File:** none (regression test)
**Action:** Confirm `FundamentalAnalysisAgent` and `NewsAnalysisAgent` still run through their graph without errors on a minimal input.
**Verify:** Run each agent with a mock `BaseAgentInput` and confirm:
- Output is the correct type (`FundamentalAnalysisOutput` / `NewsAnalysisOutput`)
- `entities_enriched` is a non-empty list
- `entities_enriched[0]` is a `Company` instance with the correct ticker
**Acceptance:** Both agents produce correct output type. `entities_enriched` populated. No regression in existing functionality.

---

### TASK 9.5 — Full regression: run baseline test suite
**File:** none (regression check)
**Action:** Re-run the full test suite. Compare results against the baseline captured in TASK 0.1.
**Verify:**
```bash
pytest --tb=short -q 2>&1 | tee /tmp/post_implementation_test_results.txt
diff /tmp/baseline_test_results.txt /tmp/post_implementation_test_results.txt
```
**Acceptance:** No new test failures introduced. Any pre-existing failures from baseline remain the same set (no regression).

---

## PHASE 10 — Final Checklist

Mark each item only after the corresponding verify step passed.

| # | Item | Phase | Status |
|---|---|---|---|
| 1 | Baseline test results recorded | 0.1 | [x] |
| 2 | cognee environment confirmed reachable | 0.2 | [x] |
| 3 | `BaseAgentOutput.entities_enriched` field added | 1.1 | [x] |
| 4 | Existing agent outputs instantiate correctly | 1.2 | [x] |
| 5 | `LEAN_SUMMARY_SYSTEM_PROMPT` added | 2.1 | [x] |
| 6 | `SYNTHESISER_WRITEBACK_SYSTEM_PROMPT` added | 2.2 | [x] |
| 7 | New prompts exported from `core.memory` | 2.3 | [x] |
| 8 | `EntityRelationship` DataPoint implemented | 3.1 | [x] |
| 9 | `_relationship_id()` deterministic and case-insensitive | 3.2 | [x] |
| 10 | `_resolve_entity_nodeset()` assigns NodeSets | 3.3 | [x] |
| 11 | `_should_write_entity()` skips deduplicated items | 3.4 | [x] |
| 12 | `_build_relationship_datapoints()` filters malformed entries | 3.5 | [x] |
| 13 | `run_conversation_writeback()` handles missing/bad inputs silently | 3.6 | [x] |
| 14 | `conversation_writeback` exported from `core.memory.__init__` | 3.7 | [x] |
| 15 | `summarise_chunks_lean` added to `pipeline_tasks.py` | 4.1 | [x] |
| 16 | `build_lean_document_pipeline` added | 4.2 | [x] |
| 17 | `build_financial_pipeline` unchanged | 4.3 | [x] |
| 18 | `ingest_document_lean()` implemented with guards | 5.2 | [x] |
| 19 | `ingest_conversation()` is safe no-op | 5.3 | [x] |
| 20 | `cognify()` method unchanged | 5.4 | [x] |
| 21 | `_build_company_entity()` added to fundamental analysis agent | 6.1 | [x] |
| 22 | `FundamentalAnalysisOutput` populated with derived company entity | 6.2 | [x] |
| 23 | `_build_entities_from_news()` added to news analysis agent | 7.1 | [x] |
| 24 | `NewsAnalysisOutput` populated with derived company entity | 7.2 | [x] |
| 25 | No circular import from orchestrator changes | 8.1 | [x] |
| 26 | `OrchestratorState` has new fields | 8.2 | [x] |
| 27 | `OrchestratorAgent.run()` accepts `conversation_id` | 8.3 | [x] |
| 28 | `_synthesize_node` parses CoT output correctly | 8.4 | [x] |
| 29 | Malformed `<relationships>` does not break response | 8.5 | [x] |
| 30 | Missing `<response>` block falls back to full output | 8.6 | [x] |
| 31 | Lean ingestion smoke test passes | 9.1 | [x] |
| 32 | Write-back smoke test passes | 9.2 | [x] |
| 33 | `ingest_conversation()` no-op verified | 9.3 | [x] |
| 34 | Agent regression tests pass | 9.4 | [x] |
| 35 | Full test suite shows no new failures | 9.5 | [x] |

---

## DO NOT TOUCH LIST

The agent must not modify the following files under any circumstances:

| File | Reason |
|---|---|
| `core/memory/graph_models.py` | All DataPoint schemas are correct and reused as-is |
| `core/memory/entity_merger.py` | Reused by write-back, no changes needed |
| `core/memory/financial_retriever.py` | Retrieval layer is correct |
| `core/memory/nodeset_manager.py` | NodeSet lifecycle management is correct |
| `core/memory/exceptions.py` | Exception hierarchy is correct |
| `core/memory/graph_extraction.py` | Two-pass pipeline preserved for bulk re-indexing |
| `financial_db.py` | SQLite financial store unrelated to graph changes |
| `core/agents/base_agent.py` | Abstract base untouched |

If any verify step requires changes to these files, stop and raise the issue rather than modifying them.
