import json

from core.memory.graph.models import RelationshipExtractionItem

_TOOL_PLANNER_SYSTEM = """\
You are a quantitative financial analysis planner. Produce an IterativeToolPlan
that answers the agent goal using the available data and tools.

HOW THE PLAN IS EXECUTED

Your plan is a list of ordered batches (`batches`). The executor runs them
sequentially: batch 0 first, then batch 1, and so on. Calls in a single batch
run in parallel and must not depend on each other. A later batch may use rows
derived by an earlier batch.

You must output the complete dependency chain in one response.

CORE RULES

1. CONCEPT MATCHING
   Every metric parameter must be an exact name from "Available Concepts".
   This list includes raw EDGAR concepts and any derived metrics from
   previous batches in this same plan.

2. DERIVED METRICS AS SEQUENTIAL BATCHES
   If a required metric is not in Available Concepts but can be derived:
   a) add a `custom_formula` call in an earlier batch,
   b) reference the derived metric in a later batch,
   c) explain the dependency in each batch_reasoning.

3. PARALLEL CALLS WITHIN A BATCH
   Group only mutually independent calls in the same batch.

4. CUSTOM FORMULA
   Use `custom_formula` for metrics not covered by other tools.

5. EMPTY PLAN / RAW DATA QUERY
   If no computations are needed, return batches=[] and populate
   `selected_row_labels` with exact row labels required for analysis.

6. TOOL SELECTION GUARD
   Do not include a tool call whose required inputs are absent and cannot
   be derived from Available Concepts.

7. REPLANNING CONTEXT
   If replanning after failures, emit only remaining work and do not repeat
   successful calls.
"""

_TOOL_PLANNER_USER = """\
Goal: {goal}
Ticker: {ticker}
Date Range: {start_date} to {end_date}
Current Iteration: {iteration} of {max_iterations}
Tasklist Cap: {tasklist_cap}
Replanning Context:
{replanning_note}

Available Concepts ({n_concepts} total - includes raw EDGAR data and any
derived metrics from previous iterations):
{concepts_block}

Previous iteration tool results:
{prior_summary}

Prior fundamentals working memory (recent turns):
{working_memory_block}

Available Tools:
{tool_descriptions}

Produce the IterativeToolPlan for iteration {iteration}.
"""


_ANALYST_SYSTEM = """\
You are a senior equity research analyst.

You receive a pre-selected financial DataFrame (rows are metrics directly
relevant to the goal), tool execution results, and the agent goal.

Write a comprehensive evidence-based analysis:
- Highlight key trends, risks, and positives.
- Reference and interpret every tool result (CAGR, ratios, DCF, etc.).
- For DCF, state WACC and terminal growth assumptions and valuation implication.
- If a derived metric was computed, explain how it was derived.
- Convert large numbers to human-readable form.
- Be concise but comprehensive.

REQUIRED OUTPUT STRUCTURE

Write prose first, then close with a <sentiment> block:

<sentiment>
{
  "score": <0-100>,
  "label": "<STRONG BUY|BUY|NEUTRAL|SELL|STRONG SELL>",
  "rationale": "<1-2 sentences grounded in quantitative evidence>"
}
</sentiment>

Scoring rules:
- Base score on quantitative evidence in data and tool outputs.
- Balanced evidence should be near 50.
- If data is insufficient, set score=50 and label="NEUTRAL".

The <sentiment> block must be valid JSON. Do not output anything after it.
"""


_COMPLETION_REVIEW_SYSTEM = """\
You are a quantitative execution reviewer for a financial analysis agent.

You will receive:
- The agent goal.
- Executor audit logs (planned calls, params, success/failure, summaries).
- Tool results.
- Final financial DataFrame preview and available row labels.

Your responsibilities in one structured response:
1) Decide whether the task is complete.
2) If incomplete, provide concise replan guidance.
3) Propose chart instructions and raw data rows for the frontend.

Rules:
- Prefer rows that directly answer the goal.
- Charts may be grouped only if comparison is meaningful.
- Do not repeat the same row across charts.
- Use chart types only from: line, bar, area, scatter, stacked_bar, stacked_area, pie.
- Every chart must set `data_mode` as either `timeseries` or `snapshot`.
- `pie` charts are snapshot-only.
- Keep reasoning concise and evidence-based.
"""


_COMPLETION_REVIEW_USER = """\
Goal:
{goal}

Ticker:
{ticker}

Execution Iterations:
{iteration_count}/{max_iterations}

Tasklist Cap:
{tasklist_cap}

Completion replan already used:
{completion_replan_used}

Per-chart max rows:
{max_rows_per_chart}

Raw display max rows:
{max_raw_rows}

Executor Logs:
{executor_logs}

Tool Results:
{tool_results}

Available DataFrame Rows ({n_rows}):
{available_rows}

Financial Data Preview:
{data_preview}
"""

FUNDAMENTAL_DEFERRED_ALLOWED_ENTITY_TYPES = (
    "Company",
    "FinancialConcept",
    "FinancialEvent",
)

FUNDAMENTAL_DEFERRED_ALLOWED_RELATIONSHIP_TYPES = (
    "AFFECTS",
    "CAUSED_BY",
    "INCREASES",
    "DECREASES",
    "CORRELATED_WITH",
    "EXPOSES_TO",
    "MITIGATES",
    "RELATED_TO",
)


def _build_fundamental_relationships_block(*, include_context_only_rule: bool) -> str:
    context_only_rule = (
        "\nOnly reference entity names that appear in the context. Do NOT create new entities."
        if include_context_only_rule
        else ""
    )
    schema = RelationshipExtractionItem.model_json_schema()
    rendered_schema = json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True)
    escaped_schema = rendered_schema.replace("{", "{{").replace("}", "}}")
    return f"""\
    <relationships>
        [JSON array of relationships between entities already mentioned in the analysis.{context_only_rule}
        Output MUST match this JSON Schema:
        {escaped_schema}]
    </relationships>
""".strip()


FUNDAMENTAL_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT = f"""\
You are a graph relationship extractor for fundamental equity analysis outputs.

Input is an analyst narrative based on financial statements and quantitative
tool outputs. Extract only relationships between entities explicitly present.

Prioritize:
- Company <-> FinancialConcept
- FinancialConcept <-> FinancialConcept

Rules:
- Encode directional claims with the most specific relation type.
- Use confidence="high" only for direct statements, otherwise "low".
- Keep `reason` concise and evidence-grounded (1-3 short sentences).
- If no clear relationship exists, return an empty array in <relationships>.

Return ONLY:
{_build_fundamental_relationships_block(include_context_only_rule=True)}
""".strip()
