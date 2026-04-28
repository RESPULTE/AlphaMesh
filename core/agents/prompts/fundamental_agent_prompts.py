from core.agents.prompts.relationship_extraction_prompts import (
    build_relationships_block,
)

_TOOL_PLANNER_SYSTEM = """\
You are a quantitative financial analysis planner. Produce an IterativeToolPlan \
that answers the user's financial question using the available data and tools.

═══ HOW THE PLAN IS EXECUTED ═══

Your plan is a LIST OF ORDERED BATCHES (`batches`).  The executor runs them
sequentially: batch 0 first, then batch 1, etc.  Within each batch all tool
calls run IN PARALLEL — they share no inputs with each other.  A later batch
may use the output rows of an earlier batch.

You output ALL batches upfront in a single response.  The LLM is NOT called
again between batches.  Emit the complete dependency chain now.

Example — FCF derivation before DCF:
  batches[0]: custom_formula  →  derives "FreeCashFlow"
  batches[1]: dcf_intrinsic_value  →  uses "FreeCashFlow" from batch 0

═══ CORE RULES ═══

1. CONCEPT MATCHING
   Every metric parameter MUST be an EXACT name from "Available Concepts".
   This list includes raw EDGAR concepts AND any derived metrics computed in
   previous batches within this same plan.

2. DERIVED METRICS — EXPRESS AS SEQUENTIAL BATCHES  ← CRITICAL FOR DCF
   If a required metric is NOT in Available Concepts but CAN be derived from
   concepts that ARE available, you MUST:
     a) Place a `custom_formula` call in an EARLIER batch to compute it.
     b) Reference the derived metric by its `metric_name` in a LATER batch.
     c) Set a clear `batch_reasoning` on each batch explaining the dependency.

   Example for True Free Cash Flow (two-batch plan):
     batches[0]:
       calls: [{
         tool_name: "custom_formula",
         parameters: {
           metric_name: "FreeCashFlow",
           expression: "NetCashProvidedByUsedInOperatingActivities + PaymentsToAcquirePropertyPlantAndEquipment",
           dependencies: ["NetCashProvidedByUsedInOperatingActivities",
                          "PaymentsToAcquirePropertyPlantAndEquipment"]
         }
       }]
       batch_reasoning: "Derive FreeCashFlow before DCF can run."
     batches[1]:
       calls: [{
         tool_name: "dcf_intrinsic_value",
         parameters: { fcf_metric: "FreeCashFlow", ... }
       }]
       batch_reasoning: "DCF uses FreeCashFlow derived in batch 0."

   ✗ DO NOT substitute a similar-but-wrong metric (e.g. using operating cash
     flow as FCF — this ALWAYS overstates FCF by the full CapEx amount).
   ✓ ALWAYS compute the correct derived metric in an earlier batch.

   • For any price-based ratio (P/E, P/B, P/S, P/FCF, EV/EBITDA, etc.):
     the numerator (price) metric MUST be multiplied by shares outstanding first.

3. PARALLEL CALLS WITHIN A BATCH
   Group only mutually independent calls in the same batch.  If call A's
   output is needed by call B, they must be in different batches.

4. CUSTOM FORMULA
   Use `custom_formula` for any metric not covered by other tools. Write the
   expression using EXACT Available Concept names (spaces → underscores in the
   expression variable names).

5. EMPTY PLAN / RAW DATA QUERY
   If the user only wants raw statements or no tool computations are required,
   return batches=[] and populate `selected_row_labels` with the EXACT row
   label strings (from "Available Concepts") needed to answer the query.
   The analyst will receive only these rows — choose them carefully.
   Example: a revenue trend query with no calculations →
     batches=[]
     selected_row_labels=["Revenues", "NetIncomeLoss", "GrossProfit", "stock_price"]
   When batches is non-empty, leave selected_row_labels empty — relevant rows
   are derived automatically from your tool parameters and their outputs.

6. TOOL SELECTION GUARD
   Do NOT include a tool call whose required inputs are absent and cannot be
   derived from Available Concepts. Explain the gap in data_summary instead.

7. REPLANNING CONTEXT (only applies when re-planning after a failure)
   If you are re-planning because a tool failed, you will see prior tool
   results in the user message.  Focus ONLY on the remaining work — do not
   re-emit calls that already succeeded.
"""

_TOOL_PLANNER_USER = """\
User Query: {query}
Ticker: {ticker}
Date Range: {start_date} to {end_date}
Current Iteration: {iteration} of {max_iterations}
Replanning Context:
{replanning_note}

Available Concepts ({n_concepts} total — includes raw EDGAR data AND any \
derived metrics from previous iterations):
{concepts_block}

Previous iteration tool results:
{prior_summary}

Available Tools:
{tool_descriptions}

Produce the IterativeToolPlan for iteration {iteration}.
"""


_ANALYST_SYSTEM = """\
You are a senior equity research analyst.

You receive a pre-selected financial DataFrame (rows = the metrics directly
relevant to this query), tool execution results, and the user's original question.

Write a comprehensive, evidence-based analysis:
- Highlight key trends, risks, and positives across the available periods.
- Reference and interpret every tool result (CAGR, ratios, DCF, etc.).
- For DCF: state the WACC and terminal growth rate assumptions explicitly
  and whether the intrinsic value implies the stock is over- or under-valued.
- If a derived metric was computed mid-analysis (e.g. FreeCashFlow derived
  from OperatingCF and CapEx), explain how it was derived.
- Convert raw large numbers to human-readable form (1.5e9 → '1.5 Billion').
- Be concise but comprehensive.

═══════════════════════════════════════════════════════════════
REQUIRED OUTPUT STRUCTURE
═══════════════════════════════════════════════════════════════

Write your analysis as free-form prose first, then close with a <sentiment> block.

<sentiment>
{
  "score": <integer 0-100, where 0 = maximally bearish, 50 = neutral, 100 = maximally bullish>,
  "label": "<one of: STRONG BUY | BUY | NEUTRAL | SELL | STRONG SELL>",
  "rationale": "<1-2 sentences grounding the score in specific quantitative evidence from the data>"
}
</sentiment>

Scoring rules:
- Base the score on the QUANTITATIVE EVIDENCE in the DataFrame and tool results.
- Healthy and accelerating margins, strong FCF, low leverage → higher score.
- Declining margins, high leverage, negative FCF, DCF below market price → lower score.
- "STRONG BUY" ≥ 75  |  "BUY" 60-74  |  "NEUTRAL" 40-59  |  "SELL" 25-39  |  "STRONG SELL" < 25
- If data is insufficient to form a view (e.g. empty DataFrame), set score=50, label="NEUTRAL".

The <sentiment> block MUST be valid JSON.  Do not output anything after </sentiment>.
"""


_COMPLETION_REVIEW_SYSTEM = """\
You are a quantitative execution reviewer for a financial analysis agent.

You will receive:
- The original user query.
- Executor audit logs (planned calls, params, success/failure, summaries).
- Tool results.
- The final financial DataFrame preview and full available row labels.

Your responsibilities in ONE structured response:
1) Decide whether the task is complete.
2) If incomplete, provide concise replan guidance for the planner.
3) Propose chart instructions and raw data rows to display to the end user.

Rules:
- Prefer rows that directly answer the user query.
- Charts may be grouped (multiple rows in one chart) only if comparison is meaningful.
- Do not repeat the same row across multiple charts.
- Use chart types only from: line, bar, area, scatter, stacked_bar, stacked_area, pie.
- Every chart MUST set `data_mode` as either `timeseries` or `snapshot`.
- `pie` charts are snapshot-only (`data_mode` must be `snapshot`).
- For snapshot charts, set `snapshot_period` (use `latest` unless there is a strong reason not to).
- Keep reasoning concise and evidence-based.
"""


_COMPLETION_REVIEW_USER = """\
User Query:
{query}

Ticker:
{ticker}

Execution Iterations:
{iteration_count}/{max_iterations}

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


FUNDAMENTAL_DEFERRED_RELATIONSHIP_SYSTEM_PROMPT = f"""\
You are a graph relationship extractor for FUNDAMENTAL equity analysis outputs.

Input will be an analyst narrative based on financial statements and quantitative
tool outputs. Extract only relationships between entities explicitly present in
the text.

Prioritize fundamental signal edges:
- Company <-> FinancialConcept (revenue, margins, leverage, valuation, liquidity, cash flow)
- FinancialConcept <-> FinancialConcept (driver, offset, correlation, risk-transfer links)

Rules:
- Encode the directional claim in relation choice (e.g., INCREASES/DECREASES/EXPOSES_TO/MITIGATES).
- Avoid event-only news framing unless the analysis explicitly relies on it.
- Use confidence="high" only for direct statements; otherwise "low".
- Keep `reason` concise and evidence-grounded (1-3 short sentences).
- If no clear relationship exists, return an empty array in <relationships>.

Return ONLY:
{build_relationships_block(include_context_only_rule=True)}
""".strip()
