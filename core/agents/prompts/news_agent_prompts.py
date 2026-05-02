from __future__ import annotations

import json
from typing import Sequence

NEWS_PLANNER_SYSTEM_PROMPT = """\
You are a retrieval planner for a financial news analysis agent.

You receive:
- Goal describing missing information that must be fetched.
- Current iteration index and compact tool-call history from this turn.

Return one PlannerDecision object:
- action: "newsapi" or "web_search"
- queries: domain-specific queries across company/sector/market/knowledge

Rules:
- Choose the action most likely to close the information gap efficiently.
- Provide 1-4 queries total, at most one per domain.
- Queries must be specific and self-contained.
- Never return empty queries.

Return ONLY a valid PlannerDecision object.
"""

NEWS_ANALYSIS_AGENT_SYSTEM_PROMPT = """
You are the sufficiency and source-selection stage of a financial news analysis workflow.

You are given:
1. Current analysis goal
2. Article-grouped evidence
3. Optional company and memory context
4. Iteration metadata, including forced_final_pass

Your output is structured metadata only. Do not generate narrative analysis text.

If forced_final_pass=false, return:
- is_context_sufficient: boolean
- missing_information_goal: string
- persist_chunk_ids: list of chunk ids
- source_chunk_ids: list of chunk ids that should support the final narrative

If forced_final_pass=true, return:
- is_context_sufficient: true
- source_chunk_ids: list of chunk ids that should support the final narrative

Decision policy:
- Be practical, not perfectionistic.
- Mark sufficient when available evidence can support a useful, caveated investor answer.
- Mark insufficient only when missing evidence materially blocks a useful answer.

When insufficient:
- Set source_chunk_ids to [].
- Keep persist_chunk_ids focused on high-signal chunks worth carrying forward.
- Write missing_information_goal as a specific retrieval objective for the next planner pass.

When sufficient:
- Select source_chunk_ids that directly support the eventual narrative.
- Keep IDs to those present in the provided context.
- Prefer high-relevance and directly related evidence over tangential chunks.
- Set persist_chunk_ids equal to source_chunk_ids unless there is a clear reason to retain extra context.

Reasoning guardrails:
- Never fabricate facts.
- Do not emit markdown or prose outside the structured response.
""".strip()

NEWS_DEFERRED_ALLOWED_ENTITY_TYPES = (
    "Company",
    "FinancialEvent",
    "FinancialConcept",
    "Sector",
    "Industry",
    "Market",
)

NEWS_DEFERRED_ALLOWED_RELATIONSHIP_TYPES = (
    "AFFECTS",
    "CAUSED_BY",
    "INCREASES",
    "DECREASES",
    "CORRELATED_WITH",
    "EXPOSES_TO",
    "MITIGATES",
    "COMPETES_WITH",
    "ACQUIRED_BY",
    "RELATED_TO",
    "BELONGS_TO",
)

NEWS_DEFERRED_CHUNK_ENTITY_SYSTEM_PROMPT = """\
You are an entity extractor for financial news chunks.

Extract only entities that are explicitly supported by each chunk's text.
Prioritize entities useful for relationship grounding in downstream graph extraction:
- Company
- FinancialEvent
- FinancialConcept

Rules:
- Keep entity names canonical and concise.
- Do not infer entities absent from the chunk text.
- For each entity, provide a short factual description grounded in the chunk.
- If a chunk has no extractable entities, return an empty entity list for that chunk.
""".strip()


def _build_relationship_schema_for_news_prompt(
    *,
    allowed_entity_types: Sequence[str],
    allowed_relationship_types: Sequence[str],
) -> str:
    entity_types = [
        str(item).strip() for item in allowed_entity_types if str(item).strip()
    ]
    relationship_types = [
        str(item).strip() for item in allowed_relationship_types if str(item).strip()
    ]
    schema: dict = {
        "type": "object",
        "title": "Relationship",
        "description": "Schema for one relationship item expected inside <relationships>.",
        "properties": {
            "from_name": {"title": "From Name", "type": "string"},
            "from_type": {
                "title": "From Type",
                "type": "string",
                "enum": entity_types,
            },
            "relationship_type": {
                "title": "Relationship Type",
                "type": "string",
                "enum": relationship_types,
            },
            "to_name": {"title": "To Name", "type": "string"},
            "to_type": {
                "title": "To Type",
                "type": "string",
                "enum": entity_types,
            },
            "confidence": {
                "title": "Confidence",
                "type": "string",
                "enum": ["high", "low"],
                "description": '"high" for explicit evidence, "low" for inferred.',
            },
            "reason": {
                "title": "Reason",
                "type": "string",
                "description": "1-2 short sentences explaining the relationship.",
            },
        },
        "required": [
            "from_name",
            "from_type",
            "relationship_type",
            "to_name",
            "to_type",
            "confidence",
            "reason",
        ],
        "additionalProperties": False,
    }
    rendered = json.dumps(schema, indent=2, ensure_ascii=True, sort_keys=True)
    return rendered.replace("{", "{{").replace("}", "}}")


def _build_news_relationships_block(
    *,
    include_context_only_rule: bool,
    allowed_entity_types: Sequence[str],
    allowed_relationship_types: Sequence[str],
) -> str:
    context_only_rule = (
        "\nOnly reference entity names that appear in the context. Do NOT create new entities."
        if include_context_only_rule
        else ""
    )
    return f"""\
    <relationships>
        [JSON array of relationships between entities already mentioned in the analysis.{context_only_rule}
        Output MUST match this JSON Schema:
        {_build_relationship_schema_for_news_prompt(
            allowed_entity_types=allowed_entity_types,
            allowed_relationship_types=allowed_relationship_types,
        )}]
    </relationships>
""".strip()


def build_news_deferred_relationship_system_prompt(
    *,
    allowed_entity_types: Sequence[str],
    allowed_relationship_types: Sequence[str],
) -> str:
    return f"""\
You are a graph relationship extractor for financial NEWS analysis outputs.

Input will be a completed news-analysis narrative. Extract only relationships
between entities that are explicitly present in that narrative.

Prioritize high-signal news structure:
- Company <-> FinancialEvent (earnings, guidance changes, downgrades/upgrades, M&A)
- Company/FinancialEvent <-> FinancialConcept (revenue growth, margins, demand, liquidity, risk)
- Company/FinancialEvent <-> Sector/Industry/Market when explicitly stated

Rules:
- Prefer the most specific edge type available over RELATED_TO.
- Use confidence="high" only for directly stated links; otherwise use "low".
- Keep `reason` factual and concise (1-3 short sentences).
- If no clear relationship exists, return an empty array in <relationships>.

Return ONLY:
{_build_news_relationships_block(
    include_context_only_rule=True,
    allowed_entity_types=allowed_entity_types,
    allowed_relationship_types=allowed_relationship_types,
)}
""".strip()


def build_news_deferred_chunk_entity_system_prompt() -> str:
    return NEWS_DEFERRED_CHUNK_ENTITY_SYSTEM_PROMPT


NEWS_ANALYSIS_USER_PROMPT = """\
Goal: {goal}

Iteration: {iteration}/{max_iterations}
Forced final pass: {forced_final_pass}

{entities_section}Article-grouped evidence (deduplicated across working memory + current retrieval):
{article_context}

Return structured output only.
""".strip()
