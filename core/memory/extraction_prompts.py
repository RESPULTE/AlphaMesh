"""
Custom extraction prompts for the financial knowledge domain.

Provides two prompt builders:
  - build_global_extraction_prompt():  Domain-only extraction, explicit PII exclusion.
  - build_user_extraction_prompt():    Personal context extraction with GLOBAL_REF stubs.

Both use standard LightRAG delimiter format and are injected via
LightRAG's ``addon_params["extract_prompt"]``.
"""

from __future__ import annotations

from typing import List

from core.memory.models import (
    GLOBAL_ENTITY_HINTS,
    USER_ENTITY_HINTS,
    FINANCIAL_RELATIONSHIP_HINTS,
    get_entity_hints_text,
)


# ──────────────────────────────────────────────────────────────────────
# LightRAG standard delimiters
# ──────────────────────────────────────────────────────────────────────

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"


# ──────────────────────────────────────────────────────────────────────
# Few-shot examples
# ──────────────────────────────────────────────────────────────────────

GLOBAL_FEW_SHOT_EXAMPLE = """
----- Example Input -----
The Federal Reserve announced a 25 basis point rate cut on December 18, 2024,
bringing the target range to 4.25-4.50%. Markets reacted positively, with the
S&P 500 rising 1.2%. Goldman Sachs analysts noted that the decision aligned
with their expectations and maintained their forecast for two additional cuts
in 2025. The unemployment rate held steady at 4.2%, while CPI showed a 2.7%
year-over-year increase.

----- Example Output -----
("entity"{tuple_delimiter}"FEDERAL RESERVE"{tuple_delimiter}"Organization"{tuple_delimiter}"The Federal Reserve is the central bank of the United States, responsible for monetary policy including setting interest rates."){record_delimiter}
("entity"{tuple_delimiter}"S&P 500"{tuple_delimiter}"FinancialInstrument"{tuple_delimiter}"The S&P 500 is a stock market index tracking 500 large US companies, a key benchmark for US equity market performance."){record_delimiter}
("entity"{tuple_delimiter}"GOLDMAN SACHS"{tuple_delimiter}"Organization"{tuple_delimiter}"Goldman Sachs is a major global investment banking and financial services firm providing analysis and market forecasts."){record_delimiter}
("entity"{tuple_delimiter}"2024 FED RATE CUT"{tuple_delimiter}"MarketEvent"{tuple_delimiter}"The Federal Reserve cut interest rates by 25 basis points on December 18, 2024, bringing the target range to 4.25-4.50%."){record_delimiter}
("entity"{tuple_delimiter}"CPI"{tuple_delimiter}"EconomicIndicator"{tuple_delimiter}"The Consumer Price Index measures the average change in prices for a basket of consumer goods, showing 2.7% year-over-year increase."){record_delimiter}
("entity"{tuple_delimiter}"UNEMPLOYMENT RATE"{tuple_delimiter}"EconomicIndicator"{tuple_delimiter}"The unemployment rate measures the percentage of the labor force that is jobless, holding steady at 4.2%."){record_delimiter}
("relationship"{tuple_delimiter}"FEDERAL RESERVE"{tuple_delimiter}"2024 FED RATE CUT"{tuple_delimiter}"The Federal Reserve enacted the December 2024 rate cut as part of its monetary policy."{tuple_delimiter}"monetary policy, rate decision"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"2024 FED RATE CUT"{tuple_delimiter}"S&P 500"{tuple_delimiter}"The rate cut triggered a positive market reaction with the S&P 500 rising 1.2%."{tuple_delimiter}"market impact, rate sensitivity"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"GOLDMAN SACHS"{tuple_delimiter}"2024 FED RATE CUT"{tuple_delimiter}"Goldman Sachs analysts provided forecasts aligned with the rate cut decision."{tuple_delimiter}"analysis, forecast"{tuple_delimiter}6){record_delimiter}
("content_keywords"{tuple_delimiter}"federal reserve, rate cut, monetary policy, market reaction, economic indicators"){completion_delimiter}
""".strip()


USER_FEW_SHOT_EXAMPLE = """
----- Example Input -----
I currently have about $200k in my 401k, mostly in index funds tracking the
S&P 500. I want to retire by age 55, which gives me about 20 years. My risk
tolerance is moderate — I don't want to lose more than 15% in a bad year.
I'm thinking about adding some bond exposure, maybe through the Vanguard
Total Bond Market ETF. I recently sold my Tesla shares after the earnings report.

----- Example Output -----
("entity"{tuple_delimiter}"USER PROFILE"{tuple_delimiter}"UserProfile"{tuple_delimiter}"Investor approximately 35 years old with moderate risk tolerance, seeking retirement at age 55 with a 20-year investment horizon."){record_delimiter}
("entity"{tuple_delimiter}"RETIREMENT 401K PORTFOLIO"{tuple_delimiter}"Portfolio"{tuple_delimiter}"401k retirement portfolio valued at approximately $200k, primarily invested in S&P 500 index funds."){record_delimiter}
("entity"{tuple_delimiter}"RETIRE BY 55"{tuple_delimiter}"FinancialGoal"{tuple_delimiter}"Goal to retire by age 55, approximately 20 years away, requiring sustained portfolio growth."){record_delimiter}
("entity"{tuple_delimiter}"MODERATE RISK PROFILE"{tuple_delimiter}"RiskProfile"{tuple_delimiter}"Moderate risk tolerance with a maximum acceptable drawdown of 15% in a bad year over a 20-year horizon."){record_delimiter}
("entity"{tuple_delimiter}"SOLD TESLA SHARES"{tuple_delimiter}"Transaction"{tuple_delimiter}"Recently sold Tesla stock holdings following the company's earnings report."){record_delimiter}
("entity"{tuple_delimiter}"S&P 500"{tuple_delimiter}"GlobalRef"{tuple_delimiter}"GLOBAL REFERENCE: S&P 500 — an entity in the global knowledge graph."){record_delimiter}
("entity"{tuple_delimiter}"VANGUARD TOTAL BOND MARKET ETF"{tuple_delimiter}"GlobalRef"{tuple_delimiter}"GLOBAL REFERENCE: Vanguard Total Bond Market ETF — an entity in the global knowledge graph."){record_delimiter}
("entity"{tuple_delimiter}"TESLA"{tuple_delimiter}"GlobalRef"{tuple_delimiter}"GLOBAL REFERENCE: Tesla — an entity in the global knowledge graph."){record_delimiter}
("relationship"{tuple_delimiter}"USER PROFILE"{tuple_delimiter}"RETIREMENT 401K PORTFOLIO"{tuple_delimiter}"User owns and manages this 401k portfolio as their primary retirement vehicle."{tuple_delimiter}"ownership, retirement planning"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"USER PROFILE"{tuple_delimiter}"RETIRE BY 55"{tuple_delimiter}"User has set retirement by 55 as a primary financial goal."{tuple_delimiter}"financial goal, retirement"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"USER PROFILE"{tuple_delimiter}"MODERATE RISK PROFILE"{tuple_delimiter}"User has a moderate risk tolerance with 15% max drawdown threshold."{tuple_delimiter}"risk tolerance, investment style"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"RETIREMENT 401K PORTFOLIO"{tuple_delimiter}"S&P 500"{tuple_delimiter}"The 401k portfolio is primarily invested in S&P 500 index funds."{tuple_delimiter}"allocation, index tracking"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"USER PROFILE"{tuple_delimiter}"SOLD TESLA SHARES"{tuple_delimiter}"User executed a sell transaction on Tesla shares."{tuple_delimiter}"transaction, sell"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"SOLD TESLA SHARES"{tuple_delimiter}"TESLA"{tuple_delimiter}"The sell transaction involved Tesla stock."{tuple_delimiter}"transaction target, equity"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"retirement planning, portfolio allocation, risk tolerance, bond diversification"){completion_delimiter}
""".strip()


# ──────────────────────────────────────────────────────────────────────
# Prompt Builders
# ──────────────────────────────────────────────────────────────────────

def build_global_extraction_prompt() -> str:
    """
    Build the extraction prompt for the **global** namespace.

    Extracts only domain entities (financial instruments, organisations,
    market events, etc.). Explicitly forbids extraction of personal user
    information such as names, ages, account details, goals, or preferences.
    """
    entity_type_descriptions = get_entity_hints_text(GLOBAL_ENTITY_HINTS)

    return f"""---Goal---
Given a text document, identify all entities and relationships relevant to
FINANCIAL MARKETS AND ECONOMICS. Extract only domain-level knowledge that
is useful to ALL users — never extract private or personal information.

Use {{language}} as output language.

---CRITICAL PRIVACY RULE---
You MUST NOT extract any personally identifiable information (PII) including:
- User names, ages, income, account numbers, or demographics
- Personal investment holdings, portfolio values, or transaction details
- Individual financial goals, risk preferences, or investment strategies
- Any information that identifies or describes a specific private individual

Only extract PUBLIC domain knowledge: market data, financial concepts,
company information, economic indicators, and regulatory information.

---Entity Types---
{entity_type_descriptions}

---Steps---
1. Identify all entities. For each entity, extract:
   - entity_name: Name of the entity. If English, capitalize the name.
   - entity_type: One of [{{entity_types}}]
   - entity_description: Comprehensive description of the entity's attributes and activities

   Format: ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)

2. Identify all pairs of (source_entity, target_entity) that are *clearly related*.
   For each pair, extract:
   - source_entity: name of the source entity
   - target_entity: name of the target entity
   - relationship_description: why they are related
   - relationship_keywords: high-level keywords summarising the relationship
   - relationship_strength: numeric score (1-10) indicating strength

   Format: ("relationship"{{tuple_delimiter}}<source_entity>{{tuple_delimiter}}<target_entity>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_keywords>{{tuple_delimiter}}<relationship_strength>)

3. Identify high-level key words summarising the main themes.
   Format: ("content_keywords"{{tuple_delimiter}}<high_level_keywords>)

4. Return output in {{language}} as a single list. Use **{{record_delimiter}}** as the list delimiter.

5. When finished, output {{completion_delimiter}}

######################
---Examples---
######################
{GLOBAL_FEW_SHOT_EXAMPLE}

#############################
---Real Data---
######################
Entity_types: [{{entity_types}}]
Text:
{{input_text}}
######################
Output:
"""


def build_user_extraction_prompt(global_entities: list[str] | None = None) -> str:
    """
    Build the extraction prompt for a **user** namespace.

    Extracts user-specific entities (profile, portfolio, goals, risk, transactions).
    When global entities are provided, instructs the LLM to create GLOBAL_REF
    stubs using the exact global entity names so that post-processing can
    link them directly via cross-namespace Neo4j edges.

    Parameters
    ----------
    global_entities : list[str] | None
        Names of entities already present in the global knowledge graph.
        If provided, the LLM will create GLOBAL_REF stubs for any mentioned.
    """
    entity_type_descriptions = get_entity_hints_text(USER_ENTITY_HINTS)

    global_entity_section = ""
    if global_entities:
        entity_list = "\n".join(f"  - {name}" for name in global_entities)
        global_entity_section = f"""
---Known Global Entities---
The following entities ALREADY EXIST in the global knowledge graph.
If the text references any of these, create a GLOBAL_REF entity using the
EXACT same name. Do NOT duplicate their descriptions — just reference them.

{entity_list}

Any financial instrument, organisation, market event, or concept mentioned
in the text that matches or closely resembles a known global entity
MUST be typed as "GlobalRef" with the description:
  "GLOBAL REFERENCE: <entity_name> — an entity in the global knowledge graph."
"""

    return f"""---Goal---
Given a text document from a user conversation, identify all entities and
relationships that capture the USER'S PERSONAL financial context.

Use {{language}} as output language.

---Focus---
Extract the user's personal financial information including:
- Personal profile details (demographics, income, financial background)
- Portfolio holdings and allocations
- Financial goals and target dates
- Risk tolerance and investment preferences
- Transactions and trades

For any well-known financial instruments, companies, market events, or
economic concepts referenced in the text, create GLOBAL_REF entities
that will be linked to the shared global knowledge graph.

---Entity Types---
{entity_type_descriptions}

{global_entity_section}
---Steps---
1. Identify all entities. For each entity, extract:
   - entity_name: Name of the entity. If English, capitalize the name.
   - entity_type: One of [{{entity_types}}]
   - entity_description: Comprehensive description of the entity's attributes

   Format: ("entity"{{tuple_delimiter}}<entity_name>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)

2. Identify all pairs of (source_entity, target_entity) that are *clearly related*.
   For each pair, extract:
   - source_entity: name of the source entity
   - target_entity: name of the target entity
   - relationship_description: why they are related
   - relationship_keywords: high-level keywords summarising the relationship
   - relationship_strength: numeric score (1-10) indicating strength

   Format: ("relationship"{{tuple_delimiter}}<source_entity>{{tuple_delimiter}}<target_entity>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_keywords>{{tuple_delimiter}}<relationship_strength>)

3. Identify high-level key words summarising the main themes.
   Format: ("content_keywords"{{tuple_delimiter}}<high_level_keywords>)

4. Return output in {{language}} as a single list. Use **{{record_delimiter}}** as the list delimiter.

5. When finished, output {{completion_delimiter}}

######################
---Examples---
######################
{USER_FEW_SHOT_EXAMPLE}

#############################
---Real Data---
######################
Entity_types: [{{entity_types}}]
Text:
{{input_text}}
######################
Output:
"""
