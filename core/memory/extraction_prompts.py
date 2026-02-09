# core/memory/extraction_prompts.py
"""
Custom extraction prompts for the Financial Knowledge Memory Module.

These prompts guide the LLM to extract appropriate entities and relationships
for each namespace (global vs user).
"""

# ==============================================================================
# GLOBAL NAMESPACE EXTRACTION PROMPT
# Used when extracting entities to the shared global knowledge base.
# ==============================================================================

GLOBAL_EXTRACTION_PROMPT = """
Focus on extracting objective financial entities and their relationships:

ENTITIES TO EXTRACT:
- Companies: name, ticker symbol, sector, exchange, founding info
- ETFs: name, ticker, expense ratio, holdings count
- Financial Events: earnings reports, Fed meetings, IPOs, M&A
- Financial Concepts: terms, strategies, indicators, ratios
- Financial News: market updates, regulatory changes, economic indicators

DO NOT EXTRACT:
- User preferences, opinions, or personal reactions
- Portfolio positions or holdings
- Investment goals or risk preferences
- Any data specific to an individual investor
- Personal financial details (income, expenditure, etc.)

Extract only factual, publicly-available financial information that would be
valuable to any user of the system.

DO NOT INCLUDE ANY USER-SPECIFIC INFORMATION IN ANY PART OF THE OUTPUT.
"""

# ==============================================================================
# USER NAMESPACE EXTRACTION PROMPT TEMPLATE
# Used when extracting personalized data to user-specific namespace.
# ==============================================================================

USER_EXTRACTION_PROMPT_TEMPLATE = """
Focus on extracting personalized user data and their relationships to entities:

ENTITIES TO EXTRACT:
- User: the user themselves with profile info (age, occupation, income, etc.)
- Portfolio: named collections of investments
- Watchlist: securities being tracked
- Position: specific holdings in securities
- UserPreference: risk tolerance, investment horizons
- UserGoal: financial goals and targets

RELATIONSHIPS TO EXTRACT (as edges linking to global entities):
- User interests, preferences toward companies/ETFs/sectors
- Holdings and positions in specific securities
- Watchlist items
- Learning progress on financial concepts
- Avoidances or negative sentiments toward entities

IMPORTANT - EXISTING GLOBAL ENTITIES:
The following entities already exist in the global namespace: {global_entity_names}

Create duplicate entities for these which are to serve as reference entities ONLY, 
with minimal detail as it will be removed later.

Example: If user "holds Apple stock", create a User node + edge "HOLDS" pointing
to a new "Apple" entity.
"""

# ==============================================================================
# HELPER FUNCTION
# ==============================================================================


def get_user_extraction_prompt(global_entity_names: list[str]) -> str:
    """Generate user extraction prompt with global entity names populated."""
    names_str = ", ".join(global_entity_names) if global_entity_names else "None"
    return USER_EXTRACTION_PROMPT_TEMPLATE.format(global_entity_names=names_str)
