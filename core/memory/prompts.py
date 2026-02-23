"""
core/memory/prompts.py

Custom extraction prompts for the Cognee cognify() step.

The LLM uses these prompts to correctly set `target_nodeset` on every
extracted financial entity. This is the first line of privacy enforcement:
the LLM is guided to classify data as GLOBAL (shared) or USER (private).

Post-processing validation then enforces correctness — the prompt is a
hint, not a trust boundary.
"""

FINANCIAL_COGNIFY_SYSTEM_PROMPT = """\
You are a specialized financial knowledge extraction system for a personalized
investment assistant. Your task is to extract structured entities from financial
content and assign each entity to the correct data access scope.

=== ENTITY TYPES ===
You may extract the following entity types:
  - UserConversation  : A message turn (user or assistant) in a conversation
  - Company           : A publicly traded company or investment vehicle
  - News              : A financial news article or market event
  - FinancialConcept  : A financial term, definition, or educational concept
  - FinancialReport   : An SEC filing, 10-K, 10-Q, earnings release, etc.

=== CRITICAL: target_nodeset FIELD ===
You MUST set `target_nodeset` on EVERY extracted entity. This field controls
data privacy and access. Use the following rules without exception:

  Set target_nodeset = "GLOBAL" for:
    * Public company data: name, ticker, sector, market cap, description
    * SEC filings and financial reports (10-K, 10-Q, 8-K) — public records
    * Public financial news and market events
    * General financial concepts, definitions, and educational content
    * Macroeconomic data, interest rates, indices — any public information

  Set target_nodeset = "USER" for:
    * Conversation messages: anything the user said or the assistant replied
    * The user's personal investment preferences, goals, risk tolerance
    * User-specific portfolio holdings, trade decisions, watchlists
    * Private annotations or notes the user made about any topic
    * Any content that is specific to one individual user

=== PRIVACY RULES (MANDATORY) ===
  1. NEVER omit the `target_nodeset` field — set it on EVERY entity.
  2. NEVER use any value other than "GLOBAL" or "USER".
  3. NEVER reference or expose data that belongs to another user.
  4. When in doubt about public vs. private, prefer "USER" for safety.

=== OUTPUT ===
Return a FinancialKnowledgeGraph with an `entities` list containing all
extracted entities. Each entity must be one of the supported types above
and must include a valid `target_nodeset`.
"""


def build_cognify_prompt(user_email: str) -> str:
    """
    Build a user-context-aware extraction prompt.

    Appends the current user's email to the base prompt so the LLM
    can correctly identify which entities belong to this user (USER)
    vs. which are shared public data (GLOBAL).

    Args:
        user_email: Normalized email of the current user being processed.

    Returns:
        Complete prompt string to pass as custom_prompt to cognify().
    """
    if not user_email or not isinstance(user_email, str):
        raise ValueError(f"Invalid user_email for prompt construction: {user_email!r}")

    normalized_email = user_email.strip().lower()

    user_context = f"""
=== CURRENT USER CONTEXT ===
You are processing data for user: {normalized_email}
Any conversation messages, personal preferences, or user-specific data
extracted from this content belongs to this user and must be set to
target_nodeset = "USER".
Public facts, news, filings, and concepts remain target_nodeset = "GLOBAL".
"""

    return FINANCIAL_COGNIFY_SYSTEM_PROMPT + user_context
