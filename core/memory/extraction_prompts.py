from typing import List
from core.memory.models import GLOBAL_ENTITY_TYPES, USER_ENTITY_TYPES

def build_global_extraction_prompt() -> str:
    """Builds the extraction prompt for the global shared knowledge graph."""
    entity_types_str = ", ".join(GLOBAL_ENTITY_TYPES)
    
    return f"""---Goal---
Given a text document and a list of entity types, identify all entities of those types from the text and all relationships among them.
Strictly ensure NO personal information (names of users, account details, personal financial goals, or personal PII) is extracted. Focus on domain-specific financial knowledge.

---Steps---
1. Identify all entities. For each identified entity, extract:
- entity_name: Name of the entity (e.g., 'S&P 500', 'Apple Inc', 'Inflation')
- entity_type: One of [{entity_types_str}]
- entity_description: Comprehensive description of the entity's attributes and role in the financial domain.

Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all relationships between pairs of entities:
- source_entity: name of the source
- target_entity: name of the target
- relationship_description: why they are related
- relationship_strength: 1-10
- relationship_keywords: summary keywords

Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Return output using ## as record delimiter and <|COMPLETE|> when done.
    """

def build_user_extraction_prompt(global_entities: List[str] = None) -> str:
    """
    Builds the extraction prompt for the user-specific private knowledge graph.
    If global_entities are provided, instructs the LLM to use GlobalRef stubs.
    """
    entity_types_str = ", ".join(USER_ENTITY_TYPES)
    
    global_ref_instruction = ""
    if global_entities:
        entities_list = ", ".join(global_entities[:100]) # Limit to avoid prompt bloat
        global_ref_instruction = f"""
IMPORTANT: The following entities ALREADY EXIST in the global shared knowledge graph: [{entities_list}].
If the text mentions these entities and links them to user-specific data, DO NOT create a new 'Organization' or 'FinancialInstrument' entity.
Instead, create a stub entity of type 'GlobalRef' with the EXACT SAME NAME as the global entity.
Link the user's entities (e.g., Portfolio, Goal) to these 'GlobalRef' stubs. T
These "GlobalRef" stubs should not be expanded upon, hence only the most relevant entity should be chosen as a stub, and 
"""

    return f"""---Goal---
Identify user-specific financial entities (Portfolio, Goal, Risk Profile, Transaction) and link them.
{global_ref_instruction}

---Steps---
1. Identify all entities. For each identified entity, extract:
- entity_name: Name (e.g., 'Retirement Goal', 'GlobalRef: S&P 500')
- entity_type: One of [{entity_types_str}]
- entity_description: User-specific context (e.g., 'User's 401k account', 'User wants to invest here')

Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify all relationships. Format as:
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Return output using ## as record delimiter and <|COMPLETE|> when done.
    """
