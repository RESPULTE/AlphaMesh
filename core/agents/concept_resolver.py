import json
import re
import os
from typing import List, Optional

FINANCIAL_CONCEPT_MAPPING = {}
try:
    mapping_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "concept_mappings.json"
    )
    with open(mapping_path, "r") as f:
        FINANCIAL_CONCEPT_MAPPING = json.load(f)
except Exception as e:
    print(f"[ConceptResolver] Error loading mapping file: {e}")
COMMON_FINANCIAL_CONCEPTS = list(FINANCIAL_CONCEPT_MAPPING.keys())
OFFICIAL_FINANCIAL_CONCEPTS = list(FINANCIAL_CONCEPT_MAPPING.values())


def build_concept_resolver(company_concepts: List[str]):
    """
    Returns a function `resolve(user_term)` that maps input words to actual financial concepts.

    Parameters:
        company_concepts: List of financial concept names for the company
        mapping_path: Optional path to JSON mapping file. Defaults to 'concept_mappings.json' in same folder.
    """
    # ------------------------
    # Load mapping
    # ------------------------

    official_concepts = set(company_concepts)

    # ------------------------
    # Normalization helper
    # ------------------------
    def normalize(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.lower()
        s = s.replace("us-gaap_", "").replace("msft_", "").replace("_", " ")
        s = re.sub(r"[^a-z0-9 ]+", "", s)
        return s.strip()

    # ------------------------
    # Resolve strategies
    # ------------------------
    def resolve_exact_mapping(term: str):
        term = term.lower().strip()
        return FINANCIAL_CONCEPT_MAPPING.get(term)

    def resolve_column_keyword(term: str):
        for col in official_concepts:
            if term.lower() in normalize(col):
                return col
        return None

    # ------------------------
    # Main resolve function
    # ------------------------
    def resolve(user_term: str) -> Optional[str]:
        if not user_term:
            return None

        for strategy in [resolve_column_keyword]:
            col = strategy(user_term)
            if col:
                return col

        print(f"[ConceptResolver] Could not resolve concept for term '{user_term}'")

        return None

    return resolve
