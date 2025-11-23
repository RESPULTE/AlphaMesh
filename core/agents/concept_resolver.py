import json
import re
from rapidfuzz import process, fuzz
import os
from typing import List, Optional


def build_concept_resolver(
    company_concepts: List[str], mapping_path: Optional[str] = None
):
    """
    Returns a function `resolve(user_term)` that maps input words to actual financial concepts.

    Parameters:
        company_concepts: List of financial concept names for the company
        mapping_path: Optional path to JSON mapping file. Defaults to 'concept_mappings.json' in same folder.
    """
    # ------------------------
    # Load mapping
    # ------------------------
    mapping = {}
    try:
        if mapping_path is None:
            mapping_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "concept_mappings.json"
            )
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    except Exception as e:
        print(f"[ConceptResolver] Error loading mapping file: {e}")

    official_concepts = set(company_concepts)
    common_concepts = list(mapping.keys())

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
        return mapping.get(term)

    def resolve_fuzzy_mapping(term: str, threshold=85):
        term = term.lower().strip()
        match_result = process.extractOne(
            term, common_concepts, scorer=fuzz.token_set_ratio
        )
        if match_result:
            best_key, score, _ = match_result
            if score >= threshold:
                return mapping[best_key]
        return None

    def resolve_column_keyword(term: str):
        for col in official_concepts:
            if term.lower() in normalize(col):
                return col
        return None

    def resolve_column_fuzzy(term: str):
        try:
            best, score, _ = process.extractOne(
                term, official_concepts, scorer=fuzz.WRatio
            )
            return best if score > 70 else None
        except Exception:
            return None

    # ------------------------
    # Main resolve function
    # ------------------------
    def resolve(user_term: str) -> Optional[str]:
        if not user_term:
            return None

        for strategy in [
            resolve_exact_mapping,
            resolve_fuzzy_mapping,
            resolve_column_keyword,
            resolve_column_fuzzy,
        ]:
            col = strategy(user_term)
            if col:
                return col

        return None

    return resolve
