import json
import re
from rapidfuzz import process, fuzz


class ConceptResolver:
    def __init__(self, mapping_path="./concept_mapping.json"):
        try:
            with open(mapping_path, "r") as f:
                self.mapping = json.load(f)
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            self.mapping = {}

        self.official_concepts = set(self.mapping.values())
        self.common_concepts = list(self.mapping.keys())

    def normalize(self, s: str) -> str:
        """
        Aggressive normalization to strip special chars and lowercase.
        """
        if not isinstance(s, str):
            return ""
        s = s.lower()
        # Remove prefixes specific to XBRL to help with direct column matching later
        s = s.replace("us-gaap_", "").replace("msft_", "").replace("_", " ")
        s = re.sub(r"[^a-z0-9 ]+", "", s)
        return s.strip()

    # ---------------------------------------------------------
    # STRATEGY 1: Exact Mapping
    # ---------------------------------------------------------
    def resolve_exact_mapping(self, term: str):
        """
        Checks if the user term exists exactly in the JSON keys.
        """
        term = term.lower().strip()

        # Check if term is a key in the JSON
        if term in self.mapping:
            xbrl_tag = self.mapping[term]
            # verify if this tag exists in the actual dataframe
            if xbrl_tag in self.official_concepts:
                return xbrl_tag
        return None

    # ---------------------------------------------------------
    # STRATEGY 2: Fuzzy Mapping on JSON Keys (The Improvement)
    # ---------------------------------------------------------
    def resolve_fuzzy_mapping(self, term: str, threshold=85):
        """
        Matches user input against the HUMAN READABLE keys in the JSON.
        Example: User types "acc payables" -> Matches JSON key "accounts payable" -> Returns "us-gaap_AccountsPayableCurrent"
        """
        term = term.lower().strip()

        # Extract the best match from the JSON keys (not the columns)
        # limit=1 returns a list of tuples [(match, score, index)]
        match_result = process.extractOne(
            term, self.common_concepts, scorer=fuzz.WRatio
        )

        if match_result:
            best_match_key, score, _ = match_result

            if score >= threshold:
                xbrl_tag = self.mapping[best_match_key]
                # We found a concept match, now does the column exist in the DF?
                if xbrl_tag in self.official_concepts:
                    return xbrl_tag

                # Edge Case: The concept matches, but the specific XBRL tag isn't in this specific company's file.
                # You might want to log this or try a fallback.

        return None

    # ---------------------------------------------------------
    # STRATEGY 3: Keyword/Partial Match on DataFrame Columns
    # ---------------------------------------------------------
    def resolve_column_keyword(self, term: str):
        """
        Fallback: If not in JSON, look at the actual dataframe columns
        and see if the words exist there.
        """
        keywords = term.lower().split()
        for col in self.official_concepts:
            norm = self.normalize(col)
            # If all user keywords appear in the normalized column name
            if all(kw in norm for kw in keywords):
                return col
        return None

    # ---------------------------------------------------------
    # STRATEGY 4: Fuzzy Match on DataFrame Columns
    # ---------------------------------------------------------
    def resolve_column_fuzzy(self, term: str):
        """
        Final Fallback: Fuzzy match against the ugly XBRL tags directly.
        """
        best, score, _ = process.extractOne(
            term, self.official_concepts, scorer=fuzz.WRatio
        )
        return best if score > 70 else None

    # ---------------------------------------------------------
    # Main Entry Point
    # ---------------------------------------------------------
    def resolve(self, user_term: str):
        if not user_term:
            return None

        # 1. Check Exact JSON Key
        col = self.resolve_exact_mapping(user_term)
        if col:
            return col

        # 2. Check Fuzzy JSON Key (Finds "closely related" words in your dictionary)
        col = self.resolve_fuzzy_mapping(user_term)
        if col:
            return col

        # 3. Check Keywords in actual DF Columns
        col = self.resolve_column_keyword(user_term)
        if col:
            return col

        # 4. Last Resort: Fuzzy match actual DF Columns
        col = self.resolve_column_fuzzy(user_term)
        return col
