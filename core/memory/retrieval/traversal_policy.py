from __future__ import annotations

from typing import Dict, List


from core.memory.graph.models import _ENTITY_TYPE_WEIGHTS, _RELATIONSHIP_WEIGHTS
from core.memory.retrieval.models import NeighborCandidate, RetrieverState


class TraversalPolicy:
    """
    Encapsulates all traversal decisions — scoring, frontier selection, stopping.

    Extracting this from DualStoreRetriever means:
    - Scoring logic is independently unit-testable without a LangGraph fixture.
    - Scoring can be tuned or swapped without touching orchestration code.
    - The retriever's node methods stay thin and readable.

    Neighbor scoring uses a three-factor composite model:

        score = structural_weight × hop_decay × (1 + query_relevance_bonus)

    structural_weight
        relationship_type weight × entity_type weight (from graph/models.py).
        Captures the semantic importance of the connection type and the target
        entity category.

    hop_decay
        1 / (1 + hop_depth).  Penalises distant graph expansions so the
        traversal stays grounded in the original query as hops accumulate.
        Without decay, all hops are treated equally and deep expansions can
        dominate retrieval with tangentially related content.

    query_relevance_bonus
        Jaccard token overlap between the query string and the candidate entity
        name.  Cheap (zero I/O, no embedding call), keeps query signal alive
        through graph hops.  Bounded by the 0.2 multiplier so it enhances but
        never overrides the structural weight.
    """

    def __init__(
        self,
        max_parallel_nodes: int,
        max_neighbor_candidates: int,
        max_iterations: int,
    ) -> None:
        self._max_parallel_nodes = max_parallel_nodes
        self._max_neighbor_candidates = max_neighbor_candidates
        self._max_iterations = max_iterations

    # ── Public API ────────────────────────────────────────────────────────────

    def cap_per_source(
        self, candidates: List[NeighborCandidate]
    ) -> List[NeighborCandidate]:
        """
        Keep at most `max_neighbor_candidates` results per source entity.
        Preserves the original order so callers can rely on stable iteration.
        """
        counts: Dict[str, int] = {}
        result: List[NeighborCandidate] = []
        for c in candidates:
            if not c.source_entity_id:
                continue
            counts.setdefault(c.source_entity_id, 0)
            if counts[c.source_entity_id] < self._max_neighbor_candidates:
                counts[c.source_entity_id] += 1
                result.append(c)
        return result

    def score_neighbor(
        self,
        candidate: NeighborCandidate,
        query: str,
        hop_depth: int,
    ) -> float:
        """Compute the composite traversal score for a single neighbor candidate."""
        structural = _RELATIONSHIP_WEIGHTS.get(
            candidate.relationship_type, 0.1
        ) * _ENTITY_TYPE_WEIGHTS.get(candidate.neighbor_type, 0.5)
        decay = 1.0 / (1 + hop_depth)
        relevance_bonus = self._lexical_overlap(query, candidate.neighbor_name)
        return structural * decay * (1.0 + 0.2 * relevance_bonus)

    def select_frontier(
        self,
        candidates: List[NeighborCandidate],
        query: str,
        hop_depth: int,
    ) -> List[str]:
        """
        Score all candidates, sort descending, return the top-N unique entity IDs
        for the next frontier.  Deduplication of entity IDs is applied post-sort
        so the highest-scored occurrence of a duplicated entity is kept.
        """
        scored = sorted(
            candidates,
            key=lambda c: self.score_neighbor(c, query, hop_depth),
            reverse=True,
        )
        seen: set = set()
        selected: List[str] = []
        for c in scored:
            if not c.neighbor_entity_id or c.neighbor_entity_id in seen:
                continue
            seen.add(c.neighbor_entity_id)
            selected.append(c.neighbor_entity_id)
            if len(selected) >= self._max_parallel_nodes:
                break
        return selected

    def should_continue(self, state: RetrieverState) -> bool:
        """Return True if traversal should proceed to another expansion cycle."""
        return (
            bool(state["should_continue"]) and state["iteration"] < self._max_iterations
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _lexical_overlap(query: str, name: str) -> float:
        """
        Jaccard token overlap between query and entity name, in [0.0, 1.0].

        Uses whitespace tokenisation — fast, no dependencies, zero I/O.
        Case-insensitive.  Returns 0.0 if either string is empty.
        """
        if not query or not name:
            return 0.0
        q_tokens = set(query.lower().split())
        n_tokens = set(name.lower().split())
        if not q_tokens or not n_tokens:
            return 0.0
        union = q_tokens | n_tokens
        return len(q_tokens & n_tokens) / len(union)
