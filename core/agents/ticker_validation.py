"""
core/agents/ticker_validation.py

Validates up to 3 ticker symbols via yfinance and enriches new ones with
canonical metadata (long name, sector, industry, business description).

Design notes
────────────
- Uses yf.Tickers for batched I/O — single HTTP session for all tickers.
- All blocking yfinance calls execute in asyncio.to_thread().
- Validation path: fast_info.quote_type determines equity / non-equity / unknown.
- Lookup fallback: triggered only when fast_info raises or returns no data,
  keeping latency low for the common case.
- Enrichment (.info): fetched for ALL valid equities so that company_context
  can be built even for already-known companies.
  TODO: For lower latency on known companies, replace the yfinance .info call
        with a Neo4j get_company_entity_by_ticker() query that returns the
        cached (name, description, sector, industry) from the graph store.
- Taxonomy upsert: Industry and Company nodes are written to Neo4j + Chroma
  as a background asyncio task so the critical path is never blocked.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yfinance as yf

from core.logger import get_logger
from core.memory.graph.models import EntityNode
from core.memory.graph.utils import canonical_entity_id

logger = get_logger(__name__)

MAX_TICKERS = 3
_EQUITY_QUOTE_TYPES = frozenset({"EQUITY", "STOCK"})


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TickerInfo:
    """Enrichment result for a single validated ticker."""

    ticker: str
    is_valid: bool
    is_equity: bool
    quote_type: Optional[str]
    long_name: str = ""
    description: str = ""  # yfinance longBusinessSummary
    sector: str = ""  # yfinance canonical sector name
    industry: str = ""  # yfinance industry string
    is_new: bool = False  # True when no Company entity exists in Neo4j yet
    needs_confirmation: bool = False  # True when non-equity or lookup suggestions found
    suggestions: List[str] = field(default_factory=list)

    def to_context_block(self) -> str:
        """
        Format enrichment data as a structured block for injection into
        agent prompts. Returns an empty string when there is no useful data.
        """
        if not self.is_valid or not self.long_name:
            return ""
        lines = [f"COMPANY BACKGROUND: {self.long_name} ({self.ticker})"]
        if self.sector:
            lines.append(f"Sector: {self.sector}")
        if self.industry:
            lines.append(f"Industry: {self.industry}")
        if self.description:
            lines.append(f"Description: {self.description}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────


class TickerValidator:
    """
    Validates and enriches up to MAX_TICKERS tickers using yfinance.

    For new companies, upserts Company and Industry entity nodes into both
    Neo4j and ChromaDB, and creates BELONGS_TO taxonomy edges
    (Company → Industry → Sector) as a non-blocking background task.
    """

    def __init__(self, neo4j_adapter, entity_chroma_adapter) -> None:
        self._neo4j = neo4j_adapter
        self._chroma = entity_chroma_adapter
        self._logger = get_logger(__name__)

    # ── Public API ────────────────────────────────────────────────────────────

    async def validate_and_enrich(self, tickers: List[str]) -> Dict[str, TickerInfo]:
        """
        Validate up to MAX_TICKERS tickers and enrich with yfinance data.

        Returns a dict of {TICKER: TickerInfo}. Tickers beyond MAX_TICKERS
        are silently dropped. All I/O is async-safe (to_thread wrapped).
        """
        tickers = [t.upper().strip() for t in tickers[:MAX_TICKERS] if t.strip()]
        if not tickers:
            return {}

        # ── Step 1: batch validate via fast_info ──────────────────────────────
        results: Dict[str, TickerInfo] = await asyncio.to_thread(
            self._batch_validate_sync, tickers
        )

        # ── Step 2: separate valid equities into new vs known ─────────────────
        valid_equities = [
            t
            for t, info in results.items()
            if info.is_valid and info.is_equity and not info.needs_confirmation
        ]

        new_tickers: List[str] = []
        for t in valid_equities:
            # TODO: swap for neo4j get_company_entity_by_ticker() to avoid
            #       the yfinance .info network call for already-known companies.
            existing_id = await self._neo4j.entity_exists_by_ticker(t)
            if existing_id:
                results[t].is_new = False
            else:
                results[t].is_new = True
                new_tickers.append(t)

        # ── Step 3: batch-fetch full .info for ALL valid equities ─────────────
        # New tickers: info used for context AND taxonomy upsert.
        # Known tickers: info used only for context block (description preserved in graph).
        if valid_equities:
            enriched = await asyncio.to_thread(self._batch_enrich_sync, valid_equities)
            for t, data in enriched.items():
                if t in results:
                    info = results[t]
                    info.long_name = data.get("long_name", t)
                    info.description = data.get("description", "")
                    info.sector = data.get("sector", "")
                    info.industry = data.get("industry", "")

        # ── Step 4: background taxonomy upsert for new companies only ─────────
        for t in new_tickers:
            info = results[t]
            if info.long_name:
                asyncio.create_task(
                    self._upsert_company_taxonomy(info),
                    name=f"taxonomy_upsert_{t}",
                )

        return results

    # ── Sync helpers (executed in thread pool) ────────────────────────────────

    def _batch_validate_sync(self, tickers: List[str]) -> Dict[str, TickerInfo]:
        """
        Batch-validate tickers using yf.Tickers fast_info.
        Falls back to yf.Lookup for tickers whose fast_info raises.
        """
        results: Dict[str, TickerInfo] = {}
        try:
            batch = yf.Tickers(" ".join(tickers))
        except Exception as exc:
            self._logger.error("yf.Tickers init failed: %s", exc)
            for t in tickers:
                results[t] = TickerInfo(
                    ticker=t, is_valid=False, is_equity=False, quote_type=None
                )
            return results

        for t in tickers:
            ticker_obj = batch.tickers.get(t)
            if ticker_obj is None:
                results[t] = TickerInfo(
                    ticker=t, is_valid=False, is_equity=False, quote_type=None
                )
                continue
            try:
                fi = ticker_obj.fast_info
                quote_type: Optional[str] = getattr(fi, "quote_type", None)
                is_valid = quote_type is not None
                is_equity = str(quote_type or "").upper() in _EQUITY_QUOTE_TYPES
                # Non-equity securities (ETF, MUTUALFUND, etc.) need confirmation
                needs_confirmation = is_valid and not is_equity
                results[t] = TickerInfo(
                    ticker=t,
                    is_valid=is_valid,
                    is_equity=is_equity,
                    quote_type=quote_type,
                    needs_confirmation=needs_confirmation,
                )
            except Exception as exc:
                # fast_info failed — ticker may be invalid or a company name
                self._logger.warning(
                    "fast_info failed for '%s': %s — attempting Lookup fallback", t, exc
                )
                suggestions = self._lookup_sync(t)
                results[t] = TickerInfo(
                    ticker=t,
                    is_valid=False,
                    is_equity=False,
                    quote_type=None,
                    needs_confirmation=bool(suggestions),
                    suggestions=suggestions,
                )

        return results

    def _batch_enrich_sync(self, tickers: List[str]) -> Dict[str, dict]:
        """Batch-fetch .info for a list of already-validated tickers."""
        enriched: Dict[str, dict] = {}
        try:
            batch = yf.Tickers(" ".join(tickers))
        except Exception as exc:
            self._logger.error("yf.Tickers enrich init failed: %s", exc)
            return enriched

        for t in tickers:
            ticker_obj = batch.tickers.get(t)
            if ticker_obj is None:
                continue
            try:
                info = ticker_obj.info or {}
                enriched[t] = {
                    "long_name": info.get("longName", t),
                    "description": info.get("longBusinessSummary", ""),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                }
            except Exception as exc:
                self._logger.warning(".info fetch failed for '%s': %s", t, exc)

        return enriched

    def _lookup_sync(self, query: str) -> List[str]:
        """Use yf.Lookup as fallback to find ticker suggestions for ambiguous input."""
        try:
            results = yf.Lookup(query).get_equity(count=3)
            if results is not None and not results.empty:
                return results.index.tolist()[:3]
        except Exception as exc:
            self._logger.debug("Lookup fallback failed for '%s': %s", query, exc)
        return []

    # ── Taxonomy upsert (runs as background asyncio task) ─────────────────────

    async def _upsert_company_taxonomy(self, info: TickerInfo) -> None:
        """
        For a newly discovered company, upsert:
          1. Industry entity node (if info.industry is populated)
          2. Company entity node
          3. BELONGS_TO edges: Company → Industry → Sector (or Company → Sector directly)

        Sector and Market nodes are assumed to already exist (bootstrapped at
        startup via NodeSetManager.initialize_default_nodesets).

        All exceptions are caught here so a taxonomy failure never surfaces
        to the user or blocks other background tasks.
        """
        try:
            # ── Industry node ─────────────────────────────────────────────────
            industry_id: Optional[str] = None
            if info.industry:
                industry_id = canonical_entity_id(info.industry, "Industry")
                industry_exists = await self._neo4j.entity_exists(industry_id)
                if not industry_exists:
                    industry_node = EntityNode(
                        id=industry_id,
                        name=info.industry,
                        entity_type="Industry",
                        description=f"Industry segment: {info.industry}.",
                    )
                    await self._neo4j.merge_entity_node(industry_node)
                    if self._chroma is not None:
                        await self._chroma.upsert_entity_embedding(
                            entity_id=industry_id,
                            name=info.industry,
                            description=industry_node.description,
                            entity_type="Industry",
                        )
                    # Industry → Sector BELONGS_TO edge
                    if info.sector:
                        sector_id = canonical_entity_id(info.sector, "Sector")
                        await self._neo4j.merge_relationship(
                            industry_id,
                            sector_id,
                            "BELONGS_TO",
                            {
                                "relationship_type": "BELONGS_TO",
                                "source_agent": "taxonomy_bootstrap",
                            },
                        )
                    self._logger.info("Upserted Industry entity: %s", info.industry)

            # ── Company node ──────────────────────────────────────────────────
            company_id = canonical_entity_id(info.long_name, "Company")
            company_exists = await self._neo4j.entity_exists(company_id)
            if not company_exists:
                company_node = EntityNode(
                    id=company_id,
                    name=info.long_name,
                    entity_type="Company",
                    description=info.description,
                    ticker=info.ticker,
                )
                await self._neo4j.merge_entity_node(company_node)
                if self._chroma is not None:
                    await self._chroma.upsert_entity_embedding(
                        entity_id=company_id,
                        name=info.long_name,
                        description=info.description,
                        entity_type="Company",
                    )

                # Company → Industry or directly → Sector BELONGS_TO edge
                if industry_id:
                    await self._neo4j.merge_relationship(
                        company_id,
                        industry_id,
                        "BELONGS_TO",
                        {
                            "relationship_type": "BELONGS_TO",
                            "source_agent": "taxonomy_bootstrap",
                        },
                    )
                elif info.sector:
                    sector_id = canonical_entity_id(info.sector, "Sector")
                    await self._neo4j.merge_relationship(
                        company_id,
                        sector_id,
                        "BELONGS_TO",
                        {
                            "relationship_type": "BELONGS_TO",
                            "source_agent": "taxonomy_bootstrap",
                        },
                    )
                self._logger.info(
                    "Upserted Company entity: %s (%s)", info.long_name, info.ticker
                )
        except Exception:
            self._logger.exception(
                "_upsert_company_taxonomy failed for ticker '%s'", info.ticker
            )
