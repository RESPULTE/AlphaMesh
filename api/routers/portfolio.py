from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from api.dependencies import get_current_user_optional
from api.models.portfolio import (
    PortfolioHolding,
    PortfolioResponse,
    TickerSearchResponse,
    TickerSearchResult,
    UpsertPortfolioHoldingRequest,
)
from api.services.portfolio_json_store import PortfolioJsonStore
from core.config import settings

router = APIRouter(prefix="/api/v1", tags=["portfolio"])

_SUPPORTED_ASSET_TYPES = {"EQUITY": "equity", "ETF": "etf"}


def get_portfolio_store(request: Request) -> PortfolioJsonStore:
    return request.app.state.portfolio_store


def _resolve_user_email(
    user_id_from_token: str | None,
    user_email_fallback: str | None,
) -> str:
    if user_id_from_token:
        return user_id_from_token
    if settings.DEV_ALLOW_USER_EMAIL_FALLBACK and user_email_fallback:
        return user_email_fallback
    raise HTTPException(
        status_code=401,
        detail="Missing user identity: provide Bearer token or dev user_email fallback.",
    )


def _normalize_asset_type(raw_quote_type: Any) -> Optional[str]:
    if not raw_quote_type:
        return None
    key = str(raw_quote_type).upper().strip()
    return _SUPPORTED_ASSET_TYPES.get(key)


def _extract_company_name(quote: dict) -> str:
    return (
        str(quote.get("shortname") or quote.get("longname") or quote.get("symbol") or "")
        .strip()
    )


def _extract_exchange(quote: dict) -> Optional[str]:
    exchange = (
        quote.get("exchDisp")
        or quote.get("exchange")
        or quote.get("exchangeName")
        or quote.get("fullExchangeName")
    )
    return str(exchange).strip() if exchange else None


def _map_search_quotes(quotes: List[dict], limit: int) -> List[TickerSearchResult]:
    seen: set[str] = set()
    results: List[TickerSearchResult] = []

    for quote in quotes:
        symbol = str(quote.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            continue

        asset_type = _normalize_asset_type(quote.get("quoteType"))
        if asset_type is None:
            continue

        company_name = _extract_company_name(quote)
        if not company_name:
            continue

        results.append(
            TickerSearchResult(
                ticker=symbol,
                company_name=company_name,
                exchange=_extract_exchange(quote),
                asset_type=asset_type,
            )
        )
        seen.add(symbol)

        if len(results) >= limit:
            break

    return results


def _run_yfinance_search(query: str, max_results: int) -> List[dict]:
    search = yf.Search(
        query=query,
        max_results=max_results,
        news_count=0,
        lists_count=0,
        include_cb=False,
        include_nav_links=False,
        include_research=False,
        include_cultural_assets=False,
        recommended=max_results,
    )
    return search.quotes or []


@router.get(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="Get user portfolio holdings",
)
async def get_portfolio(
    user_id_from_token: str | None = Depends(get_current_user_optional),
    user_email: str | None = Query(default=None, min_length=3, max_length=320),
    store: PortfolioJsonStore = Depends(get_portfolio_store),
) -> PortfolioResponse:
    resolved_user_email = _resolve_user_email(user_id_from_token, user_email)
    try:
        holdings = await store.get_holdings(resolved_user_email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to load portfolio holdings",
        ) from exc
    return PortfolioResponse(user_email=resolved_user_email, holdings=holdings)


@router.put(
    "/portfolio/holding",
    response_model=PortfolioResponse,
    summary="Create or update one holding by setting total shares",
)
async def upsert_portfolio_holding(
    body: UpsertPortfolioHoldingRequest,
    user_id_from_token: str | None = Depends(get_current_user_optional),
    store: PortfolioJsonStore = Depends(get_portfolio_store),
) -> PortfolioResponse:
    resolved_user_email = _resolve_user_email(user_id_from_token, body.user_email)
    holding = PortfolioHolding(
        ticker=body.ticker.upper(),
        company_name=body.company_name.strip(),
        exchange=body.exchange.strip() if body.exchange else None,
        asset_type=body.asset_type,
        shares=body.shares,
    )
    try:
        holdings = await store.upsert_holding(resolved_user_email, holding)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to update portfolio holdings",
        ) from exc
    return PortfolioResponse(user_email=resolved_user_email, holdings=holdings)


@router.get(
    "/tickers/search",
    response_model=TickerSearchResponse,
    summary="Search ticker suggestions (equities + ETFs)",
)
async def search_tickers(
    q: str = Query(..., max_length=64),
    limit: int = Query(8, ge=1, le=10),
) -> TickerSearchResponse:
    query = q.strip()
    if len(query) < 2:
        return TickerSearchResponse(results=[])

    try:
        raw_quotes = await asyncio.to_thread(_run_yfinance_search, query, limit * 3)
        results = _map_search_quotes(raw_quotes, limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Ticker search provider unavailable",
        ) from exc

    return TickerSearchResponse(results=results)
