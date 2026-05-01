from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth.adapter import get_auth_adapter
from api.routers import portfolio as portfolio_router
from api.services.portfolio_json_store import PortfolioJsonStore
from core.config import settings


def _build_test_client(tmp_path) -> TestClient:
    app = FastAPI()
    app.include_router(portfolio_router.router)
    store = PortfolioJsonStore(base_path=str(tmp_path / "portfolio.json"))
    app.dependency_overrides[portfolio_router.get_portfolio_store] = lambda: store
    return TestClient(app)


def test_portfolio_holding_upsert_and_get(tmp_path) -> None:
    client = _build_test_client(tmp_path)

    body = {
        "user_email": "demo@alphamesh.local",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "asset_type": "equity",
        "shares": 12,
    }
    response = client.put("/api/v1/portfolio/holding", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["holdings"]) == 1
    assert payload["holdings"][0]["ticker"] == "AAPL"
    assert payload["holdings"][0]["shares"] == 12

    # Case-insensitive ticker should replace, not duplicate.
    body["ticker"] = "aapl"
    body["shares"] = 30
    response = client.put("/api/v1/portfolio/holding", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["holdings"]) == 1
    assert payload["holdings"][0]["ticker"] == "AAPL"
    assert payload["holdings"][0]["shares"] == 30

    response = client.get("/api/v1/portfolio", params={"user_email": "demo@alphamesh.local"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_email"] == "demo@alphamesh.local"
    assert len(payload["holdings"]) == 1
    assert payload["holdings"][0]["ticker"] == "AAPL"


def test_portfolio_validation_errors(tmp_path) -> None:
    client = _build_test_client(tmp_path)

    invalid_shares = {
        "user_email": "demo@alphamesh.local",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "asset_type": "equity",
        "shares": 0,
    }
    response = client.put("/api/v1/portfolio/holding", json=invalid_shares)
    assert response.status_code == 422

    invalid_ticker = {
        "user_email": "demo@alphamesh.local",
        "ticker": "AAPL$",
        "company_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "asset_type": "equity",
        "shares": 10,
    }
    response = client.put("/api/v1/portfolio/holding", json=invalid_ticker)
    assert response.status_code == 422


def test_ticker_search_filters_and_dedupes(monkeypatch, tmp_path) -> None:
    client = _build_test_client(tmp_path)

    def fake_search(_query: str, _max_results: int):
        return [
            {
                "symbol": "AAPL",
                "shortname": "Apple Inc.",
                "exchange": "NMS",
                "quoteType": "EQUITY",
            },
            {
                "symbol": "AAPL",
                "shortname": "Apple Inc. Duplicate",
                "exchange": "NMS",
                "quoteType": "EQUITY",
            },
            {
                "symbol": "SPY",
                "shortname": "SPDR S&P 500 ETF Trust",
                "exchange": "PCX",
                "quoteType": "ETF",
            },
            {
                "symbol": "BTC-USD",
                "shortname": "Bitcoin USD",
                "exchange": "CCC",
                "quoteType": "CRYPTOCURRENCY",
            },
        ]

    monkeypatch.setattr(portfolio_router, "_run_yfinance_search", fake_search)

    response = client.get("/api/v1/tickers/search", params={"q": "apple", "limit": 8})
    assert response.status_code == 200
    payload = response.json()
    tickers = [row["ticker"] for row in payload["results"]]
    assert tickers == ["AAPL", "SPY"]


def test_ticker_search_short_query_returns_empty(monkeypatch, tmp_path) -> None:
    client = _build_test_client(tmp_path)

    called = {"value": False}

    def fake_search(_query: str, _max_results: int):
        called["value"] = True
        return []

    monkeypatch.setattr(portfolio_router, "_run_yfinance_search", fake_search)

    response = client.get("/api/v1/tickers/search", params={"q": "a"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert called["value"] is False


def test_portfolio_token_identity_overrides_user_email_fallback(tmp_path) -> None:
    client = _build_test_client(tmp_path)
    token = get_auth_adapter().create_access_token("token-owner@alphamesh.local")

    body = {
        "user_email": "different-user@alphamesh.local",
        "ticker": "MSFT",
        "company_name": "Microsoft Corp.",
        "exchange": "NASDAQ",
        "asset_type": "equity",
        "shares": 4,
    }
    response = client.put(
        "/api/v1/portfolio/holding",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_email"] == "token-owner@alphamesh.local"
    assert payload["holdings"][0]["ticker"] == "MSFT"

    response = client.get(
        "/api/v1/portfolio",
        params={"user_email": "different-user@alphamesh.local"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_email"] == "token-owner@alphamesh.local"
    assert len(payload["holdings"]) == 1


def test_portfolio_user_email_fallback_respects_dev_flag(monkeypatch, tmp_path) -> None:
    client = _build_test_client(tmp_path)
    monkeypatch.setattr(settings, "DEV_ALLOW_USER_EMAIL_FALLBACK", False)

    response = client.get(
        "/api/v1/portfolio",
        params={"user_email": "fallback@alphamesh.local"},
    )
    assert response.status_code == 401
