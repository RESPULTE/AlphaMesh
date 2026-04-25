from __future__ import annotations

import asyncio

from api.models.portfolio import PortfolioHolding
from api.services.portfolio_json_store import PortfolioJsonStore


def test_upsert_creates_file_and_holding(tmp_path) -> None:
    store = PortfolioJsonStore(base_path=str(tmp_path / "portfolio.json"))
    user = "alpha@example.com"

    holdings = asyncio.run(store.upsert_holding(
        user,
        PortfolioHolding(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            asset_type="equity",
            shares=125,
        ),
    ))

    path = store.get_portfolio_file_path(user)
    assert path.exists()
    assert len(holdings) == 1
    assert holdings[0].ticker == "AAPL"
    assert holdings[0].shares == 125


def test_get_holdings_creates_empty_file_for_new_user(tmp_path) -> None:
    store = PortfolioJsonStore(base_path=str(tmp_path / "portfolio.json"))
    user = "new.user@example.com"

    path = store.get_portfolio_file_path(user)
    assert not path.exists()

    holdings = asyncio.run(store.get_holdings(user))

    assert holdings == []
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() == "[]"


def test_upsert_replaces_existing_ticker_case_insensitive(tmp_path) -> None:
    store = PortfolioJsonStore(base_path=str(tmp_path / "portfolio.json"))
    user = "alpha@example.com"

    asyncio.run(store.upsert_holding(
        user,
        PortfolioHolding(
            ticker="aapl",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            asset_type="equity",
            shares=10,
        ),
    ))

    holdings = asyncio.run(store.upsert_holding(
        user,
        PortfolioHolding(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            asset_type="equity",
            shares=25,
        ),
    ))

    assert len(holdings) == 1
    assert holdings[0].ticker == "AAPL"
    assert holdings[0].shares == 25


def test_per_user_file_isolation(tmp_path) -> None:
    store = PortfolioJsonStore(base_path=str(tmp_path / "portfolio.json"))

    asyncio.run(store.upsert_holding(
        "user1@example.com",
        PortfolioHolding(
            ticker="TSLA",
            company_name="Tesla, Inc.",
            exchange="NASDAQ",
            asset_type="equity",
            shares=15,
        ),
    ))
    asyncio.run(store.upsert_holding(
        "user2@example.com",
        PortfolioHolding(
            ticker="TSLA",
            company_name="Tesla, Inc.",
            exchange="NASDAQ",
            asset_type="equity",
            shares=40,
        ),
    ))

    holdings_user_1 = asyncio.run(store.get_holdings("user1@example.com"))
    holdings_user_2 = asyncio.run(store.get_holdings("user2@example.com"))
    assert holdings_user_1[0].shares == 15
    assert holdings_user_2[0].shares == 40
    assert store.get_portfolio_file_path("user1@example.com") != store.get_portfolio_file_path(
        "user2@example.com"
    )


def test_concurrent_upsert_keeps_single_row(tmp_path) -> None:
    store = PortfolioJsonStore(base_path=str(tmp_path / "portfolio.json"))
    user = "alpha@example.com"

    async def set_shares(shares: float, ticker: str) -> None:
        await store.upsert_holding(
            user,
            PortfolioHolding(
                ticker=ticker,
                company_name="Apple Inc.",
                exchange="NASDAQ",
                asset_type="equity",
                shares=shares,
            ),
        )

    async def run_updates() -> None:
        await asyncio.gather(
            set_shares(33, "AAPL"),
            set_shares(77, "aapl"),
        )

    asyncio.run(run_updates())

    holdings = asyncio.run(store.get_holdings(user))
    assert len(holdings) == 1
    assert holdings[0].ticker == "AAPL"
    assert holdings[0].shares in {33, 77}
