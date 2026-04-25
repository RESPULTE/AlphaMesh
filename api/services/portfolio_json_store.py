"""
api/services/portfolio_json_store.py

Per-user JSON portfolio persistence with atomic writes.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import List

from api.models.portfolio import PortfolioHolding
from core.config import settings


class PortfolioJsonStore:
    """
    Stores one JSON portfolio file per user under the data directory.

    File format is a plain list of holdings:
    [
      {"ticker": "...", "company_name": "...", "exchange": "...", "asset_type": "...", "shares": 12.0}
    ]
    """

    def __init__(self, base_path: str | None = None) -> None:
        base = Path(base_path or settings.PORTFOLIO_JSON_PATH)
        self._dir = base.parent
        self._prefix = base.stem
        self._lock = asyncio.Lock()

    def _sanitize_user_email(self, user_email: str) -> str:
        value = (user_email or "").strip().lower()
        safe = re.sub(r"[^a-z0-9._-]+", "_", value).strip("._-")
        if not safe:
            raise ValueError("Invalid user_email")
        return safe

    def get_portfolio_file_path(self, user_email: str) -> Path:
        safe_user = self._sanitize_user_email(user_email)
        return self._dir / f"{self._prefix}_{safe_user}.json"

    def _read_holdings_sync(self, path: Path) -> List[PortfolioHolding]:
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("Portfolio file content must be a JSON list")
        return [PortfolioHolding.model_validate(item) for item in payload]

    def _ensure_file_sync(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        path.write_text("[]", encoding="utf-8")

    def _write_holdings_sync(self, path: Path, holdings: List[PortfolioHolding]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        payload = [
            holding.model_dump(mode="json", exclude_none=True) for holding in holdings
        ]
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _sort_holdings(holdings: List[PortfolioHolding]) -> List[PortfolioHolding]:
        return sorted(holdings, key=lambda item: item.ticker)

    async def get_holdings(self, user_email: str) -> List[PortfolioHolding]:
        path = self.get_portfolio_file_path(user_email)
        async with self._lock:
            await asyncio.to_thread(self._ensure_file_sync, path)
            holdings = await asyncio.to_thread(self._read_holdings_sync, path)
        return self._sort_holdings(holdings)

    async def upsert_holding(
        self,
        user_email: str,
        holding: PortfolioHolding,
    ) -> List[PortfolioHolding]:
        path = self.get_portfolio_file_path(user_email)
        normalized = holding.model_copy(update={"ticker": holding.ticker.upper()})

        async with self._lock:
            await asyncio.to_thread(self._ensure_file_sync, path)
            holdings = await asyncio.to_thread(self._read_holdings_sync, path)
            replace_index: int | None = None

            for index, current in enumerate(holdings):
                if current.ticker.upper() == normalized.ticker:
                    replace_index = index
                    break

            if replace_index is None:
                holdings.append(normalized)
            else:
                holdings[replace_index] = normalized

            holdings = self._sort_holdings(holdings)
            await asyncio.to_thread(self._write_holdings_sync, path, holdings)

        return holdings

