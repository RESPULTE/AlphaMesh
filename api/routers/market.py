from fastapi import APIRouter, Depends

from api.dependencies import get_market_data_service
from core.market_data_service import MarketDataService

router = APIRouter(tags=["market"])


@router.get("/{ticker}/quote")
async def get_quote(
    ticker: str,
    svc: MarketDataService = Depends(get_market_data_service),
):
    return await svc.get_quote(ticker)


@router.get("/{ticker}/intraday")
async def get_intraday(
    ticker: str,
    svc: MarketDataService = Depends(get_market_data_service),
):
    return await svc.get_intraday(ticker)
