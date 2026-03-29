from fastapi import APIRouter, Depends

from api.deps import get_current_user, get_market_data_service
from api.services.market_data_service import MarketDataService

router = APIRouter(tags=["market"])


@router.get("/{ticker}/quote")
async def get_quote(
    ticker: str,
    _: str = Depends(get_current_user),
    svc: MarketDataService = Depends(get_market_data_service),
):
    return await svc.get_quote(ticker)


@router.get("/{ticker}/intraday")
async def get_intraday(
    ticker: str,
    _: str = Depends(get_current_user),
    svc: MarketDataService = Depends(get_market_data_service),
):
    return await svc.get_intraday(ticker)
