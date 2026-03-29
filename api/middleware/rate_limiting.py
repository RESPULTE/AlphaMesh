"""
api/middleware/rate_limiting.py

Token-bucket rate limiter per authenticated user.

• In-process dict — no Redis required; suitable for single uvicorn worker.
• For horizontal scaling, replace _BUCKETS with a Redis-backed store (e.g.
  slowapi with aioredis) — the middleware interface stays identical.
• Unauthenticated requests pass through unrestricted; auth failures are
  handled downstream by the dependency layer.
• The /api/analyze path gets a tighter limit (RATE_LIMIT_ANALYZE, default 10
  rpm) because each call runs the full LLM agent pipeline.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)

# bucket: user_identity → (tokens_remaining, last_refill_unix)
_BUCKETS: Dict[str, Tuple[float, float]] = {}


def _extract_user(request: Request) -> str:
    """
    Attempt to extract the user identity from the request without full JWT
    verification (expensive).  Returns "anonymous" on any failure; the auth
    dependency will reject invalid tokens independently.
    """
    token = (
        request.query_params.get("token")
        or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    if not token:
        return "anonymous"
    try:
        from jose import jwt as _jwt

        payload = _jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},  # expiry checked by auth dep
        )
        return payload.get("sub", "anonymous") or "anonymous"
    except Exception:
        return "anonymous"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        user = _extract_user(request)
        if user == "anonymous":
            # Unauthenticated requests pass through; auth deps handle rejection
            return await call_next(request)

        limit = (
            settings.RATE_LIMIT_ANALYZE
            if "/analyze" in request.url.path
            else settings.RATE_LIMIT_DEFAULT
        )

        now = time.time()
        tokens, last = _BUCKETS.get(user, (float(limit), now))
        elapsed = now - last
        # Refill at <limit> tokens per minute
        tokens = min(float(limit), tokens + elapsed * (limit / 60.0))

        if tokens < 1.0:
            logger.warning(
                "Rate limit exceeded for user '%s' on %s", user, request.url.path
            )
            return Response(
                status_code=429,
                content='{"detail":"Rate limit exceeded. Please wait before re-submitting."}',
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        _BUCKETS[user] = (tokens - 1.0, now)
        return await call_next(request)
