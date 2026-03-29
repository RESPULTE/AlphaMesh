"""
api/auth/adapter.py

Authentication adapter interface + dummy implementation.

Design
──────
All authentication logic is behind the `AuthAdapter` abstract base class.
The `DummyAuthAdapter` is the current implementation — it accepts any
well-formed email as a valid identity, which is sufficient for local
development and integration testing without a real identity provider.

Replacement plan
────────────────
When Firebase Authentication is ready, implement `FirebaseAuthAdapter`
which validates the Bearer token against Firebase's token verification
endpoint.  Wire it into `api/deps.py` by swapping the adapter instance —
no other files need to change.

Usage
─────
The active adapter is a module-level singleton (`_adapter`) returned by
`get_auth_adapter()`.  FastAPI dependency functions in `api/deps.py` call
this function rather than importing the concrete class, preserving the
swap-without-change promise.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from jose import JWTError, jwt

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────


class AuthAdapter(ABC):
    """
    Contract every auth implementation must satisfy.

    verify_access_token(token) → user_email | None
    create_access_token(email) → signed JWT string
    create_refresh_token(email) → signed JWT string
    verify_refresh_token(token) → user_email | None
    """

    @abstractmethod
    def verify_access_token(self, token: str) -> Optional[str]:
        """Return the user's email if token is valid, else None."""

    @abstractmethod
    def create_access_token(self, email: str) -> str:
        """Issue a new access token for the given email."""

    @abstractmethod
    def create_refresh_token(self, email: str) -> str:
        """Issue a new refresh token for the given email."""

    @abstractmethod
    def verify_refresh_token(self, token: str) -> Optional[str]:
        """Return the user's email if the refresh token is valid, else None."""


# ── Dummy implementation (replace with FirebaseAuthAdapter when ready) ─────────


class DummyAuthAdapter(AuthAdapter):
    """
    Development-only adapter.

    • Accepts any syntactically valid email as a login credential (no password check).
    • Issues real HMAC-signed JWTs using core/config.py JWT_SECRET_KEY.
    • The token format is identical to what a production adapter would produce,
      so the rest of the system (deps.py, middleware) needs zero changes at swap time.

    SECURITY: Do not use in production.  There is no password validation.
    """

    def _make_token(self, email: str, token_type: str, expire_seconds: int) -> str:
        payload = {
            "sub": email,
            "type": token_type,
            "iat": time.time(),
            "exp": time.time() + expire_seconds,
        }
        return jwt.encode(
            payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

    def _decode(self, token: str, expected_type: str) -> Optional[str]:
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            if payload.get("type") != expected_type:
                return None
            email: str = payload.get("sub", "")
            return email if email else None
        except JWTError as exc:
            logger.debug("DummyAuthAdapter: token verification failed: %s", exc)
            return None

    def verify_access_token(self, token: str) -> Optional[str]:
        return self._decode(token, "access")

    def create_access_token(self, email: str) -> str:
        return self._make_token(
            email, "access", settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    def create_refresh_token(self, email: str) -> str:
        return self._make_token(
            email, "refresh", settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 86400
        )

    def verify_refresh_token(self, token: str) -> Optional[str]:
        return self._decode(token, "refresh")


# ── Singleton ─────────────────────────────────────────────────────────────────

# Swap this line to activate a different adapter:
#   _adapter: AuthAdapter = FirebaseAuthAdapter()
_adapter: AuthAdapter = DummyAuthAdapter()


def get_auth_adapter() -> AuthAdapter:
    """Return the active auth adapter singleton."""
    return _adapter
