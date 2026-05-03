from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.auth.adapter import get_auth_adapter
from api.dependencies import get_current_user, get_runner, get_session_service
from api.routers import auth as auth_router
from api.routers import chat as chat_router


class _FakeSessionService:
    async def ensure_session(self, user_id: str, session_id: str | None = None) -> str:
        if session_id:
            return session_id
        return f"session-{user_id}"


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def launch(
        self, request_id: str, _body, *, user_id: str, session_id: str
    ) -> str:
        self.calls.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "session_id": session_id,
            }
        )
        return "conv-token-user"


def _build_auth_client() -> TestClient:
    app = FastAPI()
    app.include_router(auth_router.router)
    app.dependency_overrides[get_session_service] = lambda: _FakeSessionService()
    return TestClient(app)


def test_signup_and_login_issue_valid_tokens() -> None:
    client = _build_auth_client()

    signup = client.post("/api/v1/auth/signup", json={"email": "User@Test.com"})
    assert signup.status_code == 200
    payload = signup.json()

    assert payload["token_type"] == "bearer"
    assert payload["user_email"] == "user@test.com"
    assert payload["session_id"] == "session-user@test.com"
    assert payload["expires_in"] > 0
    assert get_auth_adapter().verify_access_token(payload["access_token"]) == "user@test.com"
    assert get_auth_adapter().verify_refresh_token(payload["refresh_token"]) == "user@test.com"

    login = client.post("/api/v1/auth/login", json={"email": "user@test.com"})
    assert login.status_code == 200
    login_payload = login.json()
    assert login_payload["user_email"] == "user@test.com"


def test_refresh_returns_new_access_for_valid_refresh_token() -> None:
    client = _build_auth_client()
    issued = client.post("/api/v1/auth/login", json={"email": "refresh@test.com"})
    assert issued.status_code == 200
    refresh_token = issued.json()["refresh_token"]

    refreshed = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert refreshed.status_code == 200
    payload = refreshed.json()

    assert payload["user_email"] == "refresh@test.com"
    assert get_auth_adapter().verify_access_token(payload["access_token"]) == "refresh@test.com"


def test_refresh_rejects_invalid_token() -> None:
    client = _build_auth_client()
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": "invalid-token"})
    assert response.status_code == 401


def test_chat_accepts_bearer_identity_without_user_email() -> None:
    app = FastAPI()
    app.include_router(auth_router.router)
    app.include_router(chat_router.router)

    fake_runner = _FakeRunner()
    app.dependency_overrides[get_session_service] = lambda: _FakeSessionService()
    app.dependency_overrides[get_runner] = lambda: fake_runner

    client = TestClient(app)
    auth = client.post("/api/v1/auth/login", json={"email": "chat@test.com"})
    assert auth.status_code == 200
    token = auth.json()["access_token"]

    chat = client.post(
        "/api/v1/chat",
        json={"message": "Analyze AAPL", "conversation_id": None, "session_id": None},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert chat.status_code == 202
    payload = chat.json()
    assert payload["conversation_id"] == "conv-token-user"
    assert payload["session_id"] == "session-chat@test.com"
    assert fake_runner.calls and fake_runner.calls[0]["user_id"] == "chat@test.com"


def test_get_current_user_accepts_query_token_for_sse_style_requests() -> None:
    app = FastAPI()

    @app.get("/whoami")
    async def whoami(user_id: str = Depends(get_current_user)):
        return {"user_id": user_id}

    token = get_auth_adapter().create_access_token("sse@test.com")
    client = TestClient(app)

    response = client.get(f"/whoami?token={token}")
    assert response.status_code == 200
    assert response.json()["user_id"] == "sse@test.com"
