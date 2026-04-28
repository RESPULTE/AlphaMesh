"""
api/routers/conversations.py

Conversation history endpoints.

GET /api/v1/conversations                        list recent conversations
GET /api/v1/conversations/{id}/messages          full message history
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import (
    get_current_user_optional,
    get_session_service,
    get_store,
)
from api.models.responses import (
    ConversationHistoryResponse,
    ConversationMessage,
    ConversationSummary,
    ConversationTurn,
    ConversationTurnsResponse,
)
from api.services.conversation_service import ConversationStore
from api.services.session_service import SessionService
from core.config import settings

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


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


@router.get(
    "",
    response_model=List[ConversationSummary],
    summary="List recent conversations",
)
async def list_conversations(
    user_id_from_token: str | None = Depends(get_current_user_optional),
    user_email: str | None = Query(default=None),
    limit: int = Query(50, ge=1, le=200),
    store: ConversationStore = Depends(get_store),
) -> List[ConversationSummary]:
    resolved_user_email = _resolve_user_email(user_id_from_token, user_email)
    rows = await store.list_conversations(user_email=resolved_user_email, limit=limit)
    return [
        ConversationSummary(
            conversation_id=r["conversation_id"],
            created_at=r["created_at"],
            last_message_at=r["last_message_at"],
            message_count=r["message_count"],
        )
        for r in rows
    ]


@router.get(
    "/{conversation_id}/messages",
    response_model=ConversationHistoryResponse,
    summary="Retrieve full message history for a conversation",
)
async def get_messages(
    conversation_id: str,
    user_id_from_token: str | None = Depends(get_current_user_optional),
    user_email: str | None = Query(default=None),
    session_svc: SessionService = Depends(get_session_service),
    store: ConversationStore = Depends(get_store),
) -> ConversationHistoryResponse:
    resolved_user_email = _resolve_user_email(user_id_from_token, user_email)
    allowed = await session_svc.user_has_conversation(
        user_id=resolved_user_email,
        conversation_id=conversation_id,
    )
    if not allowed:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await store.get_history(
        conversation_id,
        user_email=resolved_user_email,
    )
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=[
            ConversationMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", ""),
            )
            for m in messages
        ],
    )


@router.get(
    "/{conversation_id}/turns",
    response_model=ConversationTurnsResponse,
    summary="Retrieve structured turn history for a conversation",
)
async def get_turns(
    conversation_id: str,
    user_id_from_token: str | None = Depends(get_current_user_optional),
    user_email: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=100),
    before_turn_id: str | None = Query(default=None),
    session_svc: SessionService = Depends(get_session_service),
    store: ConversationStore = Depends(get_store),
) -> ConversationTurnsResponse:
    resolved_user_email = _resolve_user_email(user_id_from_token, user_email)
    allowed = await session_svc.user_has_conversation(
        user_id=resolved_user_email,
        conversation_id=conversation_id,
    )
    if not allowed:
        raise HTTPException(status_code=404, detail="Conversation not found")
    turns, has_more, next_before_turn_id = await store.get_turns_paginated(
        conversation_id,
        user_email=resolved_user_email,
        limit=limit,
        before_turn_id=before_turn_id,
    )
    parsed_turns: list[ConversationTurn] = []
    for turn in turns:
        try:
            parsed_turns.append(ConversationTurn.model_validate(turn))
        except Exception:
            continue
    return ConversationTurnsResponse(
        conversation_id=conversation_id,
        turns=parsed_turns,
        has_more=has_more,
        next_before_turn_id=next_before_turn_id,
    )
