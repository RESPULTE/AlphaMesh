"""
api/routers/conversations.py

Conversation history endpoints.

GET /api/v1/conversations                        list recent conversations
GET /api/v1/conversations/{id}/messages          full message history
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_current_user, get_session_service, get_store
from api.models.responses import (
    ConversationHistoryResponse,
    ConversationMessage,
    ConversationSummary,
)
from api.services.conversation_service import ConversationStore
from api.services.session_service import SessionService

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.get(
    "",
    response_model=List[ConversationSummary],
    summary="List recent conversations",
)
async def list_conversations(
    user_email: str = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    store: ConversationStore = Depends(get_store),
) -> List[ConversationSummary]:
    rows = await store.list_conversations(user_email=user_email, limit=limit)
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
    user_id: str = Depends(get_current_user),
    session_svc: SessionService = Depends(get_session_service),
    store: ConversationStore = Depends(get_store),
) -> ConversationHistoryResponse:
    allowed = await session_svc.user_has_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    if not allowed:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await store.get_history(conversation_id)
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
