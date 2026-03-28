"""
api/routers/conversations.py

Conversation history endpoints.

GET /api/v1/conversations                        — list recent conversations
GET /api/v1/conversations/{id}/messages          — full message history
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_store
from api.models.responses import ConversationHistoryResponse, ConversationSummary
from api.services.conversation_store import ConversationStore

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


@router.get(
    "",
    response_model=List[ConversationSummary],
    summary="List recent conversations",
)
async def list_conversations(
    user_email: Optional[str] = Query(
        default=None, description="Filter by user e-mail."
    ),
    limit: int = Query(default=50, ge=1, le=200),
    store: ConversationStore = Depends(get_store),
) -> List[ConversationSummary]:
    return await store.list_conversations(user_email=user_email, limit=limit)


@router.get(
    "/{conversation_id}/messages",
    response_model=ConversationHistoryResponse,
    summary="Retrieve full message history for a conversation",
)
async def get_messages(
    conversation_id: str,
    store: ConversationStore = Depends(get_store),
) -> ConversationHistoryResponse:
    try:
        return await store.get_history_response(conversation_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
