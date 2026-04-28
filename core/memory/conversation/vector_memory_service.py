from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import settings
from core.logger import get_logger
from core.memory.graph.nodeset_manager import hash_user_email
from core.memory.stores.chroma_adapter import ChromaDBAdapter

logger = get_logger(__name__)

_COLLECTION_PREFIX = "conv_private"
_MAX_SUMMARY_CHARS = 280


def _safe_turn_id(turn: dict) -> str:
    return str(turn.get("turn_id") or turn.get("request_id") or "").strip()


def _safe_created_at(turn: dict) -> str:
    return str(turn.get("created_at") or "").strip()


def _trim(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


class ConversationVectorMemoryService:
    """Private per-user conversation chunk indexing and retrieval service."""

    def __init__(self, chroma_adapter: ChromaDBAdapter, llm: Any) -> None:
        self._chroma = chroma_adapter
        self._llm = llm
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self._indexed_turn_ids: Dict[str, set[str]] = defaultdict(set)
        self._locks: Dict[str, asyncio.Lock] = {}

    @staticmethod
    def _cursor_key(user_email: str, conversation_id: str) -> str:
        return f"{user_email.strip().lower()}::{conversation_id.strip()}"

    @staticmethod
    def _collection_name_for_user(user_email: str) -> str:
        return f"{_COLLECTION_PREFIX}_{hash_user_email(user_email)}"

    def _get_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _build_turn_source_text(self, turn: dict) -> str:
        user_message = _trim(turn.get("user_message"), 1600)
        synthesis = _trim(turn.get("assistant_synthesis"), 1800)
        summaries = turn.get("agent_memory_summaries") or {}
        summary_text = ""
        if isinstance(summaries, dict) and summaries:
            summary_text = _trim(
                json.dumps(summaries, ensure_ascii=True, sort_keys=True),
                _MAX_SUMMARY_CHARS,
            )
        sections = [
            f"User: {user_message}" if user_message else "",
            f"Assistant: {synthesis}" if synthesis else "",
            f"AgentSummaries: {summary_text}" if summary_text else "",
        ]
        return "\n".join(part for part in sections if part).strip()

    async def _count_total_tokens(self, turns: List[dict]) -> int:
        if not turns:
            return 0
        text_parts: List[str] = []
        for turn in turns:
            rendered = self._build_turn_source_text(turn)
            if rendered:
                text_parts.append(rendered)
        if not text_parts:
            return 0

        try:
            return int(
                await asyncio.to_thread(self._llm.get_num_tokens, "\n\n".join(text_parts))
            )
        except Exception:
            logger.exception("_count_total_tokens: tokenizer call failed")
            return 0

    async def _is_threshold_exceeded(self, turns: List[dict]) -> bool:
        total_tokens = await self._count_total_tokens(turns)
        return total_tokens > settings.CONVERSATION_MEMORY_TOKEN_LIMIT

    def _make_chunk_id(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        chunk_index: int,
        chunk_text: str,
    ) -> str:
        digest = hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()[:12]
        return f"convmem::{conversation_id}::{turn_id}::{chunk_index}::{digest}"

    async def ensure_index(
        self,
        *,
        conversation_id: str,
        user_email: str,
        turns: List[dict],
    ) -> bool:
        """Index all unindexed turns when threshold is exceeded."""
        conversation_key = self._cursor_key(user_email, conversation_id)
        async with self._get_lock(conversation_key):
            threshold_exceeded = await self._is_threshold_exceeded(turns)
            if not threshold_exceeded:
                return False

            indexed_turn_ids = self._indexed_turn_ids[conversation_key]
            pending = [
                turn
                for turn in turns
                if _safe_turn_id(turn) and _safe_turn_id(turn) not in indexed_turn_ids
            ]
            if not pending:
                return True

            chunk_ids: List[str] = []
            chunk_texts: List[str] = []
            metadatas: List[dict] = []
            for turn in pending:
                turn_id = _safe_turn_id(turn)
                if not turn_id:
                    continue
                turn_text = self._build_turn_source_text(turn)
                if not turn_text:
                    indexed_turn_ids.add(turn_id)
                    continue
                chunks = self._splitter.split_text(turn_text)
                for idx, chunk_text in enumerate(chunks):
                    chunk_id = self._make_chunk_id(
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        chunk_index=idx,
                        chunk_text=chunk_text,
                    )
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(chunk_text)
                    metadatas.append(
                        {
                            "chunk_id": chunk_id,
                            "conversation_id": conversation_id,
                            "turn_id": turn_id,
                            "created_at": _safe_created_at(turn),
                            "chunk_index": idx,
                            "source": "conversation_private",
                        }
                    )
                indexed_turn_ids.add(turn_id)

            if not chunk_ids:
                return True

            try:
                await self._chroma.upsert_chunks(
                    chunk_ids=chunk_ids,
                    texts=chunk_texts,
                    metadatas=metadatas,
                    collection_name=self._collection_name_for_user(user_email),
                )
                return True
            except Exception:
                logger.exception("ensure_index: failed to upsert conversation chunks")
                return False

    async def retrieve(
        self,
        *,
        conversation_id: str,
        user_email: str,
        query: str,
    ) -> List[dict]:
        """Retrieve relevant private conversation chunks for the current query."""
        if not query.strip():
            return []
        try:
            docs_with_scores = await self._chroma.query(
                query_text=query,
                n_results=settings.MEMORY_VECTOR_TOP_K,
                search_type="similarity",
                where={"conversation_id": conversation_id},
                collection_name=self._collection_name_for_user(user_email),
            )
        except Exception:
            logger.exception("retrieve: failed to query private conversation memory")
            return []

        hits: List[dict] = []
        for doc, score in docs_with_scores:
            numeric_score = float(score) if score is not None else 0.0
            if numeric_score < settings.MEMORY_SIMILARITY_THRESHOLD:
                continue
            metadata = dict(doc.metadata or {})
            hits.append(
                {
                    "chunk_id": str(doc.id or metadata.get("chunk_id") or ""),
                    "turn_id": str(metadata.get("turn_id") or ""),
                    "created_at": str(metadata.get("created_at") or ""),
                    "text": _trim(doc.page_content or "", 420),
                    "score": numeric_score,
                }
            )
        return hits

    @staticmethod
    def render_hits_block(hits: List[dict]) -> str:
        if not hits:
            return "(none)"
        lines: List[str] = []
        for idx, hit in enumerate(hits, start=1):
            ts = str(hit.get("created_at") or "unknown_time")
            turn_id = str(hit.get("turn_id") or "unknown_turn")
            score = float(hit.get("score") or 0.0)
            text = str(hit.get("text") or "").strip()
            lines.append(
                f"{idx}. [{ts}] turn={turn_id} score={score:.3f}\n"
                f"   {text or '(empty)'}"
            )
        return "\n".join(lines)

    async def ensure_index_and_retrieve(
        self,
        *,
        conversation_id: str,
        user_email: str,
        turns: List[dict],
        query: str,
    ) -> Tuple[str, List[dict]]:
        threshold_exceeded = await self.ensure_index(
            conversation_id=conversation_id,
            user_email=user_email,
            turns=turns,
        )
        if not threshold_exceeded:
            return "(none)", []
        hits = await self.retrieve(
            conversation_id=conversation_id,
            user_email=user_email,
            query=query,
        )
        return self.render_hits_block(hits), hits
