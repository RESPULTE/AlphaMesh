"""
FinancialMemory — the primary public interface for the memory module.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from lightrag.lightrag import LightRAG

from core.memory.config import memory_config
from core.memory.cross_namespace_linker import CrossNamespaceLinker
from core.memory.extraction_prompts import build_user_extraction_prompt
from core.memory.lightrag_manager import LightRAGManager
from core.memory.models import (
    IngestionResult,
    IngestionStatus,
    QueryResult,
)

logger = logging.getLogger(__name__)


class FinancialMemory:
    """
    Dual-namespace financial memory system.
    Manages a global knowledge graph and per-user private graphs.
    """

    def __init__(self):
        self._manager = LightRAGManager()
        self._linker = CrossNamespaceLinker()
        self._initialized = False

    async def initialize(self):
        """Eagerly initialize the global LightRAG instance."""
        if self._initialized:
            return
        await self._manager.get_global()
        self._initialized = True
        logger.info("FinancialMemory initialized")

    # ──────────────────────────────────────────────────────────────────
    # Ingestion — Conversations
    # ──────────────────────────────────────────────────────────────────

    async def ingest_conversation(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        date_range: Optional[Tuple[str, str]] = None,
    ) -> IngestionResult:
        """Ingest a conversation with date-period tagging."""
        formatted_text = self._format_conversation(messages, date_range)
        return await self._dual_namespace_ingest(
            user_id=user_id,
            text=formatted_text,
            content_type="conversation",
        )

    # ──────────────────────────────────────────────────────────────────
    # Ingestion — Documents
    # ──────────────────────────────────────────────────────────────────

    async def ingest_document(
        self,
        user_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """Ingest a document with metadata tagging."""
        formatted_text = self._format_document(text, metadata)
        return await self._dual_namespace_ingest(
            user_id=user_id,
            text=formatted_text,
            content_type="document",
        )

    # ──────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────

    async def query(
        self,
        user_id: str,
        query: str,
        mode: str = "hybrid",
    ) -> QueryResult:
        """Query both namespaces and merge results."""
        global_rag = await self._manager.get_global()
        user_rag = await self._manager.get_user(user_id)

        # Query both namespaces concurrently
        async with asyncio.TaskGroup() as tg:
            global_task = tg.create_task(self._safe_query(global_rag, query, mode, "global"))
            user_task = tg.create_task(self._safe_query(user_rag, query, mode, "user"))

        global_context = global_task.result()
        user_context = user_task.result()

        # Resolve cross-namespace references
        cross_refs = []
        try:
            cross_refs = await self._linker.resolve_cross_refs(user_id, user_rag)
        except Exception as exc:
            logger.error(f"Cross-ref resolution failed: {exc}")

        # Merge results
        merged = self._merge_contexts(global_context, user_context, cross_refs)

        return QueryResult(
            user_id=user_id,
            query=query,
            mode=mode,
            global_context=global_context,
            user_context=user_context,
            merged_context=merged,
            cross_references=cross_refs,
        )

    # ──────────────────────────────────────────────────────────────────
    # Status & Administration
    # ──────────────────────────────────────────────────────────────────

    async def get_ingestion_status(
        self, track_id: str, namespace: str = "global"
    ) -> IngestionStatus:
        """Check the status of a background ingestion job."""
        try:
            if namespace == "global":
                rag = await self._manager.get_global()
            else:
                user_id = namespace.removeprefix("user_")
                rag = await self._manager.get_user(user_id)

            pipelines = rag.doc_status
            docs = await pipelines.get_docs_by_track_id(track_id)
            
            if not docs:
                return IngestionStatus(track_id=track_id, namespace=namespace, status="unknown")

            # Aggregate status: if any failed, it's failed; if all processed, it's processed
            all_statuses = [d.status for d in docs.values()]
            if any(s == "failed" for s in all_statuses):
                error_msg = next((d.error_msg for d in docs.values() if d.status == "failed"), "Unknown error")
                return IngestionStatus(track_id=track_id, namespace=namespace, status="failed", error=error_msg)
            
            if all(s == "processed" for s in all_statuses):
                return IngestionStatus(track_id=track_id, namespace=namespace, status="processed")
            
            return IngestionStatus(track_id=track_id, namespace=namespace, status="processing")
        except Exception as exc:
            return IngestionStatus(
                track_id=track_id,
                namespace=namespace,
                status="error",
                error=str(exc),
            )

    async def delete_user_data(self, user_id: str):
        """Purge all data for a specific user."""
        user_workspace = f"user_{user_id}"
        try:
            user_rag = await self._manager.get_user(user_id)
            driver = user_rag.chunk_entity_relation_graph._driver
            async with driver.session(database=memory_config.neo4j_database) as session:
                # Delete cross-namespace edges pointing FROM this user
                await session.run(
                    f"MATCH (n:`{user_workspace}`)-[r:CROSS_REF]->() DELETE r"
                )
                # Delete user nodes
                await session.run(
                    f"MATCH (n:`{user_workspace}`) DETACH DELETE n"
                )
            logger.info(f"All data deleted for user '{user_id}'")
        except Exception as exc:
            logger.error(f"Error deleting user data for '{user_id}': {exc}")
            raise

    async def close(self):
        """Gracefully shutdown all connections."""
        await self._manager.close_all()
        # self._linker.close() # Linker no longer manages state
        self._initialized = False
        logger.info("FinancialMemory shut down")

    # ──────────────────────────────────────────────────────────────────
    # Internal — Dual-namespace ingestion pipeline
    # ──────────────────────────────────────────────────────────────────

    async def _dual_namespace_ingest(
        self,
        user_id: str,
        text: str,
        content_type: str,
    ) -> IngestionResult:
        """
        1. Classify text -> global/user
        2. Ingest global
        3. Link user stubs -> global entities
        """
        global_rag = await self._manager.get_global()
        user_rag = await self._manager.get_user(user_id)

        # Step 1: LLM Classification
        global_text, _ = await self._classify_content(text)

        logger.info(f"Global text: {global_text}")

        result = IngestionResult(user_id=user_id, content_type=content_type)

        # Step 2: Global ingestion
        if global_text.strip():
            try:
                track_id = await global_rag.ainsert(global_text)
                result.global_track_id = track_id
            except Exception as exc:
                logger.error(f"Global ingestion fail: {exc}")
                result.message += f"Global error: {exc}. "

        # Step 3: Global entity extraction for user prompt context
        global_entities = await self._get_workspace_entities(global_rag)
        logger.info(f"Global entities: {global_entities}")
        # Step 4: User ingestion with global context
        if text.strip():
            try:
                # Update extraction prompt with latest global entities
                user_rag.addon_params["extract_prompt"] = build_user_extraction_prompt(global_entities)
                
                track_id = await user_rag.ainsert(text)
                result.user_track_id = track_id
            except Exception as exc:
                logger.error(f"User ingestion fail: {exc}")
                result.message += f"User error: {exc}. "

        # Step 5: Post-process linking
        try:
            link_stats = await self._linker.link_and_cleanup(user_rag, global_rag, user_id)
            result.message += f"Linked {link_stats['links_created']} items. "
        except Exception as exc:
            logger.error(f"Linking fail: {exc}")
            result.message += f"Linking error: {exc}. "

        result.status = "completed"
        return result

    # ──────────────────────────────────────────────────────────────────
    # Internal — Helpers
    # ──────────────────────────────────────────────────────────────────

    async def _classify_content(self, text: str) -> Tuple[str, str]:
        """Classify text into global-safe and user-specific parts."""
        try:
            global_rag = await self._manager.get_global()
            prompt = f"""Separate the following text into:
1. GLOBAL: Financial domain knowledge (concepts, markets, companies). NO PII.
2. USER: Personal info (goals, holdings, preferences, identity).

Format:
===GLOBAL===
[global content]
===USER===
[user content]

Text:
{text}
"""
            llm_out = await global_rag.llm_model_func(prompt, system_prompt="Precise classifier.")
            
            if "===GLOBAL===" in llm_out and "===USER===" in llm_out:
                parts = llm_out.split("===USER===")
                g_text = parts[0].replace("===GLOBAL===", "").strip()
                u_text = parts[1].strip() if len(parts) > 1 else ""
                return g_text, u_text
        except Exception as exc:
            logger.warning(f"Classification failed: {exc}")
        return text, text

    async def _get_workspace_entities(self, rag: LightRAG) -> List[str]:
        """Fetch entity names from Neo4j for prompt context."""
        try:
            # Simple approximation: get names from memory graph if possible
            # For now, we'll try to get all labels from the graph
            return await rag.chunk_entity_relation_graph.get_all_labels()
        except:
            return []

    async def _safe_query(self, rag: LightRAG, query: str, mode: str, namespace: str) -> str:
        try:
            from lightrag.base import QueryParam
            params = QueryParam(mode=mode, stream=False)
            result = await rag.aquery(query, param=params)
            
            if isinstance(result, str):
                return result
            
            # Handle AsyncIterator if it somehow leaked through
            if hasattr(result, "__aiter__"):
                content = []
                async for chunk in result:
                    content.append(chunk)
                return "".join(content)
            
            return str(result)
        except Exception as exc:
            logger.error(f"Query {namespace} fail: {exc}")
            return ""

    def _merge_contexts(self, g_ctx: str, u_ctx: str, refs: List[Dict[str, Any]]) -> str:
        merged = []
        if g_ctx: merged.append(f"Domain Knowledge:\n{g_ctx}")
        if u_ctx: merged.append(f"Personal Context:\n{u_ctx}")
        if refs:
            ref_str = "\n".join([f"- {r['source_entity']} refers to {r['global_entity']} ({r['global_type']})" for r in refs])
            merged.append(f"Cross-References:\n{ref_str}")
        return "\n\n".join(merged)

    def _format_conversation(self, messages: List[Dict[str, str]], dates: Optional[Tuple[str, str]]) -> str:
        header = f"[Period: {dates[0]} to {dates[1]}]" if dates else f"[Date: {datetime.now().isoformat()}]"
        body = "\n".join([f"{m.get('role', 'System').capitalize()}: {m.get('content', '')}" for m in messages])
        return f"{header}\n\n{body}"

    def _format_document(self, text: str, meta: Optional[Dict[str, Any]]) -> str:
        header = f"[Metadata: {meta}]" if meta else ""
        return f"{header}\n\n{text}"
