"""Async Redis-backed storage for in-memory subgraphs."""

from __future__ import annotations

import json
from typing import Optional

import networkx as nx
import redis.asyncio as redis
from networkx.readwrite import json_graph


class SubgraphStore:
    def __init__(self, redis_url: str, ttl: int) -> None:
        self._redis_url = redis_url
        self._ttl = ttl
        self._client: Optional[redis.Redis] = None

    async def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self._redis_url)
        return self._client

    async def save(self, key: str, graph: nx.DiGraph) -> str:
        if not key:
            raise ValueError("SubgraphStore.save requires a non-empty key.")
        client = await self._get_client()
        payload = json.dumps(json_graph.node_link_data(graph))
        await client.set(key, payload, ex=self._ttl)
        return key

    async def load(self, key: str) -> Optional[nx.DiGraph]:
        if not key:
            return None
        client = await self._get_client()
        payload = await client.get(key)
        if payload is None:
            return None
        payload_text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
        data = json.loads(payload_text)
        return json_graph.node_link_graph(data, directed=True)

    async def delete(self, key: str) -> None:
        if not key:
            return
        client = await self._get_client()
        await client.delete(key)

    @staticmethod
    def make_key(agent_name: str, conversation_id: str) -> str:
        return f"subgraph:{agent_name}:{conversation_id}"



