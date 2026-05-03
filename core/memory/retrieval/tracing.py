"""Tracing helpers for dual-store retrieval provenance."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import networkx as nx
from networkx.readwrite import json_graph


@dataclass(slots=True)
class RetrievalTraceEvent:
    """Single retrieval trace event."""

    run_id: str
    parent_run_id: Optional[str]
    domain: str
    stage: str
    hop: int
    layer: int
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class RetrievalTraceSink(Protocol):
    """Trace sink contract."""

    def record(self, event: RetrievalTraceEvent) -> None:
        """Record one trace event."""


class NullRetrievalTraceSink:
    """No-op sink used as default to keep tracing disabled by default."""

    def record(self, event: RetrievalTraceEvent) -> None:
        _ = event


@dataclass(slots=True)
class PrefilterTraceContext:
    """Context passed into CompositePrefilter for optional tracing."""

    run_id: str
    parent_run_id: Optional[str]
    domain: str
    sink: RetrievalTraceSink
    hop: int = 0
    layer: int = 0


class NetworkXRetrievalTraceSink:
    """In-memory run-scoped retrieval trace sink backed by NetworkX MultiDiGraph."""

    def __init__(self, max_runs: int = 20) -> None:
        self._max_runs = max(1, int(max_runs))
        self._runs: OrderedDict[str, nx.MultiDiGraph] = OrderedDict()

    def record(self, event: RetrievalTraceEvent) -> None:
        graph = self._get_or_create_graph(event)
        graph.graph["updated_at"] = event.timestamp
        graph.graph["last_stage"] = event.stage
        graph.graph["events"] = int(graph.graph.get("events", 0)) + 1
        self._apply_event(graph, event)

    def list_runs(self) -> List[str]:
        return list(self._runs.keys())

    def get_run_graph(self, run_id: str) -> Optional[nx.MultiDiGraph]:
        graph = self._runs.get(run_id)
        if graph is None:
            return None
        self._runs.move_to_end(run_id)
        return graph

    def export_node_link_json(self, run_id: str, output_path: str) -> str:
        graph = self._export_graph(self._require_graph(run_id))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(graph, edges="links")
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(output)

    def export_graphml(self, run_id: str, output_path: str) -> str:
        graph = self._to_graphml_safe_graph(self._export_graph(self._require_graph(run_id)))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graph, output)
        return str(output)

    def export_html(self, run_id: str, output_path: str) -> str:
        graph = self._export_graph(self._require_graph(run_id))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        nodes = [self._html_node(node_id, attrs) for node_id, attrs in graph.nodes(data=True)]
        edges = [self._html_edge(source, target, attrs) for source, target, attrs in graph.edges(data=True)]
        chunks = self._html_chunks(graph)

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Retrieval Trace</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #f8fafc; color: #0f172a; font-family: Arial, sans-serif; }}
    #layout {{ width: 100%; height: 100%; display: flex; }}
    #network {{ flex: 1 1 auto; min-width: 0; border-right: 1px solid #d6dde8; }}
    #chunk-panel {{ width: 38%; min-width: 360px; max-width: 700px; padding: 12px; overflow-y: auto; background: #fff; }}
    #chunk-search {{ width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; padding: 8px; margin-bottom: 10px; }}
    .chunk-item {{ border: 1px solid #d6dde8; border-radius: 8px; padding: 10px; margin-bottom: 10px; background: #f8fafc; cursor: pointer; }}
    .chunk-item.active {{ border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.14); }}
    .chunk-id {{ font-weight: 700; margin-bottom: 4px; }}
    .chunk-raw {{ font-size: 12px; color: #475569; margin-bottom: 6px; word-break: break-word; }}
    .chunk-meta {{ font-size: 12px; color: #334155; margin-bottom: 6px; word-break: break-word; }}
    .chunk-text {{ font-size: 13px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div id="layout">
    <div id="network"></div>
    <aside id="chunk-panel">
      <input id="chunk-search" placeholder="Search chunk id/raw id/title/url/text" />
      <div id="chunk-list"></div>
    </aside>
  </div>
  <script>
    const nodeData = {json.dumps(nodes)};
    const edgeData = {json.dumps(edges)};
    const chunkData = {json.dumps(chunks)};

    const nodes = new vis.DataSet(nodeData);
    const edges = new vis.DataSet(edgeData);
    const network = new vis.Network(
      document.getElementById('network'),
      {{ nodes, edges }},
      {{
        physics: {{ enabled: true, solver: 'forceAtlas2Based' }},
        interaction: {{ hover: true, navigationButtons: true }},
        nodes: {{
          borderWidth: 1,
          size: 23,
          shape: 'circle',
          font: {{ color: '#0f172a', size: 14, multi: 'md' }}
        }}
      }}
    );

    const listEl = document.getElementById('chunk-list');
    const searchEl = document.getElementById('chunk-search');

    function renderChunks(activeNodeId = null) {{
      const query = (searchEl.value || '').toLowerCase().trim();
      listEl.innerHTML = '';
      for (const chunk of chunkData) {{
        const haystack = `${{chunk.id}} ${{chunk.raw_id}} ${{chunk.title}} ${{chunk.url}} ${{chunk.text}}`.toLowerCase();
        if (query && !haystack.includes(query)) continue;

        const card = document.createElement('div');
        card.className = 'chunk-item' + (activeNodeId === chunk.node_id ? ' active' : '');

        const id = document.createElement('div');
        id.className = 'chunk-id';
        id.textContent = chunk.id || '-';

        const raw = document.createElement('div');
        raw.className = 'chunk-raw';
        raw.textContent = `raw_id=${{chunk.raw_id || '-'}}`;

        const meta = document.createElement('div');
        meta.className = 'chunk-meta';
        meta.textContent = `title=${{chunk.title || '-'}} | url=${{chunk.url || '-'}}`;

        const text = document.createElement('div');
        text.className = 'chunk-text';
        text.textContent = chunk.text || '-';

        card.appendChild(id);
        card.appendChild(raw);
        card.appendChild(meta);
        card.appendChild(text);
        card.onclick = () => {{
          network.selectNodes([chunk.node_id]);
          network.focus(chunk.node_id, {{ scale: 1.05, animation: true }});
          renderChunks(chunk.node_id);
        }};
        listEl.appendChild(card);
      }}
    }}

    network.on('click', (params) => {{
      renderChunks(params.nodes.length ? params.nodes[0] : null);
    }});
    searchEl.addEventListener('input', () => {{
      renderChunks(network.getSelectedNodes()[0] || null);
    }});
    renderChunks();
  </script>
</body>
</html>
"""
        output.write_text(html, encoding="utf-8")
        return str(output)

    def _get_or_create_graph(self, event: RetrievalTraceEvent) -> nx.MultiDiGraph:
        graph = self._runs.get(event.run_id)
        if graph is not None:
            self._runs.move_to_end(event.run_id)
            return graph

        graph = nx.MultiDiGraph(
            run_id=event.run_id,
            parent_run_id=event.parent_run_id or "",
            domain=event.domain,
            created_at=event.timestamp,
            _chunk_index_map={},
            _next_chunk_index=1,
        )
        self._runs[event.run_id] = graph
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)
        return graph

    def _require_graph(self, run_id: str) -> nx.MultiDiGraph:
        graph = self.get_run_graph(run_id)
        if graph is None:
            raise ValueError(f"No retrieval trace run found: {run_id}")
        return graph

    @staticmethod
    def _is_chunk_hit_edge(attrs: Dict[str, Any]) -> bool:
        stage = str(attrs.get("stage") or "").strip()
        edge_type = str(attrs.get("edge_type") or "").strip()
        return (
            stage == "vector_seed"
            or edge_type == "vector_seed"
            or stage == "frontier_chunks"
            or edge_type == "graph_chunk_hit"
        )

    def _export_graph(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Derive an export-only subgraph containing only paths that lead to chunk hits.

        In-memory run graphs remain unfiltered for diagnostics.
        """
        kept_edges: set[tuple[str, str, Any]] = set()
        kept_nodes: set[str] = set()
        to_visit: list[str] = []

        for source, target, key, attrs in graph.edges(keys=True, data=True):
            if not self._is_chunk_hit_edge(attrs):
                continue
            target_type = str(graph.nodes.get(target, {}).get("node_type") or "")
            if target_type != "chunk":
                continue
            kept_edges.add((source, target, key))
            kept_nodes.add(source)
            kept_nodes.add(target)
            to_visit.append(source)
            to_visit.append(target)

        if not kept_edges:
            empty = nx.MultiDiGraph()
            empty.graph.update(graph.graph)
            return empty

        while to_visit:
            node_id = to_visit.pop()
            for source, _target, key, _attrs in graph.in_edges(
                node_id, keys=True, data=True
            ):
                edge_key = (source, node_id, key)
                if edge_key in kept_edges:
                    continue
                kept_edges.add(edge_key)
                if source not in kept_nodes:
                    kept_nodes.add(source)
                    to_visit.append(source)
                if node_id not in kept_nodes:
                    kept_nodes.add(node_id)

        pruned = nx.MultiDiGraph()
        pruned.graph.update(graph.graph)
        for node_id in kept_nodes:
            pruned.add_node(node_id, **dict(graph.nodes[node_id]))
        for source, target, key in kept_edges:
            pruned.add_edge(source, target, key=key, **dict(graph.edges[source, target, key]))
        return pruned

    @staticmethod
    def _sanitize_attr(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple, dict, set)):
            return json.dumps(value, sort_keys=True)
        return str(value)

    @classmethod
    def _to_graphml_safe_graph(cls, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        safe = nx.MultiDiGraph()
        safe.graph.update(
            {
                key: cls._sanitize_attr(value)
                for key, value in graph.graph.items()
                if not str(key).startswith("_")
            }
        )
        for node_id, attrs in graph.nodes(data=True):
            safe.add_node(node_id, **{k: cls._sanitize_attr(v) for k, v in attrs.items()})
        for source, target, key, attrs in graph.edges(keys=True, data=True):
            safe.add_edge(source, target, key=key, **{k: cls._sanitize_attr(v) for k, v in attrs.items()})
        return safe

    def _chunk_node(self, graph: nx.MultiDiGraph, raw_chunk_id: str) -> tuple[str, int, str]:
        index_map = graph.graph["_chunk_index_map"]
        if raw_chunk_id not in index_map:
            next_index = int(graph.graph["_next_chunk_index"])
            index_map[raw_chunk_id] = next_index
            graph.graph["_next_chunk_index"] = next_index + 1
        chunk_index = int(index_map[raw_chunk_id])
        display_id = f"C{chunk_index}"
        return f"chunk:{chunk_index}", chunk_index, display_id

    def _ensure_node(self, graph: nx.MultiDiGraph, node_id: str, **attrs: Any) -> None:
        if not graph.has_node(node_id):
            graph.add_node(node_id, **attrs)
            return
        current = graph.nodes[node_id]
        for key, value in attrs.items():
            if value in (None, ""):
                continue
            if key in ("iteration", "layer") and isinstance(value, int):
                prior = current.get(key)
                current[key] = min(prior, value) if isinstance(prior, int) else value
                continue
            current[key] = value

    def _add_edge(
        self,
        graph: nx.MultiDiGraph,
        source: str,
        target: str,
        *,
        event: RetrievalTraceEvent,
        edge_type: str,
        score: Any = None,
        selected: Any = None,
    ) -> None:
        graph.add_edge(
            source,
            target,
            key=f"{event.stage}:{source}:{target}:{event.timestamp}",
            run_id=event.run_id,
            parent_run_id=event.parent_run_id or "",
            domain=event.domain,
            stage=event.stage,
            edge_type=edge_type,
            hop=event.hop,
            layer=event.layer,
            score=score,
            selected=selected,
            timestamp=event.timestamp,
        )

    def _ensure_chunk_node(
        self,
        graph: nx.MultiDiGraph,
        *,
        raw_chunk_id: str,
        domain: str,
        layer: int,
        iteration: int,
        source: str = "",
        chunk_text: str = "",
        article_title: str = "",
        source_url: str = "",
        **attrs: Any,
    ) -> str:
        node_id, chunk_index, display_id = self._chunk_node(graph, raw_chunk_id)
        self._ensure_node(
            graph,
            node_id,
            node_type="chunk",
            domain=domain,
            layer=layer,
            iteration=iteration,
            label=display_id,
            display_id=display_id,
            raw_chunk_id=raw_chunk_id,
            chunk_index=chunk_index,
            source=source,
            chunk_text=chunk_text,
            article_title=article_title,
            source_url=source_url,
            **attrs,
        )
        return node_id

    def _apply_event(self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent) -> None:
        if event.stage == "vector_seed":
            self._apply_vector_seed(graph, event)
            return
        if event.stage == "seed_entities":
            self._apply_seed_entities(graph, event)
            return
        if event.stage == "neighbor_expansion":
            self._apply_neighbor_expansion(graph, event)
            return
        if event.stage == "frontier_chunks":
            self._apply_frontier_chunks(graph, event)
            return
        if event.stage == "prefilter_output":
            self._apply_prefilter_output(graph, event)

    def _apply_vector_seed(self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent) -> None:
        query_node_id = f"query:{event.run_id}"
        query_text = str(event.payload.get("query") or "")
        self._ensure_node(
            graph,
            query_node_id,
            node_type="query",
            domain=event.domain,
            layer=event.layer,
            iteration=event.hop,
            label="Query",
            display_label=query_text[:80] or f"query:{event.domain}",
        )

        for item in event.payload.get("chunks", []):
            raw_chunk_id = str(item.get("chunk_id") or "").strip()
            if not raw_chunk_id:
                continue
            chunk_node_id = self._ensure_chunk_node(
                graph,
                raw_chunk_id=raw_chunk_id,
                domain=event.domain,
                layer=event.layer,
                iteration=event.hop,
                source="vector",
                chunk_text=str(item.get("chunk_text") or item.get("text") or ""),
                article_title=str(item.get("article_title") or item.get("title") or ""),
                source_url=str(item.get("source_url") or item.get("url") or ""),
            )
            self._add_edge(
                graph,
                query_node_id,
                chunk_node_id,
                event=event,
                edge_type="vector_seed",
                score=item.get("score"),
            )

    def _apply_seed_entities(self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent) -> None:
        for item in event.payload.get("links", []):
            raw_chunk_id = str(item.get("source_chunk_id") or item.get("chunk_id") or "").strip()
            entity_id = str(item.get("entity_id") or "").strip()
            if not raw_chunk_id or not entity_id:
                continue

            chunk_node_id = self._ensure_chunk_node(
                graph,
                raw_chunk_id=raw_chunk_id,
                domain=event.domain,
                layer=event.layer,
                iteration=event.hop,
            )
            entity_node_id = f"entity:{entity_id}"
            entity_name = str(item.get("entity_name") or entity_id)
            self._ensure_node(
                graph,
                entity_node_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
                iteration=event.hop,
                label=entity_name,
                display_label=entity_name,
                entity_type=str(item.get("entity_type") or ""),
            )
            self._add_edge(
                graph,
                chunk_node_id,
                entity_node_id,
                event=event,
                edge_type="seed_mentions",
            )

    def _apply_neighbor_expansion(self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent) -> None:
        for item in event.payload.get("candidates", []):
            source_entity_id = str(item.get("source_entity_id") or "").strip()
            neighbor_entity_id = str(item.get("neighbor_entity_id") or "").strip()
            if not source_entity_id or not neighbor_entity_id:
                continue

            source_node_id = f"entity:{source_entity_id}"
            neighbor_node_id = f"entity:{neighbor_entity_id}"
            source_name = str(item.get("source_entity_name") or source_entity_id)
            neighbor_name = str(item.get("neighbor_name") or neighbor_entity_id)

            self._ensure_node(
                graph,
                source_node_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
                iteration=event.hop,
                label=source_name,
                display_label=source_name,
            )
            self._ensure_node(
                graph,
                neighbor_node_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
                iteration=event.hop,
                label=neighbor_name,
                display_label=neighbor_name,
                entity_type=str(item.get("neighbor_type") or ""),
            )
            self._add_edge(
                graph,
                source_node_id,
                neighbor_node_id,
                event=event,
                edge_type=str(item.get("relationship_type") or "RELATED_TO"),
                score=item.get("score"),
                selected=bool(item.get("selected", False)),
            )

    def _apply_frontier_chunks(self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent) -> None:
        chunk_meta = {
            str(item.get("chunk_id") or "").strip(): item
            for item in event.payload.get("chunks", [])
            if str(item.get("chunk_id") or "").strip()
        }
        for item in event.payload.get("links", []):
            supporting_entity_id = str(item.get("supporting_entity_id") or "").strip()
            supporting_entity_name = str(
                item.get("supporting_entity_name") or supporting_entity_id
            )
            raw_chunk_id = str(item.get("chunk_id") or "").strip()
            if not supporting_entity_id or not raw_chunk_id:
                continue

            entity_node_id = f"entity:{supporting_entity_id}"
            self._ensure_node(
                graph,
                entity_node_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
                iteration=event.hop,
                label=supporting_entity_name,
                display_label=supporting_entity_name,
            )

            meta = chunk_meta.get(raw_chunk_id, {})
            chunk_node_id = self._ensure_chunk_node(
                graph,
                raw_chunk_id=raw_chunk_id,
                domain=event.domain,
                layer=event.layer,
                iteration=event.hop,
                source="graph",
                chunk_text=str(meta.get("chunk_text") or meta.get("text") or ""),
                article_title=str(meta.get("article_title") or meta.get("title") or ""),
                source_url=str(meta.get("source_url") or meta.get("url") or ""),
            )
            self._add_edge(
                graph,
                entity_node_id,
                chunk_node_id,
                event=event,
                edge_type="graph_chunk_hit",
            )

    def _apply_prefilter_output(self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent) -> None:
        for item in event.payload.get("ranked_chunks", []):
            raw_chunk_id = str(item.get("chunk_id") or "").strip()
            if not raw_chunk_id:
                continue

            self._ensure_chunk_node(
                graph,
                raw_chunk_id=raw_chunk_id,
                domain=str(item.get("domain") or event.domain),
                layer=event.layer,
                iteration=event.hop,
                source=str(item.get("source") or ""),
                chunk_text=str(item.get("chunk_text") or item.get("text") or ""),
                article_title=str(item.get("article_title") or item.get("title") or ""),
                source_url=str(item.get("source_url") or item.get("url") or ""),
                graph_depth=item.get("graph_depth"),
                embedding_score=item.get("embedding_score"),
                composite_score=item.get("composite_score"),
                prefilter_rank=item.get("rank"),
                prefilter_selected=bool(item.get("selected", False)),
            )

    def _html_node(self, node_id: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        node_type = str(attrs.get("node_type") or "generic")
        domain = str(attrs.get("domain") or "unknown")
        layer = int(attrs.get("layer") or 0)
        iteration = int(attrs.get("iteration") or 0)
        display_label = str(
            attrs.get("display_id")
            or attrs.get("display_label")
            or attrs.get("label")
            or node_id
        )
        if len(display_label) > 18:
            display_label = display_label[:16] + ".."

        title = [
            f"id={node_id}",
            f"label={attrs.get('display_id') or attrs.get('display_label') or attrs.get('label') or node_id}",
            f"type={node_type}",
            f"domain={domain}",
            f"layer={layer}",
            f"iteration={iteration}",
        ]
        if node_type == "chunk":
            title.extend(
                [
                    f"raw_chunk_id={attrs.get('raw_chunk_id') or ''}",
                    f"article_title={attrs.get('article_title') or ''}",
                    f"source_url={attrs.get('source_url') or ''}",
                ]
            )

        node_label = display_label
        if node_type != "entity":
            node_label = f"{display_label}\nI{iteration}"

        return {
            "id": str(node_id),
            "label": node_label,
            "title": "<br>".join(title),
            "group": f"{domain}:L{layer}:{node_type}",
            "size": 23,
        }

    def _html_edge(self, source: str, target: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "from": str(source),
            "to": str(target),
            "label": str(attrs.get("edge_type") or attrs.get("stage") or ""),
            "title": "<br>".join(
                [
                    f"stage={attrs.get('stage', '')}",
                    f"hop={attrs.get('hop', '')}",
                    f"layer={attrs.get('layer', '')}",
                    f"score={attrs.get('score', '')}",
                ]
            ),
            "arrows": "to",
        }

    def _html_chunks(self, graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        payload = []
        for node_id, attrs in graph.nodes(data=True):
            if str(attrs.get("node_type") or "") != "chunk":
                continue
            payload.append(
                {
                    "id": str(attrs.get("display_id") or ""),
                    "raw_id": str(attrs.get("raw_chunk_id") or ""),
                    "node_id": str(node_id),
                    "text": str(attrs.get("chunk_text") or ""),
                    "title": str(attrs.get("article_title") or ""),
                    "url": str(attrs.get("source_url") or ""),
                    "index": int(attrs.get("chunk_index") or 0),
                }
            )
        payload.sort(key=lambda item: item["index"])
        return payload
