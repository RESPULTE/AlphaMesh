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
    """
    In-memory run-scoped retrieval trace sink backed by NetworkX MultiDiGraph.

    Each run gets one graph with bounded LRU retention.
    """

    def __init__(self, max_runs: int = 20) -> None:
        if nx is None:
            raise ImportError(
                "networkx is required for NetworkXRetrievalTraceSink. "
                "Install with: pip install networkx"
            )
        self._max_runs = max(1, int(max_runs))
        self._runs: OrderedDict[str, nx.MultiDiGraph] = OrderedDict()

    def record(self, event: RetrievalTraceEvent) -> None:
        graph = self._get_or_create_graph(event)
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
        if json_graph is None:
            raise ImportError(
                "networkx json_graph module is unavailable. "
                "Install/upgrade networkx."
            )
        graph = self._require_graph(run_id)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = json_graph.node_link_data(graph)
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return str(output)

    def export_graphml(self, run_id: str, output_path: str) -> str:
        graph = self._require_graph(run_id)
        graphml_graph = self._to_graphml_safe_graph(graph)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(graphml_graph, output)
        return str(output)

    def export_html(self, run_id: str, output_path: str) -> str:
        """
        Export an interactive HTML graph.

        Prefers pyvis; falls back to a lightweight vis-network HTML template
        when pyvis is unavailable.
        """
        graph = self._require_graph(run_id)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._export_html_pyvis(graph, output)
        except Exception:
            self._export_html_vis_network(graph, output)
        return str(output)

    # -- Internal ------------------------------------------------------------

    def _get_or_create_graph(self, event: RetrievalTraceEvent) -> nx.MultiDiGraph:
        existing = self._runs.get(event.run_id)
        if existing is not None:
            self._runs.move_to_end(event.run_id)
            return existing

        graph = nx.MultiDiGraph()
        graph.graph["run_id"] = event.run_id
        graph.graph["parent_run_id"] = event.parent_run_id or ""
        graph.graph["domain"] = event.domain
        graph.graph["created_at"] = event.timestamp
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
    def _sanitize_attr_value(value: Any) -> Any:
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
            {k: cls._sanitize_attr_value(v) for k, v in graph.graph.items()}
        )

        for node_id, attrs in graph.nodes(data=True):
            safe.add_node(
                node_id,
                **{k: cls._sanitize_attr_value(v) for k, v in attrs.items()},
            )

        for source, target, key, attrs in graph.edges(keys=True, data=True):
            safe.add_edge(
                source,
                target,
                key=key,
                **{k: cls._sanitize_attr_value(v) for k, v in attrs.items()},
            )

        return safe

    def _ensure_node(
        self,
        graph: nx.MultiDiGraph,
        node_id: str,
        *,
        label: Optional[str] = None,
        node_type: Optional[str] = None,
        domain: Optional[str] = None,
        layer: Optional[int] = None,
        **attrs: Any,
    ) -> None:
        payload = {
            "label": label or node_id,
            "node_type": node_type or "generic",
            "domain": domain or graph.graph.get("domain", ""),
        }
        if layer is not None:
            payload["layer"] = layer
        payload.update(attrs)
        if graph.has_node(node_id):
            graph.nodes[node_id].update(payload)
        else:
            graph.add_node(node_id, **payload)

    def _apply_event(self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent) -> None:
        graph.graph["updated_at"] = event.timestamp
        graph.graph["last_stage"] = event.stage
        graph.graph.setdefault("events", 0)
        graph.graph["events"] += 1

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
            return

    def _apply_vector_seed(
        self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent
    ) -> None:
        query_node_id = f"query:{event.run_id}"
        query_text = str(event.payload.get("query") or "")
        self._ensure_node(
            graph,
            query_node_id,
            label=query_text[:80] or f"query:{event.domain}",
            node_type="query",
            domain=event.domain,
            layer=event.layer,
        )

        for item in event.payload.get("chunks", []):
            chunk_id = str(item.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            chunk_node_id = f"chunk:{chunk_id}"
            self._ensure_node(
                graph,
                chunk_node_id,
                label=chunk_id,
                node_type="chunk",
                domain=event.domain,
                layer=event.layer,
                source="vector",
            )
            graph.add_edge(
                query_node_id,
                chunk_node_id,
                key=f"{event.stage}:{chunk_id}:{event.timestamp}",
                run_id=event.run_id,
                parent_run_id=event.parent_run_id or "",
                domain=event.domain,
                stage=event.stage,
                edge_type="vector_seed",
                hop=event.hop,
                layer=event.layer,
                score=item.get("score"),
                timestamp=event.timestamp,
            )

    def _apply_seed_entities(
        self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent
    ) -> None:
        for item in event.payload.get("links", []):
            chunk_id = str(item.get("source_chunk_id") or item.get("chunk_id") or "")
            entity_id = str(item.get("entity_id") or "")
            if not chunk_id or not entity_id:
                continue

            chunk_node_id = f"chunk:{chunk_id}"
            entity_node_id = f"entity:{entity_id}"

            self._ensure_node(
                graph,
                chunk_node_id,
                label=chunk_id,
                node_type="chunk",
                domain=event.domain,
                layer=event.layer,
            )
            self._ensure_node(
                graph,
                entity_node_id,
                label=item.get("entity_name") or entity_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
                entity_type=item.get("entity_type") or "",
            )
            graph.add_edge(
                chunk_node_id,
                entity_node_id,
                key=f"{event.stage}:{chunk_id}:{entity_id}:{event.timestamp}",
                run_id=event.run_id,
                parent_run_id=event.parent_run_id or "",
                domain=event.domain,
                stage=event.stage,
                edge_type="seed_mentions",
                hop=event.hop,
                layer=event.layer,
                timestamp=event.timestamp,
            )

    def _apply_neighbor_expansion(
        self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent
    ) -> None:
        for item in event.payload.get("candidates", []):
            source_entity_id = str(item.get("source_entity_id") or "")
            neighbor_entity_id = str(item.get("neighbor_entity_id") or "")
            if not source_entity_id or not neighbor_entity_id:
                continue

            source_node_id = f"entity:{source_entity_id}"
            neighbor_node_id = f"entity:{neighbor_entity_id}"
            self._ensure_node(
                graph,
                source_node_id,
                label=source_entity_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
            )
            self._ensure_node(
                graph,
                neighbor_node_id,
                label=item.get("neighbor_name") or neighbor_entity_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
                entity_type=item.get("neighbor_type") or "",
            )
            graph.add_edge(
                source_node_id,
                neighbor_node_id,
                key=f"{event.stage}:{source_entity_id}:{neighbor_entity_id}:{event.timestamp}",
                run_id=event.run_id,
                parent_run_id=event.parent_run_id or "",
                domain=event.domain,
                stage=event.stage,
                edge_type=item.get("relationship_type") or "RELATED_TO",
                hop=event.hop,
                layer=event.layer,
                score=item.get("score"),
                selected=bool(item.get("selected", False)),
                timestamp=event.timestamp,
            )

    def _apply_frontier_chunks(
        self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent
    ) -> None:
        for item in event.payload.get("links", []):
            supporting_entity_id = str(item.get("supporting_entity_id") or "")
            chunk_id = str(item.get("chunk_id") or "")
            if not supporting_entity_id or not chunk_id:
                continue
            entity_node_id = f"entity:{supporting_entity_id}"
            chunk_node_id = f"chunk:{chunk_id}"

            self._ensure_node(
                graph,
                entity_node_id,
                label=supporting_entity_id,
                node_type="entity",
                domain=event.domain,
                layer=event.layer,
            )
            self._ensure_node(
                graph,
                chunk_node_id,
                label=chunk_id,
                node_type="chunk",
                domain=event.domain,
                layer=event.layer,
                source="graph",
            )
            graph.add_edge(
                entity_node_id,
                chunk_node_id,
                key=f"{event.stage}:{supporting_entity_id}:{chunk_id}:{event.timestamp}",
                run_id=event.run_id,
                parent_run_id=event.parent_run_id or "",
                domain=event.domain,
                stage=event.stage,
                edge_type="graph_chunk_hit",
                hop=event.hop,
                layer=event.layer,
                timestamp=event.timestamp,
            )

    def _apply_prefilter_output(
        self, graph: nx.MultiDiGraph, event: RetrievalTraceEvent
    ) -> None:
        for item in event.payload.get("ranked_chunks", []):
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            chunk_node_id = f"chunk:{chunk_id}"
            self._ensure_node(
                graph,
                chunk_node_id,
                label=chunk_id,
                node_type="chunk",
                domain=item.get("domain") or event.domain,
                layer=event.layer,
                source=item.get("source") or "",
                graph_depth=item.get("graph_depth"),
                embedding_score=item.get("embedding_score"),
                composite_score=item.get("composite_score"),
                prefilter_rank=item.get("rank"),
                prefilter_selected=bool(item.get("selected", False)),
            )

    def _export_html_pyvis(self, graph: nx.MultiDiGraph, output: Path) -> None:
        from pyvis.network import Network

        net = Network(height="850px", width="100%", directed=True)
        for node_id, attrs in graph.nodes(data=True):
            label = str(attrs.get("label") or node_id)
            node_type = str(attrs.get("node_type") or "generic")
            layer = int(attrs.get("layer", 0))
            domain = str(attrs.get("domain") or "unknown")
            group = f"{domain}:L{layer}:{node_type}"
            title = (
                f"id={node_id}<br>"
                f"type={node_type}<br>"
                f"domain={domain}<br>"
                f"layer={layer}"
            )
            net.add_node(
                str(node_id),
                label=label,
                title=title,
                group=group,
            )

        for source, target, attrs in graph.edges(data=True):
            edge_label = str(attrs.get("edge_type") or attrs.get("stage") or "")
            title = (
                f"stage={attrs.get('stage', '')}<br>"
                f"hop={attrs.get('hop', '')}<br>"
                f"layer={attrs.get('layer', '')}<br>"
                f"score={attrs.get('score', '')}"
            )
            net.add_edge(
                str(source),
                str(target),
                label=edge_label,
                title=title,
                arrows="to",
            )

        net.set_options(
            """
            {
              "physics": {"enabled": true, "solver": "forceAtlas2Based"},
              "interaction": {"hover": true, "navigationButtons": true}
            }
            """
        )
        net.save_graph(str(output))

    def _export_html_vis_network(self, graph: nx.MultiDiGraph, output: Path) -> None:
        node_payload = []
        for node_id, attrs in graph.nodes(data=True):
            node_payload.append(
                {
                    "id": str(node_id),
                    "label": str(attrs.get("label") or node_id),
                    "group": f"{attrs.get('domain', 'unknown')}:L{attrs.get('layer', 0)}",
                    "title": json.dumps(
                        {
                            "node_type": attrs.get("node_type"),
                            "domain": attrs.get("domain"),
                            "layer": attrs.get("layer"),
                        }
                    ),
                }
            )

        edge_payload = []
        for source, target, attrs in graph.edges(data=True):
            edge_payload.append(
                {
                    "from": str(source),
                    "to": str(target),
                    "label": str(attrs.get("edge_type") or attrs.get("stage") or ""),
                    "title": json.dumps(
                        {
                            "stage": attrs.get("stage"),
                            "hop": attrs.get("hop"),
                            "layer": attrs.get("layer"),
                            "score": attrs.get("score"),
                        }
                    ),
                    "arrows": "to",
                }
            )

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Retrieval Trace</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body, #network {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      background: #f8fafc;
    }}
  </style>
</head>
<body>
  <div id="network"></div>
  <script>
    const nodes = new vis.DataSet({json.dumps(node_payload)});
    const edges = new vis.DataSet({json.dumps(edge_payload)});
    const container = document.getElementById('network');
    const data = {{ nodes, edges }};
    const options = {{
      physics: {{ enabled: true, solver: 'forceAtlas2Based' }},
      interaction: {{ hover: true, navigationButtons: true }}
    }};
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
        output.write_text(html, encoding="utf-8")
