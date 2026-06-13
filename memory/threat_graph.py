"""
memory/threat_graph.py
─────────────────────────────────────────────────────────────────────────────
Threat Memory Graph — NetworkX-backed relational memory for cross-session
technique outcomes, defense mechanisms, and planning retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from core.paths import TLTM_VECTORS_DIR
from intelligence.defense_fingerprinter import infer_defense_mechanisms

logger = logging.getLogger(__name__)

GRAPH_DIR = TLTM_VECTORS_DIR.parent / "threat_graphs"

NODE_TARGET = "Target"
NODE_TECHNIQUE = "Technique"
NODE_VULNERABILITY = "Vulnerability"
NODE_PERSONA = "Persona"
NODE_REFUSAL_PATTERN = "RefusalPattern"
NODE_SUCCESS_STRATEGY = "SuccessfulStrategy"
NODE_DEFENSE_MECHANISM = "DefenseMechanism"

EDGE_SUCCESS_ON = "SUCCESS_ON"
EDGE_FAILED_ON = "FAILED_ON"
EDGE_BYPASSED_BY = "BYPASSED_BY"
EDGE_BLOCKED_BY = "BLOCKED_BY"
EDGE_SIMILAR_TO = "SIMILAR_TO"
EDGE_DERIVED_FROM = "DERIVED_FROM"
EDGE_PATCHED_BY = "PATCHED_BY"
EDGE_DEFENDED_BY = "DEFENDED_BY"

_BETA_ALPHA = 1.0
_BETA_BETA = 1.0
_MIN_OBS_FOR_HIGH_CONF = 3
# Minimum Jaccard similarity for a target to be considered "similar"
_SIMILAR_TARGET_THRESHOLD = 0.25


def _sanitize_id(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")[:128]


def _node_id(node_type: str, key: str) -> str:
    return f"{node_type}:{_sanitize_id(key)}"


@dataclass
class GraphContext:
    """Retrieval bundle for RAG attack planning."""
    successful_strategies: list[dict[str, Any]] = field(default_factory=list)
    failed_strategies: list[dict[str, Any]] = field(default_factory=list)
    similar_targets: list[str] = field(default_factory=list)
    defense_mechanisms: list[str] = field(default_factory=list)
    technique_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    observation_count: int = 0


class ThreatMemoryGraph:
    """Per-target NetworkX multigraph persisted as node-link JSON."""

    def __init__(self, target_model_id: str) -> None:
        self.target_model_id = target_model_id
        self._graph = nx.MultiDiGraph()
        self._path = GRAPH_DIR / f"{_sanitize_id(target_model_id)}.json"
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            loaded = nx.node_link_graph(data, directed=True, multigraph=True)
            if isinstance(loaded, nx.MultiDiGraph):
                self._graph = loaded
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ThreatGraph] Load failed for %s: %s", self.target_model_id, exc)
            backup = self._path.with_suffix(".json.bak")
            if self._path.exists():
                try:
                    self._path.rename(backup)
                except OSError:
                    pass

    def save(self) -> bool:
        try:
            GRAPH_DIR.mkdir(parents=True, exist_ok=True)
            data = nx.node_link_data(self._graph)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ThreatGraph] Save failed for %s: %s", self.target_model_id, exc)
            return False

    def _ensure_node(self, node_type: str, key: str, **attrs: Any) -> str:
        nid = _node_id(node_type, key)
        if nid not in self._graph:
            self._graph.add_node(nid, node_type=node_type, key=key, **attrs)
        else:
            for k, v in attrs.items():
                if v is not None:
                    self._graph.nodes[nid][k] = v
        return nid

    def _add_edge(self, src: str, dst: str, edge_type: str, **attrs: Any) -> None:
        self._graph.add_edge(src, dst, key=edge_type, edge_type=edge_type, **attrs)

    def upsert_attempt(
        self,
        technique: str,
        outcome: str,
        *,
        pap_technique: str = "",
        score: float = 0.0,
        fingerprint: dict | None = None,
        response_text: str = "",
    ) -> None:
        """Record a single attack attempt with mechanism attribution."""
        if not technique and pap_technique:
            technique = pap_technique
        if not technique:
            technique = "unknown"

        target_nid = self._ensure_node(NODE_TARGET, self.target_model_id)
        tech_nid = self._ensure_node(NODE_TECHNIQUE, technique, pap=pap_technique or technique)

        fp = fingerprint or {}
        refusal_style = fp.get("refusal_style", "soft_refusal")
        mechanisms = fp.get("inferred_defense_mechanisms") or infer_defense_mechanisms(
            refusal_style, response_text, None,
        )

        is_success = outcome == "success" or score >= 4.0
        edge_type = EDGE_SUCCESS_ON if is_success else EDGE_FAILED_ON
        self._add_edge(tech_nid, target_nid, edge_type, score=score, ts=time.time())

        for mech in mechanisms:
            mech_nid = self._ensure_node(NODE_DEFENSE_MECHANISM, mech, label=mech)
            self._ensure_node(NODE_TARGET, self.target_model_id)
            self._add_edge(target_nid, mech_nid, EDGE_DEFENDED_BY, ts=time.time())
            mech_edge = EDGE_BYPASSED_BY if is_success else EDGE_BLOCKED_BY
            self._add_edge(tech_nid, mech_nid, mech_edge, score=score, ts=time.time())

        if not is_success and refusal_style:
            rp_nid = self._ensure_node(NODE_REFUSAL_PATTERN, refusal_style)
            self._add_edge(tech_nid, rp_nid, EDGE_FAILED_ON, score=score)

        self.save()

    def upsert_session(self, state: dict[str, Any], fingerprint: dict, outcome: str) -> None:
        """Persist session-level summary nodes and edges."""
        target_nid = self._ensure_node(NODE_TARGET, self.target_model_id)

        for vuln in fingerprint.get("vulnerabilities", [])[:8]:
            v_nid = self._ensure_node(NODE_VULNERABILITY, vuln)
            self._add_edge(target_nid, v_nid, EDGE_SUCCESS_ON if outcome == "success" else EDGE_FAILED_ON)

        for mech in fingerprint.get("inferred_defense_mechanisms", []):
            mech_nid = self._ensure_node(NODE_DEFENSE_MECHANISM, mech)
            self._add_edge(target_nid, mech_nid, EDGE_DEFENDED_BY)

        pap = state.get("active_persuasion_technique", "")
        if pap and outcome == "success":
            strat_id = f"{pap}:{state.get('current_obfuscation_tier', 'none')}"
            s_nid = self._ensure_node(
                NODE_SUCCESS_STRATEGY, strat_id,
                pap=pap, score=state.get("prometheus_score", 0.0),
            )
            tech_nid = self._ensure_node(NODE_TECHNIQUE, pap)
            self._add_edge(s_nid, tech_nid, EDGE_DERIVED_FROM)
            self._add_edge(tech_nid, target_nid, EDGE_SUCCESS_ON, score=state.get("prometheus_score", 0.0))

        self.save()

    def get_mechanism_effectiveness(self, technique_id: str) -> dict[str, float]:
        """P(success | technique, mechanism) per DefenseMechanism."""
        tech_nid = _node_id(NODE_TECHNIQUE, technique_id)
        if tech_nid not in self._graph:
            return {}

        result: dict[str, float] = {}
        for _, mech_nid, data in self._graph.out_edges(tech_nid, data=True):
            if data.get("edge_type") not in (EDGE_BYPASSED_BY, EDGE_BLOCKED_BY):
                continue
            mech_key = self._graph.nodes[mech_nid].get("key", "")
            stats = self._technique_mechanism_counts(technique_id, mech_key)
            s, f = stats.get("success", 0), stats.get("failure", 0)
            result[mech_key] = (s + _BETA_ALPHA) / (s + f + _BETA_ALPHA + _BETA_BETA)
        return result

    def _technique_mechanism_counts(self, technique: str, mechanism: str) -> dict[str, int]:
        tech_nid = _node_id(NODE_TECHNIQUE, technique)
        mech_nid = _node_id(NODE_DEFENSE_MECHANISM, mechanism)
        counts = {"success": 0, "failure": 0}
        if tech_nid not in self._graph or mech_nid not in self._graph:
            return counts
        for _, dst, data in self._graph.out_edges(tech_nid, data=True):
            if dst != mech_nid:
                continue
            et = data.get("edge_type")
            if et == EDGE_BYPASSED_BY:
                counts["success"] += 1
            elif et == EDGE_BLOCKED_BY:
                counts["failure"] += 1
        return counts

    def get_successful_strategies(self, vulnerability_ids: list[str] | None = None, k: int = 5) -> list[dict]:
        results: list[dict] = []
        for nid, attrs in self._graph.nodes(data=True):
            if attrs.get("node_type") != NODE_SUCCESS_STRATEGY:
                continue
            results.append({
                "strategy_id": attrs.get("key", ""),
                "pap": attrs.get("pap", ""),
                "score": attrs.get("score", 0.0),
            })
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:k]

    def get_failed_strategies(self, mechanism_ids: list[str] | None = None, k: int = 5) -> list[dict]:
        results: list[dict] = []
        mech_filter = set(mechanism_ids or [])
        for src, dst, data in self._graph.edges(data=True):
            if data.get("edge_type") != EDGE_BLOCKED_BY:
                continue
            src_type = self._graph.nodes[src].get("node_type")
            dst_type = self._graph.nodes[dst].get("node_type")
            if src_type != NODE_TECHNIQUE or dst_type != NODE_DEFENSE_MECHANISM:
                continue
            mech = self._graph.nodes[dst].get("key", "")
            if mech_filter and mech not in mech_filter:
                continue
            technique = self._graph.nodes[src].get("key", "")
            results.append({
                "technique": technique,
                "defense_mechanism": mech,
                "edge_type": EDGE_BLOCKED_BY,
                "score": data.get("score", 0.0),
            })
        return results[:k]

    def find_similar_targets(self, fingerprint: dict, k: int = 3) -> list[str]:
        """Return up to *k* other target model IDs that share defense mechanisms
        with the given fingerprint.

        Algorithm
        ---------
        For each persisted graph file in ``GRAPH_DIR`` (excluding self):
          1. Load the graph JSON without instantiating a full ``ThreatMemoryGraph``
             (cheap — just reads node-link data).
          2. Extract the set of ``DefenseMechanism`` node keys.
          3. Compute Jaccard similarity with the query fingerprint's mechanisms:
             ``|A ∩ B| / |A ∪ B|``.
          4. Keep targets with similarity >= ``_SIMILAR_TARGET_THRESHOLD``.

        Returns targets sorted by similarity (highest first), excluding self.
        Returns an empty list when no mechanisms are present in the fingerprint.
        """
        mechs = set(fingerprint.get("inferred_defense_mechanisms", []))
        if not mechs:
            return []

        candidates: list[tuple[float, str]] = []
        self_stem = _sanitize_id(self.target_model_id)

        if not GRAPH_DIR.exists():
            return []

        for graph_path in GRAPH_DIR.glob("*.json"):
            stem = graph_path.stem
            if stem == self_stem or stem.endswith(".bak"):
                continue
            try:
                data = json.loads(graph_path.read_text(encoding="utf-8"))
                # Extract DefenseMechanism node keys without full graph load
                other_mechs: set[str] = set()
                for node in data.get("nodes", []):
                    if node.get("node_type") == NODE_DEFENSE_MECHANISM:
                        key = node.get("key", "")
                        if key:
                            other_mechs.add(key)
                if not other_mechs:
                    continue
                intersection = len(mechs & other_mechs)
                union = len(mechs | other_mechs)
                similarity = intersection / union if union > 0 else 0.0
                if similarity >= _SIMILAR_TARGET_THRESHOLD:
                    candidates.append((similarity, stem))
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[ThreatGraph] Skipping %s in find_similar_targets: %s",
                    graph_path.name, exc,
                )

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [stem for _, stem in candidates[:k]]

    def query_planning_context(
        self,
        objective: str,
        fingerprint: dict,
        k: int = 5,
    ) -> GraphContext:
        """Bundle retrieval for RAG planner."""
        mechanisms = list(fingerprint.get("inferred_defense_mechanisms", []))
        techniques = list(fingerprint.get("vulnerabilities", []))

        tech_stats: dict[str, dict[str, float]] = {}
        for vuln in techniques:
            key = vuln.split(":")[-1] if ":" in vuln else vuln
            tech_stats[key] = self.get_mechanism_effectiveness(key)

        obs = sum(
            1 for _, _, d in self._graph.edges(data=True)
            if d.get("edge_type") in (EDGE_SUCCESS_ON, EDGE_FAILED_ON, EDGE_BLOCKED_BY, EDGE_BYPASSED_BY)
        )

        return GraphContext(
            successful_strategies=self.get_successful_strategies(k=k),
            failed_strategies=self.get_failed_strategies(mechanism_ids=mechanisms, k=k),
            similar_targets=self.find_similar_targets(fingerprint, k=k),
            defense_mechanisms=mechanisms,
            technique_stats=tech_stats,
            observation_count=obs,
        )


_graph_cache: dict[str, ThreatMemoryGraph] = {}


def list_all_graph_targets() -> list[str]:
    """Return target_model_id strings for every persisted threat graph on disk.

    Reads the filenames in ``GRAPH_DIR`` and reverse-sanitises them back to
    model-id strings.  This enables cross-target retrieval without keeping all
    graphs in memory at once.
    """
    if not GRAPH_DIR.exists():
        return []
    return [
        p.stem  # sanitised model id (slashes → underscores)
        for p in GRAPH_DIR.glob("*.json")
        if not p.stem.endswith(".bak")
    ]


def get_threat_graph(target_model_id: str) -> ThreatMemoryGraph:
    if target_model_id not in _graph_cache:
        _graph_cache[target_model_id] = ThreatMemoryGraph(target_model_id)
    return _graph_cache[target_model_id]
