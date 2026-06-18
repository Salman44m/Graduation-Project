"""
Threat Graph Architecture.
Persists per-target Defense Mechanisms and Attack Technique edges.
"""

import json
import logging
from pathlib import Path
from typing import Any
import networkx as nx

logger = logging.getLogger(__name__)

class ThreatMemoryGraph:
    def __init__(self, target_id: str):
        self.target_id = target_id
        self.db_dir = Path("data/memory/threat_graphs")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.db_dir / f"{self.target_id}.json"
        
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data, edges="links")
            except Exception as e:
                logger.error(f"Failed to load threat graph for {target_id}: {e}")
                self.graph = nx.MultiDiGraph()
        else:
            self.graph = nx.MultiDiGraph()
            
        # Ensure target node exists
        if not self.graph.has_node(self.target_id):
            self.graph.add_node(self.target_id, type="Target")

    def upsert_session(self, fingerprint: dict, outcome: str):
        mechanisms = fingerprint.get("inferred_defense_mechanisms", [])
        if not mechanisms:
            style = fingerprint.get("refusal_style", "unknown")
            mechanisms = []
            if style == "hard_refusal": mechanisms.append("rlhf_refusal")
            elif style == "policy_cite": mechanisms.append("policy_filter")
            elif style in ("soft_refusal", "deflect", "redirect"): mechanisms.append("semantic_filter")
            
        if not mechanisms:
            mechanisms = ["rlhf_refusal"]
            
        for mech in mechanisms:
            if not self.graph.has_node(mech):
                self.graph.add_node(mech, type="DefenseMechanism")
            
            # Check if DEFENDED_BY edge exists
            has_edge = False
            if self.graph.has_edge(self.target_id, mech):
                for key, data in self.graph[self.target_id][mech].items():
                    if data.get("type") == "DEFENDED_BY":
                        has_edge = True
                        break
            if not has_edge:
                self.graph.add_edge(self.target_id, mech, type="DEFENDED_BY")

    def upsert_attempt(self, technique_id: str, mechanism_id: str, outcome: str, turns: int):
        if not self.graph.has_node(technique_id):
            self.graph.add_node(technique_id, type="AttackTechnique")
        if not self.graph.has_node(mechanism_id):
            self.graph.add_node(mechanism_id, type="DefenseMechanism")
            
        edge_type = "BYPASSED_BY" if outcome in ("success", "full_comply", "partial_comply") else "BLOCKED_BY"
        
        # Check if edge of this type exists between technique and mechanism
        edge_key_to_update = None
        if self.graph.has_edge(technique_id, mechanism_id):
            for key, data in self.graph[technique_id][mechanism_id].items():
                if data.get("type") == edge_type:
                    edge_key_to_update = key
                    break
        
        if edge_key_to_update is not None:
            # Update attributes
            count = self.graph[technique_id][mechanism_id][edge_key_to_update].get("count", 0)
            avg_turn = self.graph[technique_id][mechanism_id][edge_key_to_update].get("avg_turn", 0)
            
            new_count = count + 1
            new_avg = ((avg_turn * count) + turns) / new_count
            
            self.graph[technique_id][mechanism_id][edge_key_to_update]["count"] = new_count
            self.graph[technique_id][mechanism_id][edge_key_to_update]["avg_turn"] = new_avg
        else:
            self.graph.add_edge(technique_id, mechanism_id, type=edge_type, count=1, avg_turn=turns)
            
        # Target edges
        target_edge_type = "SUCCESS_ON" if edge_type == "BYPASSED_BY" else "FAILED_ON"
        if not self.graph.has_node(self.target_id):
            self.graph.add_node(self.target_id, type="Target")
            
        has_target_edge = False
        if self.graph.has_edge(technique_id, self.target_id):
            for key, data in self.graph[technique_id][self.target_id].items():
                if data.get("type") == target_edge_type:
                    has_target_edge = True
                    break
        if not has_target_edge:
            self.graph.add_edge(technique_id, self.target_id, type=target_edge_type)

    def record_block(self, technique_id: str, 
                     mechanism_id: str) -> None:
        self.upsert_attempt(
            technique_id=technique_id,
            mechanism_id=mechanism_id,
            outcome="blocked",
            turns=0
        )

    def get_weakest_mechanisms(self) -> list:
        return []

    def get_failed_strategies(self, mechanism_ids: list[str], k=5) -> list[dict]:
        failed = []
        for mech in mechanism_ids:
            if not self.graph.has_node(mech):
                continue
            
            # Find predecessors (AttackTechnique) connected by BLOCKED_BY
            for pred in self.graph.predecessors(mech):
                for key, data in self.graph[pred][mech].items():
                    if data.get("type") == "BLOCKED_BY":
                        failed.append({
                            "technique": pred,
                            "mechanism": mech,
                            "count": data.get("count", 0),
                            "avg_turn": data.get("avg_turn", 0.0)
                        })
        
        # Sort by count desc
        failed.sort(key=lambda x: x["count"], reverse=True)
        return failed[:k]

    def get_best_techniques_against(self, mechanism_id: str, 
                                   top_k: int = 3) -> list[dict]:
        """
        Return top_k AttackTechniques that most successfully 
        bypassed the given mechanism_id.
        Traverses BYPASSED_BY edges from mechanism to technique.
        Returns list of dicts: [{technique_id, success_rate, 
                                 avg_turns_to_bypass}]
        Returns [] if mechanism unknown or no data.
        Never raises.
        """
        try:
            if mechanism_id not in self.graph.nodes:
                return []
            results = []
            for src, dst, data in self.graph.edges(data=True):
                if (data.get("type") == "BYPASSED_BY" and 
                    dst == mechanism_id):
                    results.append({
                        "technique_id": src,
                        "success_rate": data.get("count", 0),
                        "avg_turns_to_bypass": data.get(
                            "avg_turn", 0)
                    })
            results.sort(key=lambda x: x["success_rate"], 
                        reverse=True)
            return results[:top_k]
        except Exception:
            return []

    def get_technique_stats(self) -> dict[str, dict[str, float]]:
        """Return historical success probability per technique per mechanism.

        Output schema matches what ``generate_attack_plan()`` / ``_score_route()``
        expect for ``technique_stats``:

        .. code-block:: python

            {
                "attack_swarm": {"rlhf_refusal": 0.75, "policy_filter": 0.40},
                "gci":          {"rlhf_refusal": 0.20},
                ...
            }

        P(success) is estimated as:
            bypassed_count / (bypassed_count + blocked_count)
        using additive smoothing (Laplace, alpha=1) so new techniques start at 0.5
        rather than being undefined.

        Never raises.
        """
        try:
            stats: dict[str, dict[str, float]] = {}
            for src, dst, data in self.graph.edges(data=True):
                edge_type = data.get("type", "")
                count = int(data.get("count", 1))
                if edge_type not in ("BYPASSED_BY", "BLOCKED_BY"):
                    continue
                technique_id = src
                mechanism_id = dst
                entry = stats.setdefault(technique_id, {})
                if mechanism_id not in entry:
                    entry[mechanism_id] = {"bypassed": 0, "blocked": 0}
                if edge_type == "BYPASSED_BY":
                    entry[mechanism_id]["bypassed"] += count  # type: ignore[assignment]
                else:
                    entry[mechanism_id]["blocked"] += count   # type: ignore[assignment]

            # Convert raw counts to P(success) using Laplace smoothing (alpha=1)
            result: dict[str, dict[str, float]] = {}
            for technique, mechs in stats.items():
                result[technique] = {}
                for mech, counts in mechs.items():
                    b = counts["bypassed"]  # type: ignore[index]
                    bl = counts["blocked"]  # type: ignore[index]
                    result[technique][mech] = (b + 1) / (b + bl + 2)
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ThreatGraph] get_technique_stats failed: %s", exc)
            return {}

    def get_successful_strategies(self, mechanism_ids: list[str], k: int = 5) -> list[dict[str, Any]]:
        """Return top-k AttackTechniques that successfully bypassed the given mechanisms.

        Mirrors ``get_failed_strategies()`` but traverses BYPASSED_BY edges.
        Returns list of dicts: {technique, mechanism, count, avg_turn}.
        Never raises.
        """
        try:
            successful: list[dict[str, Any]] = []
            for mech in mechanism_ids:
                if not self.graph.has_node(mech):
                    continue
                for pred in self.graph.predecessors(mech):
                    for key, data in self.graph[pred][mech].items():
                        if data.get("type") == "BYPASSED_BY":
                            successful.append({
                                "technique": pred,
                                "mechanism": mech,
                                "count": data.get("count", 0),
                                "avg_turn": data.get("avg_turn", 0.0),
                            })
            successful.sort(key=lambda x: x["count"], reverse=True)
            return successful[:k]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ThreatGraph] get_successful_strategies failed: %s", exc)
            return []

    def save(self):
        try:
            data = nx.node_link_data(self.graph, edges="links")
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save threat graph for {self.target_id}: {e}")


# ── Module-level factory ─────────────────────────────────────────────────────

def get_threat_graph(target_id: str) -> ThreatMemoryGraph:
    """Return a ``ThreatMemoryGraph`` for *target_id*.

    This is the factory function imported by
    ``intelligence.rag_attack_planner.build_graph_retrieval_context``.
    Raising is intentional: callers are expected to wrap in try/except.
    """
    return ThreatMemoryGraph(target_id=target_id)
