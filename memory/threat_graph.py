"""
Threat Graph Architecture.
Persists per-target Defense Mechanisms and Attack Technique edges.
"""

import json
import logging
from pathlib import Path
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

    def save(self):
        try:
            data = nx.node_link_data(self.graph, edges="links")
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save threat graph for {self.target_id}: {e}")
