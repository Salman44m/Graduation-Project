import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def export_session(self, state: dict) -> dict:
        try:
            record = {
                "session_id": state.get("session_id"),
                "target_model_id": state.get("target_model_id"),
                "core_malicious_objective": state.get("core_malicious_objective"),
                "attack_status": state.get("attack_status"),
                "turn_count": state.get("turn_count"),
                "cooperation_score": state.get("cooperation_score"),
                "prometheus_score": state.get("prometheus_score"),
                "defense_fingerprint": state.get("defense_fingerprint"),
                "curriculum_stage": state.get("curriculum_stage"),
                "active_persuasion_technique": state.get("active_persuasion_technique"),
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "phase_1_active": True,
                "schema_version": "1.0.0"
            }
            
            if "ensemble_scores" in state:
                record["ensemble_scores"] = state["ensemble_scores"]
            if "mined_patterns_summary" in state:
                record["mined_patterns_summary"] = state["mined_patterns_summary"]
                
            return record
        except Exception as e:
            logger.warning(f"[DatasetBuilder] Error exporting session: {e}")
            return {}

    def save_to_jsonl(self, record: dict, path: str = "data/research/sessions.jsonl"):
        try:
            if not record:
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"[DatasetBuilder] Failed to save JSONL: {e}")

    def export_attack_trajectory(self, state: dict) -> list[dict]:
        trajectory = []
        try:
            messages = state.get("messages", [])
            for i, msg in enumerate(messages):
                role = getattr(msg, "type", None) or getattr(msg, "role", "unknown")
                content = getattr(msg, "content", "")
                content_str = content if isinstance(content, str) else str(content)
                
                # Try to determine node, assume scout or hive_mind for human, unknown for others
                node = "unknown"
                if role in ("human", "user"):
                    # A very simplistic heuristic; in reality state might track this better
                    node = getattr(msg, "name", "unknown")

                turn_record = {
                    "turn": i,
                    "role": role,
                    "content_length": len(content_str),
                    "node": node,
                    "cooperation_score_at_turn": state.get("cooperation_score", 0.0)
                }
                trajectory.append(turn_record)
            return trajectory
        except Exception as e:
            logger.warning(f"[DatasetBuilder] Error exporting trajectory: {e}")
            return trajectory

    def get_dataset_stats(self, path: str = "data/research/sessions.jsonl") -> dict:
        stats = {
            "total_sessions": 0,
            "success_rate": 0.0,
            "avg_turns": 0.0,
            "top_techniques": [],
            "domains_seen": []
        }
        try:
            if not os.path.exists(path):
                return stats
                
            success_count = 0
            total_turns = 0
            techniques = {}
            domains = set()
            
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        stats["total_sessions"] += 1
                        
                        if record.get("attack_status") == "success":
                            success_count += 1
                            
                        total_turns += record.get("turn_count", 0)
                        
                        tech = record.get("active_persuasion_technique")
                        if tech:
                            techniques[tech] = techniques.get(tech, 0) + 1
                            
                        obj = record.get("core_malicious_objective")
                        if obj:
                            domains.add(obj[:50]) # Simple domain proxy
                    except json.JSONDecodeError:
                        continue
                        
            if stats["total_sessions"] > 0:
                stats["success_rate"] = success_count / stats["total_sessions"]
                stats["avg_turns"] = total_turns / stats["total_sessions"]
                
            sorted_techs = sorted(techniques.items(), key=lambda x: x[1], reverse=True)
            stats["top_techniques"] = [t[0] for t in sorted_techs[:3]]
            stats["domains_seen"] = list(domains)
            
            return stats
        except Exception as e:
            logger.warning(f"[DatasetBuilder] Error getting stats: {e}")
            return stats
