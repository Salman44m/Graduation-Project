import os
import yaml
from memory.threat_graph import ThreatMemoryGraph

class PatternMiner:
    MIN_FREQUENCY_THRESHOLD = 3

    def __init__(self, target_id: str = "global"):
        self.target_id = target_id
        self.graph = ThreatMemoryGraph(target_id)

    def mine_failures(self) -> list[dict]:
        failures_count = {}
        
        for u, v, k, data in self.graph.graph.edges(keys=True, data=True):
            if data.get("type") == "BLOCKED_BY":
                # u is technique, v is mechanism
                tech = str(u)
                mech = str(v)
                count = data.get("count", 1)
                
                pair = (tech, mech)
                failures_count[pair] = failures_count.get(pair, 0) + count
                
        results = []
        for (tech, mech), count in failures_count.items():
            if count >= self.MIN_FREQUENCY_THRESHOLD:
                pattern_id = f"fail_{tech}_{mech}"
                results.append({
                    "pattern_id": pattern_id,
                    "technique": tech,
                    "defense_mechanism": mech,
                    "failure_count": count,
                    "avoid_instruction": f"Avoid using {tech} as it is consistently blocked by {mech}"
                })
                
        if results:
            self._write_yaml_safe("data/tactics/mined_failures.yaml", results, "pattern_id")
            
        return results

    def mine_successes(self) -> list[dict]:
        success_count = {}
        
        for u, v, k, data in self.graph.graph.edges(keys=True, data=True):
            if data.get("type") in ("BYPASSED_BY", "SUCCESS_ON"):
                tech = str(u)
                mech = str(v)
                count = data.get("count", 1)
                
                pair = (tech, mech)
                success_count[pair] = success_count.get(pair, 0) + count
                
        results = []
        for (tech, mech), count in success_count.items():
            if count >= self.MIN_FREQUENCY_THRESHOLD:
                pattern_id = f"succ_{tech}_{mech}"
                results.append({
                    "pattern_id": pattern_id,
                    "technique": tech,
                    "bypassed_mechanism": mech,
                    "success_count": count
                })
                
        if results:
            self._write_yaml_safe("data/tactics/mined_templates.yaml", results, "pattern_id")
            
        return results

    def mine_session(self, state: dict) -> dict:
        try:
            failures  = self.mine_failures()
            successes = self.mine_successes()
            return {
                "templates": successes if successes else [],
                "failures":  failures  if failures  else [],
            }
        except Exception:
            return {"templates": [], "failures": []}

    def _write_yaml_safe(self, filepath: str, new_data: list[dict], key: str):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            existing_data = []
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    existing_data = yaml.safe_load(f) or []
            
            existing_dict = {item[key]: item for item in existing_data if key in item}
            
            for item in new_data:
                item_key = item[key]
                if item_key in existing_dict:
                    count_field = "success_count" if "succ_" in item_key else "failure_count"
                    existing_dict[item_key][count_field] = max(
                        existing_dict[item_key].get(count_field, 0),
                        item[count_field]
                    )
                else:
                    existing_dict[item_key] = item
                    
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(list(existing_dict.values()), f, default_flow_style=False, sort_keys=False)
                
        except Exception:
            pass
