from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class HITLAction:
    action: Literal[
        "approve",        # send payload as-is
        "edit",           # send modified payload
        "switch_pap",     # change PAP technique then send
        "abort",          # terminate session with reason
        "select_branch",  # pick a specific TAP branch
    ]
    
    # For "edit"
    edited_payload: Optional[str] = None
    
    # For "switch_pap"
    new_pap_technique: Optional[str] = None
    
    # For "abort"
    abort_reason: Optional[str] = None
    
    # For "select_branch"  
    branch_index: Optional[int] = None


class HITLHandler:
    def process(
        self, 
        action: HITLAction, 
        state: dict,
    ) -> dict:
        """
        Process the HITL action and return a state delta.
        """
        a = action.action
        if a == "approve":
            return {}
        elif a == "edit":
            return {"pending_payload": action.edited_payload or ""}
        elif a == "switch_pap":
            return {
                "active_persuasion_technique": action.new_pap_technique or "",
                "route_decision": "attack_swarm"
            }
        elif a == "abort":
            return {
                "attack_status": "aborted",
                "latest_feedback": f"ABORT: {action.abort_reason or 'operator abort'}",
                "route_decision": "terminal"
            }
        elif a == "select_branch":
            idx = action.branch_index or 0
            candidates = state.get("candidate_branches", [])
            payload = ""
            if 0 <= idx < len(candidates):
                # candidates[idx] is a BranchDict — extract the string payload.
                # Use payload_delivered (obfuscated form sent to target) when
                # present, falling back to prompt_variant for backward compat.
                branch = candidates[idx]
                payload = (
                    branch.get("payload_delivered")
                    or branch.get("prompt_variant", "")
                )
            return {"pending_payload": payload}
        else:
            raise ValueError(f"Invalid HITL action: {a}")
