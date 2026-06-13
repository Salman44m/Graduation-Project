"""
research/dataset_builder.py
─────────────────────────────────────────────────────────────────────────────
Dataset Builder — structured session logs for offline predictive modeling.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "research" / "sessions.jsonl"
SCHEMA_VERSION = "1.0.0"


def _dataset_path() -> Path:
    env_path = os.getenv("RESEARCH_DATASET_PATH", "")
    return Path(env_path) if env_path else DEFAULT_DATASET_PATH


def _include_payloads() -> bool:
    return os.getenv("RESEARCH_LOG_INCLUDE_PAYLOADS", "false").lower() in ("1", "true", "yes")


def build_session_record(state: dict[str, Any]) -> dict[str, Any]:
    """Build ML-ready session record from AuditorState."""
    fingerprint = dict(state.get("defense_fingerprint") or {})
    attack_plan = dict(state.get("attack_plan") or {})
    graph_ctx = dict(state.get("graph_retrieval_context") or {})

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "session_id": state.get("session_id", ""),
        "target_model_id": state.get("target_model_id", "unknown"),
        "timestamp": time.time(),
        "fingerprint": fingerprint,
        "attack_plan": {
            k: v for k, v in attack_plan.items()
            if k != "candidate_plans" or _include_payloads()
        },
        "curriculum_stage_reached": int(state.get("curriculum_stage", 0)),
        "result": state.get("attack_status", "unknown"),
        "prometheus_score": float(state.get("prometheus_score", 0.0)),
        "rahs_score": float(state.get("rahs_score", 0.0)),
        "judge_ensemble_scores": dict(state.get("judge_ensemble_scores") or {}),
        "primary_defense_mechanisms": fingerprint.get("inferred_defense_mechanisms", []),
        "techniques_used": [
            state.get("active_persuasion_technique", ""),
            state.get("evolved_technique", ""),
        ],
        "turn_count": int(state.get("turn_count", 0)),
        "graph_context_summary": {
            "observation_count": graph_ctx.get("observation_count", 0),
            "defense_mechanisms": graph_ctx.get("defense_mechanisms", []),
            "predicted_success": attack_plan.get("expected_success_probability"),
            "plan_confidence": attack_plan.get("confidence"),
        },
    }

    if _include_payloads():
        record["objective_snippet"] = str(state.get("core_malicious_objective", ""))[:200]

    return record


def log_session_record(state: dict[str, Any]) -> bool:
    """Append one session record to JSONL dataset. Non-blocking."""
    try:
        path = _dataset_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = build_session_record(state)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("[DatasetBuilder] Logged session %s", record.get("session_id", "")[:8])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("[DatasetBuilder] Failed: %s", exc)
        return False
