"""
intelligence/pattern_miner.py
─────────────────────────────────────────────────────────────────────────────
Attack Pattern Mining — success and failure pattern extraction from sessions.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

TACTICS_DIR = Path(__file__).resolve().parent.parent / "data" / "tactics"
MIN_FREQUENCY = 2
"""Minimum number of occurrences before a pattern graduates to the output.

Lowered from 3 to 2: a pattern seen twice is sufficiently reliable to
include in avoidance/reinforcement guidance without waiting for a third
confirmation. Still filters single-session noise.
"""

COLD_START_SCORE_THRESHOLD_SUCCESS = 4.0
"""Success patterns above this avg_score bypass the MIN_FREQUENCY gate.

A single strong success (prometheus_score >= 4.0) is immediately useful
as a template even before a second confirmation.
"""

COLD_START_SCORE_THRESHOLD_FAILURE = 1.5
"""Failure patterns below this avg_score bypass the MIN_FREQUENCY gate.

A hard refusal (prometheus_score <= 1.5) is immediately useful as an
avoidance signal even before a second confirmation.
"""


def _pattern_key(pap: str, obfuscation: str, mechanism: str) -> str:
    return f"{pap}|{obfuscation}|{mechanism}"


def mine_session_patterns(state: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Extract success and failure pattern dicts from a completed session."""
    pap = state.get("active_persuasion_technique", "unknown")
    obfuscation = state.get("current_obfuscation_tier", "none")
    fingerprint = dict(state.get("defense_fingerprint") or {})
    mechanisms = fingerprint.get("inferred_defense_mechanisms") or ["rlhf_refusal"]
    mechanism = mechanisms[0] if mechanisms else "rlhf_refusal"
    score = float(state.get("prometheus_score", 0.0))
    outcome = state.get("attack_status", "failure")
    refusal_style = fingerprint.get("refusal_style", "soft_refusal")

    success_patterns: list[dict] = []
    failure_patterns: list[dict] = []

    key = _pattern_key(pap, obfuscation, mechanism)

    if outcome == "success" or score >= 4.0:
        success_patterns.append({
            "pattern_id": key,
            "pap": pap,
            "obfuscation": obfuscation,
            "defense_mechanism": mechanism,
            "template_hint": f"Successful {pap} with {obfuscation} against {mechanism}",
            "avg_score": score,
            "failure_count": 0,
        })
    else:
        failure_patterns.append({
            "pattern_id": key,
            "technique": pap,
            "defense_mechanism": mechanism,
            "failure_count": 1,
            "avg_score": score,
            "refusal_style": refusal_style,
            "avoid_instruction": f"Avoid {pap} with {obfuscation} when target uses {mechanism}",
        })

    for entry in state.get("pruned_failure_context") or []:
        mt = entry.get("mutation_type", "unknown")
        failure_patterns.append({
            "pattern_id": f"{mt}|{mechanism}",
            "technique": mt,
            "defense_mechanism": mechanism,
            "failure_count": 1,
            "avg_score": entry.get("score", 1.0),
            "refusal_style": refusal_style,
            "avoid_instruction": f"Avoid mutation {mt} — {entry.get('failure_reason', '')[:80]}",
        })

    return success_patterns, failure_patterns


def _load_yaml_patterns(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return list(data.get("patterns", []))
    except Exception:  # noqa: BLE001
        return []


def _save_yaml_patterns(path: Path, patterns: list[dict], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": "1.0", "label": label, "patterns": patterns}
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def _merge_and_threshold(
    existing: list[dict],
    new_items: list[dict],
    count_key: str = "failure_count",
) -> list[dict]:
    """Merge new pattern items into existing and apply quality thresholds.

    A pattern is included in the output if ANY of these conditions hold:
      1. ``count_key`` >= MIN_FREQUENCY  (seen enough times)
      2. ``avg_score`` >= COLD_START_SCORE_THRESHOLD_SUCCESS  (strong success)
      3. ``avg_score`` <= COLD_START_SCORE_THRESHOLD_FAILURE and
         count_key == "failure_count"  (hard failure — immediate avoidance signal)
    """
    counter: Counter[str] = Counter()
    merged: dict[str, dict] = {p.get("pattern_id", ""): dict(p) for p in existing if p.get("pattern_id")}

    for item in new_items:
        pid = item.get("pattern_id", "")
        if not pid:
            continue
        counter[pid] += 1
        if pid in merged:
            merged[pid][count_key] = merged[pid].get(count_key, 0) + 1
        else:
            merged[pid] = dict(item)
            merged[pid][count_key] = merged[pid].get(count_key, 0) + 1

    result = []
    for p in merged.values():
        cnt = p.get(count_key, 0)
        avg = p.get("avg_score", 0.0)
        # Condition 1: enough repetitions
        if cnt >= MIN_FREQUENCY:
            result.append(p)
            continue
        # Condition 2: cold-start bypass for strong successes
        if avg >= COLD_START_SCORE_THRESHOLD_SUCCESS:
            result.append(p)
            continue
        # Condition 3: cold-start bypass for hard failures (avoid immediately)
        if count_key == "failure_count" and 0 < avg <= COLD_START_SCORE_THRESHOLD_FAILURE:
            result.append(p)
            continue
    return result


def run_pattern_miner(state: dict[str, Any]) -> dict[str, Any]:
    """Mine patterns and persist to YAML. Returns state delta with mined lists."""
    try:
        success_new, failure_new = mine_session_patterns(state)

        templates_path = TACTICS_DIR / "mined_templates.yaml"
        failures_path = TACTICS_DIR / "mined_failures.yaml"

        existing_success = _load_yaml_patterns(templates_path)
        existing_failure = _load_yaml_patterns(failures_path)

        merged_success = _merge_and_threshold(existing_success, success_new, "success_count")
        merged_failure = _merge_and_threshold(existing_failure, failure_new, "failure_count")

        if success_new:
            _save_yaml_patterns(templates_path, merged_success, "success_templates")
        if failure_new:
            _save_yaml_patterns(failures_path, merged_failure, "failure_anti_patterns")

        logger.info(
            "[PatternMiner] Mined success=%d failure=%d (stored %d/%d)",
            len(success_new), len(failure_new), len(merged_success), len(merged_failure),
        )
        return {
            "mined_patterns": merged_success[-20:],
            "mined_failures": merged_failure[-20:],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PatternMiner] Failed (non-fatal): %s", exc)
        return {}
