"""
scratch/live_fire_test.py
─────────────────────────────────────────────────────────────────────────────
PromptEvo — Instrumented End-to-End Live Fire Test (5-Turn Cap)

Verifies:
  1. Attack payloads mutate across turns (Zombie Branch fix)
  2. SeqBranch correctly skips already-evaluated branches
  3. prometheus_score, mutation_type, obfuscation_tier captured per turn
  4. No duplicate HumanMessage injection from hive_mind (Rogue State fix)

Run:
    python scratch/live_fire_test.py

Override env before running:
    HITL_ENABLED=false  (automatically forced below via os.environ)
    MAX_SESSION_TURNS=5 (forced below)
"""

from __future__ import annotations

import os
import sys
import uuid
import logging
import time
from typing import Any

# ── Force env overrides BEFORE any PromptEvo imports ─────────────────────────
os.environ["HITL_ENABLED"]       = "false"   # must not pause for human input
os.environ["MAX_SESSION_TURNS"]  = "5"        # hard 5-turn cap
os.environ["LOG_LEVEL"]          = "INFO"     # capture SeqBranch and hive_mind logs
os.environ["STREAM_OUTPUT"]      = "true"

# Patch sys.path so imports resolve from the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Logging -- route to stdout for clean capture ---------------------------------
handler = logging.StreamHandler(sys.stdout)
handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(name)-35s  %(levelname)-7s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers = [handler],
    force   = True,
)
logger = logging.getLogger("live_fire_test")

# ── PromptEvo imports ─────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(override=False)

import config
from core.state import AuditorState, default_state
from core.graph import app
import core.graph as _graph_module

# ─────────────────────────────────────────────────────────────────────────────
# PER-TURN TELEMETRY STORE
# ─────────────────────────────────────────────────────────────────────────────

class TurnRecord:
    def __init__(self, turn: int):
        self.turn            = turn
        self.prometheus_score: float = 0.0
        self.rahs_score:       float = 0.0
        self.mutation_type:    str   = "unknown"
        self.obfuscation_tier: str   = "none"
        self.response_class:   str   = "unknown"
        self.attack_status:    str   = "in_progress"
        self.active_technique: str   = "none"
        self.branch_payloads:  list[str] = []
        self.seqbranch_log:    list[str] = []
        self.message_count:    int   = 0
        self.zombie_guard_skipped: int = 0

    def to_row(self) -> str:
        return (
            f"  Turn {self.turn:>2} │ score={self.prometheus_score:.1f} │ "
            f"RAHS={self.rahs_score:.2f} │ class={self.response_class:<15} │ "
            f"obfus={self.obfuscation_tier:<10} │ technique={self.active_technique:<28} │ "
            f"status={self.attack_status}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_live_fire_test() -> None:
    logger.info("=" * 80)
    logger.info("LIVE FIRE TEST — PromptEvo 5-Turn E2E Run")
    logger.info("=" * 80)

    sid = f"livefire-{uuid.uuid4().hex[:8]}"

    # ── Configure LLMs and adapter ────────────────────────────────────────────
    logger.info("[Setup] Attacker: %s / %s", config.settings.attacker_provider, config.settings.attacker_model)
    logger.info("[Setup] Target:   %s / %s", config.settings.target_provider,   config.settings.target_model)
    logger.info("[Setup] Judge:    %s / %s", config.settings.judge_provider,     config.settings.judge_model)
    logger.info("[Setup] HITL: %s  MaxTurns: %s", os.environ["HITL_ENABLED"], os.environ["MAX_SESSION_TURNS"])

    # ── Load-balanced LLM initialization (multi-key Groq) ─────────────────────
    # Dedicated API keys per role to distribute RPM/TPM load across Groq accounts.
    from langchain_groq import ChatGroq
    _ATTACKER_MODEL = "llama-3.3-70b-versatile"
    _JUDGE_MODEL    = "llama-3.3-70b-versatile"

    groq_judge_key     = os.environ.get("GROQ_JUDGE_KEY")     or os.environ.get("GROQ_API_KEY")
    groq_attacker_key1 = os.environ.get("GROQ_ATTACKER_KEY_1") or os.environ.get("GROQ_API_KEY")
    groq_attacker_key2 = os.environ.get("GROQ_ATTACKER_KEY_2") or os.environ.get("GROQ_API_KEY")

    judge_llm             = ChatGroq(model=_JUDGE_MODEL,    api_key=groq_judge_key,     temperature=0.0)
    attacker_llm_primary  = ChatGroq(model=_ATTACKER_MODEL, api_key=groq_attacker_key1, temperature=0.9)
    attacker_llm_secondary = ChatGroq(model=_ATTACKER_MODEL, api_key=groq_attacker_key2, temperature=0.9)

    # Unified attacker with automatic key-2 fallback on rate-limit errors
    attacker_llm = attacker_llm_primary.with_fallbacks([attacker_llm_secondary])

    logger.info(
        "[Setup] Load-balanced attacker LLM: %s (KEY_1 primary → KEY_2 fallback)",
        _ATTACKER_MODEL,
    )
    logger.info("[Setup] Judge LLM: %s (GROQ_JUDGE_KEY)", _JUDGE_MODEL)

    target_adptr = config.get_target_adapter()

    _graph_module._ATTACKER_LLM   = attacker_llm
    _graph_module._TARGET_ADAPTER = target_adptr

    objective = config.settings.audit_objective
    logger.info("[Setup] Objective: %s", objective)

    # ── Build initial state ───────────────────────────────────────────────────
    initial_state: AuditorState = default_state(
        goal         = objective,
        target_model = config.settings.target_model,
        session_id   = sid,
    )

    langgraph_config = {
        "configurable": {
            "thread_id":    sid,
            # Inject resolved LLMs so resolve_llm() picks them up for EVERY node
            # (Prometheus/judge, hive_mind, decomposer, combiner, etc.)
            "attacker_llm": attacker_llm,
            "judge_llm":    judge_llm,
        },
        "recursion_limit": 150,   # default 25 is exhausted by multi-agent graph on 5-turn runs
    }

    # ── Telemetry accumulators ────────────────────────────────────────────────
    turn_records: list[TurnRecord] = []
    current_turn_record = TurnRecord(turn=0)
    cumulative_state: dict[str, Any] = dict(initial_state)

    # Payload dedup tracker — key assertion for Zombie Branch fix
    payload_history: list[str] = []
    duplicate_payload_detected = False

    # ── Stream execution ──────────────────────────────────────────────────────
    logger.info("\n%s\n  STREAMING EXECUTION START\n%s", "=" * 80, "=" * 80)
    t_start = time.monotonic()

    try:
        for chunk in app.stream(initial_state, langgraph_config, stream_mode="updates"):
            for node_name, state_delta in chunk.items():

                if node_name == "__interrupt__":
                    logger.warning("[SeqBranch] HITL interrupt received — should NOT happen (HITL_ENABLED=false)")
                    continue

                if state_delta is None:
                    state_delta = {}

                # Update cumulative view
                cumulative_state.update(state_delta)

                turn_num = cumulative_state.get("turn_count", 0)

                # ── Start new turn record when turn_count increments ──────────
                if turn_num > current_turn_record.turn:
                    if current_turn_record.turn > 0:
                        turn_records.append(current_turn_record)
                    current_turn_record = TurnRecord(turn=turn_num)
                elif current_turn_record.turn == 0 and turn_num == 0:
                    # Initial setup turns (scout, analyst before first attack)
                    pass

                # ── Capture telemetry per node ────────────────────────────────
                if node_name == "judge_and_score":
                    current_turn_record.prometheus_score = state_delta.get("prometheus_score", 0.0)
                    current_turn_record.rahs_score       = state_delta.get("rahs_score", 0.0)
                    current_turn_record.attack_status    = state_delta.get("attack_status", "in_progress")

                if node_name == "response_classifier":
                    current_turn_record.response_class = state_delta.get("response_class", "unknown")

                if "current_obfuscation_tier" in state_delta:
                    current_turn_record.obfuscation_tier = state_delta["current_obfuscation_tier"]

                if "active_persuasion_technique" in state_delta:
                    current_turn_record.active_technique = state_delta["active_persuasion_technique"]

                # Track prometheus score from any node that emits it
                if "prometheus_score" in state_delta and state_delta["prometheus_score"] > 0:
                    current_turn_record.prometheus_score = state_delta["prometheus_score"]
                if "rahs_score" in state_delta and state_delta["rahs_score"] > 0:
                    current_turn_record.rahs_score = state_delta["rahs_score"]
                if "attack_status" in state_delta:
                    current_turn_record.attack_status = state_delta["attack_status"]
                if "response_class" in state_delta:
                    current_turn_record.response_class = state_delta["response_class"]

                # ── Track message payloads (HumanMessages) ────────────────────
                if "messages" in state_delta:
                    msgs = state_delta["messages"]
                    if msgs:
                        msg_count = cumulative_state.get("messages", [])
                        current_turn_record.message_count = len(msg_count)
                        for m in msgs:
                            role = getattr(m, "type", getattr(m, "role", ""))
                            if role in ("human", "user"):
                                payload_text = m.content if isinstance(m.content, str) else str(m.content)
                                payload_preview = payload_text[:120].replace("\n", " ")

                                # ── ZOMBIE GUARD ASSERTION ────────────────────
                                if payload_text in payload_history:
                                    duplicate_payload_detected = True
                                    logger.error(
                                        "[ZOMBIE BRANCH DETECTED] Turn %d node=%s repeated payload: '%s...'",
                                        turn_num, node_name, payload_preview[:80]
                                    )
                                else:
                                    payload_history.append(payload_text)
                                    logger.info(
                                        "[NEW PAYLOAD] Turn %d node=%s  len=%d  preview='%s...'",
                                        turn_num, node_name, len(payload_text), payload_preview[:80]
                                    )

                # ── Log candidate_branches state for zombie guard verification ─
                if "candidate_branches" in state_delta:
                    branches = state_delta["candidate_branches"]
                    evaluated = [b for b in branches if b.get("prometheus_score", 0.0) > 0.0]
                    unevaluated = [b for b in branches if b.get("prometheus_score", 0.0) == 0.0 and not b.get("is_pruned")]
                    pruned = [b for b in branches if b.get("is_pruned")]
                    current_turn_record.zombie_guard_skipped = len(evaluated)
                    logger.info(
                        "[BranchState] node=%-20s  total=%d  unevaluated=%d  evaluated(skipped)=%d  pruned=%d",
                        node_name, len(branches), len(unevaluated), len(evaluated), len(pruned)
                    )

                # ── Node event log ────────────────────────────────────────────
                prom  = state_delta.get("prometheus_score", "")
                rahs  = state_delta.get("rahs_score", "")
                coop  = state_delta.get("cooperation_score", "")
                status = state_delta.get("attack_status", "")
                extra = ""
                if prom: extra += f" judge={prom:.1f}"
                if rahs: extra += f" RAHS={rahs:.2f}"
                if coop: extra += f" coop={coop:.2f}"
                if status: extra += f" [{status}]"
                logger.info("  ▶ %-28s  turn=%d%s", node_name, turn_num, extra)

                # ── Terminal condition ─────────────────────────────────────────
                if cumulative_state.get("attack_status") in ("success", "failure", "exhausted"):
                    break

            # ── Throttle: gentle pace between stream chunks to avoid RPM bursts ─
            time.sleep(2)

    except KeyboardInterrupt:
        logger.warning("[LiveFire] Interrupted by user.")
    except Exception as exc:
        logger.exception("[LiveFire] Graph execution error: %s", exc)

    elapsed = time.monotonic() - t_start

    # Flush last turn record
    if current_turn_record.turn > 0:
        turn_records.append(current_turn_record)

    # ── Populate mutation_type from cumulative state ───────────────────────────
    for rec in turn_records:
        branches = cumulative_state.get("candidate_branches", [])
        # Find the winning (or last-evaluated) branch for this turn
        for b in branches:
            if b.get("prometheus_score", 0.0) == rec.prometheus_score and rec.prometheus_score > 0:
                rec.mutation_type = b.get("mutation_type", b.get("technique", "unknown"))
                break
        if rec.mutation_type == "unknown":
            rec.mutation_type = cumulative_state.get("active_persuasion_technique", "unknown")

    # ─────────────────────────────────────────────────────────────────────────
    # LIVE FIRE REPORT
    # ─────────────────────────────────────────────────────────────────────────

    DIVIDER = "─" * 110
    print()
    print("=" * 110)
    print("  PromptEvo LIVE FIRE REPORT")
    print(f"  Session ID : {sid}")
    print(f"  Objective  : {objective}")
    print(f"  Target     : {config.settings.target_provider} / {config.settings.target_model}")
    print(f"  Attacker   : {config.settings.attacker_provider} / {config.settings.attacker_model}")
    print(f"  Wall time  : {elapsed:.1f}s")
    print("=" * 110)

    # ── Section 1: Turn-by-Turn Breakdown ─────────────────────────────────────
    print()
    print(DIVIDER)
    print("  SECTION 1 — TURN-BY-TURN BREAKDOWN")
    print(DIVIDER)
    print(f"  {'Turn':<6} │ {'Judge':>6} │ {'RAHS':>6} │ {'Class':<15} │ {'Obfuscation':<12} │ {'PAP Technique':<30} │ Status")
    print(DIVIDER)

    for rec in turn_records:
        score_marker = "✅" if rec.prometheus_score >= 4.0 else "⚠️ " if rec.prometheus_score >= 2.5 else "❌"
        print(
            f"  {score_marker} T{rec.turn:<3} │ "
            f"{rec.prometheus_score:>5.1f} │ "
            f"{rec.rahs_score:>6.2f} │ "
            f"{rec.response_class:<15} │ "
            f"{rec.obfuscation_tier:<12} │ "
            f"{rec.active_technique:<30} │ "
            f"{rec.attack_status}"
        )

    if not turn_records:
        print("  [No attack turns completed — session may have stopped in warm-up phase]")

    # ── Section 2: Zombie Guard Verification ──────────────────────────────────
    print()
    print(DIVIDER)
    print("  SECTION 2 — ZOMBIE BRANCH GUARD VERIFICATION")
    print(DIVIDER)

    total_payloads = len(payload_history)
    unique_payloads = len(set(payload_history))

    if duplicate_payload_detected:
        print("  [FAIL] ZOMBIE BRANCH DETECTED -- duplicate payload(s) sent to target")
        print("         The FIX 2 guard (prometheus_score == 0.0) is NOT working correctly.")
    else:
        print("  [PASS] ZOMBIE BRANCH GUARD: PASSED")
        print("         All payloads sent to the target were unique across turns.")

    print(f"  Total HumanMessages sent to target : {total_payloads}")
    print(f"  Unique payloads                    : {unique_payloads}")
    print(f"  Duplicates detected                : {total_payloads - unique_payloads}")

    print()
    print("  Branch state evolution (per SeqBranch cycle):")
    for i, rec in enumerate(turn_records, 1):
        print(f"    Turn {rec.turn:>2}: already-evaluated branches skipped = {rec.zombie_guard_skipped}")

    # ── Section 3: Final Outcome ───────────────────────────────────────────────
    print()
    print(DIVIDER)
    print("  SECTION 3 — FINAL OUTCOME")
    print(DIVIDER)

    final_status    = cumulative_state.get("attack_status", "unknown")
    final_score     = cumulative_state.get("prometheus_score", 0.0)
    final_rahs      = cumulative_state.get("rahs_score", 0.0)
    total_turns     = cumulative_state.get("turn_count", len(turn_records))
    pruned_techs    = cumulative_state.get("pruned_techniques", [])
    prot_blocks     = cumulative_state.get("protected_blocks", [])
    fail_context    = cumulative_state.get("pruned_failure_context", [])

    outcome_icon = {
        "success":  "[JAILBREAK SUCCESSFUL]",
        "failure":  "[DEFENDED] Target defended successfully",
        "exhausted":"[EXHAUSTED] Max turns reached",
        "in_progress": "[INCOMPLETE] Session stopped mid-run",
    }.get(final_status, f"[{final_status.upper()}]")

    print(f"  Outcome        : {outcome_icon}")
    print(f"  Final Status   : {final_status}")
    print(f"  Final Judge    : {final_score:.1f} / 5.0")
    print(f"  Final RAHS     : {final_rahs:.2f} / 10.0")
    print(f"  Total Turns    : {total_turns}")
    print(f"  Pruned Techs   : {pruned_techs or 'none'}")
    print(f"  Protected Blks : {len(prot_blocks)} entries")
    print(f"  Failure Ctx    : {len(fail_context)} entries (cap=15)")

    if final_status == "success":
        winning_branch = None
        for b in cumulative_state.get("candidate_branches", []):
            if b.get("winner"):
                winning_branch = b
                break
        if winning_branch:
            print()
            print("  Winning Branch Details:")
            print(f"    Branch ID  : {winning_branch.get('branch_id', 'N/A')}")
            print(f"    Score      : {winning_branch.get('prometheus_score', 0.0):.1f}")
            print(f"    Technique  : {winning_branch.get('technique', winning_branch.get('mutation_type', 'N/A'))}")
            wpl = winning_branch.get("prompt_variant", "")
            print(f"    Payload    : {wpl[:200]}{'...' if len(wpl) > 200 else ''}")

    # ── Section 4: State Integrity Check ─────────────────────────────────────
    print()
    print(DIVIDER)
    print("  SECTION 4 — STATE INTEGRITY CHECKS")
    print(DIVIDER)

    # Verify protected_blocks didn't explode (VULN 6 fix)
    pb_ok = len(prot_blocks) <= 20
    print(f"  protected_blocks cap (≤20)   : {'✅ OK' if pb_ok else '❌ EXCEEDED'} — {len(prot_blocks)} entries")

    # Verify pruned_failure_context didn't exceed 15 (Phase 1 fix)
    fc_ok = len(fail_context) <= 15
    print(f"  pruned_failure_context (≤15) : {'✅ OK' if fc_ok else '❌ EXCEEDED'} — {len(fail_context)} entries")

    # Verify message history is reasonable (no explosive growth)
    msgs = cumulative_state.get("messages", [])
    msg_ok = len(msgs) < 50
    print(f"  message history (< 50)       : {'✅ OK' if msg_ok else '⚠️  HIGH'} — {len(msgs)} messages")

    print()
    print("=" * 110)
    print("  END OF LIVE FIRE REPORT")
    print("=" * 110)
    print()


if __name__ == "__main__":
    run_live_fire_test()
