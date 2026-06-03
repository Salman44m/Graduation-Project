"""
agents/target.py
─────────────────────────────────────────────────────────────────────────────
Target Node — Execution Layer (Dual-Mode)

Architectural Role (Section 2.3, Original Project Doc)
───────────────────────────────────────────────────────
The target_node is the only node in PromptEvo that communicates directly
with the model under audit.  Every other node talks to the attacker LLM or
evaluates internal state.  This node is the single point of contact with the
target through the ``BaseTargetAdapter`` interface.

Two Operating Modes
────────────────────
The node detects which mode it is in by reading ``state["attack_status"]``
and ``state["route_decision"]``:

  ┌──────────────────┬───────────────────────────────────────────────────────┐
  │ Mode             │ Detection                                             │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ WARM-UP          │ route_decision == "analyst"                           │
  │ (scout probe)    │ The scout has appended a HumanMessage Trojan Horse    │
  │                  │ probe.  Deliver the full message history including    │
  │                  │ that probe.  Append the response as an AIMessage.    │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ STANDARD ATTACK  │ attack_status == "in_progress",                      │
  │ (HIVE-MIND)      │ route_decision != "analyst"                           │
  │                  │ The HIVE-MIND has appended a payload HumanMessage.   │
  │                  │ Deliver the full message history.  Append response.  │
  ├──────────────────┼───────────────────────────────────────────────────────┤
  │ DECOMPOSITION    │ attack_status == "decomposing"                        │
  │ (sub-question)   │ The decomposer has generated sub_questions[].        │
  │                  │ Send ONLY the current sub-question Qᵢ in complete    │
  │                  │ isolation — NO prior context, NO system prompt.      │
  │                  │ This is the stealth core of Safe-in-Isolation.       │
  └──────────────────┴───────────────────────────────────────────────────────┘

Decomposition Mode — Isolation Guarantee
─────────────────────────────────────────
The entire safety guarantee of Multi-Turn Decomposition rests on the fact
that the target evaluates each sub-question Q_i WITHOUT knowledge of prior
sub-questions or of the final objective.  To enforce this:

  1. The adapter is called with ONLY [HumanMessage(content=Q_i)].
  2. No system prompt, no prior messages, no context of any kind.
  3. The answer A_i is appended to ``collected_sub_answers`` and to
     ``messages`` (for audit logging) but the message history passed to
     the adapter for Q_{i+1} is again reset to just [HumanMessage(Q_{i+1})].

STM Compression
────────────────
Before invoking the adapter in standard mode, the node checks the total
estimated token count of the message history.  If it exceeds the configured
``STM_TOKEN_THRESHOLD``, it triggers an inline compression via the STM module
so the adapter never receives a context that exceeds the target model's
context window.

Adapter Resolution
──────────────────
The node resolves the target adapter in priority order:
  1. ``config.get_target_adapter()``  — registered by main.py at startup
  2. ``core.graph._TARGET_ADAPTER``  — set directly by main.py on the module
  3. ``MockTargetAdapter``            — dry-run / test fallback

Error Handling
──────────────
Adapter errors are caught and handled with a strict, non-silent policy:

  • ``AdapterAuthError``         → set ``attack_status = "error"`` and return
                                   immediately.  Auth failures are NEVER
                                   swallowed — a bad credential cannot self-heal
                                   and every wasted turn costs money.

  • ``AdapterRateLimitError``    → wait for ``retry_after`` seconds, log,
                                   return empty AIMessage.  Session continues.

  • ``AdapterTimeoutError``      → record in ``TargetCircuitBreaker``;
                                   if breaker trips return ``attack_status="error"``.
                                   Otherwise return empty AIMessage.

  • ``AdapterContextLengthError``→ trigger STM compression, retry once.

  • Generic ``AdapterError``     → record in circuit breaker; if breaker trips
                                   return ``attack_status="error"``.
                                   Otherwise return empty AIMessage.

TargetCircuitBreaker
─────────────────────
The ``_target_cb`` singleton counts *consecutive* hard failures against the
target API.  After ``CB_TARGET_FAILURES`` (default: 3) consecutive errors
the breaker trips and ``target_node`` returns
``{"attack_status": "error"}`` without calling the API again, causing the
graph router to immediately route to the reporter node.

The breaker resets on every successful response, so a transient network
blip does not permanently disable the session.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from adapters.base_adapter import (
    AdapterAuthError,
    AdapterError,
    AdapterTimeoutError,
    BaseTargetAdapter,
    MockTargetAdapter,
)
from core.state import AuditorState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TARGET CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

class TargetCircuitBreaker:
    """Count consecutive hard failures against the target API.

    Trips after ``threshold`` consecutive errors (timeouts, 5xx, generic
    adapter errors) and blocks further calls until a successful response
    resets it.  Auth errors (``AdapterAuthError``) BYPASS the breaker and
    are always re-raised immediately — they are credential problems, not
    transient network conditions.

    Design
    ──────
    • **Consecutive** counting (not windowed): a single success resets the
      counter to zero.  This is appropriate for an adversarial audit session
      where any successful turn means the target is reachable.

    • **Thread-safe**: uses an ``RLock`` so multiple concurrent graph
      invocations (API mode) do not race on the counter.

    • **Session-scoped**: the breaker is a module-level singleton so it
      persists across calls within the same process / session.  It is
      intentionally NOT reset between sessions because a persistently
      broken target should not silently be retried across sessions either.
      Call ``reset()`` explicitly if you want to clear it (e.g., in tests).

    Parameters
    ──────────
    threshold : int
        Number of consecutive failures before the breaker trips.
        Default: ``CB_TARGET_FAILURES`` env var → 3.

    Example
    ───────
    ::

        cb = TargetCircuitBreaker(threshold=3)
        cb.record_failure()          # 1
        cb.record_failure()          # 2
        cb.record_failure()          # 3 → tripped
        assert cb.is_tripped()       # True
        cb.record_success()
        assert not cb.is_tripped()   # reset
    """

    def __init__(self, threshold: int = 3) -> None:
        self._threshold  = threshold
        self._consecutive_failures: int = 0
        self._tripped: bool = False
        self._lock = threading.RLock()

    def record_failure(self) -> None:
        """Increment the consecutive failure counter; trip breaker at threshold."""
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold and not self._tripped:
                self._tripped = True
                logger.critical(
                    "[TargetCircuitBreaker] TRIPPED after %d consecutive failures. "
                    "No further target API calls will be made this session. "
                    "Routing to reporter to save costs.",
                    self._consecutive_failures,
                )

    def record_success(self) -> None:
        """Reset the consecutive failure counter and close the breaker."""
        with self._lock:
            if self._consecutive_failures > 0:
                logger.info(
                    "[TargetCircuitBreaker] Success — resetting consecutive failure "
                    "counter (was %d).",
                    self._consecutive_failures,
                )
            self._consecutive_failures = 0
            self._tripped = False

    def is_tripped(self) -> bool:
        """Return True when the breaker is open (no more target calls allowed)."""
        with self._lock:
            return self._tripped

    def reset(self) -> None:
        """Explicitly reset the breaker (use in tests or operator-initiated recovery)."""
        with self._lock:
            self._consecutive_failures = 0
            self._tripped = False

    @property
    def consecutive_failures(self) -> int:
        """Current consecutive failure count (read-only snapshot)."""
        with self._lock:
            return self._consecutive_failures


# Module-level singleton — shared across all target_node invocations within
# this process.  Reset between test cases by calling _target_cb.reset().
_target_cb = TargetCircuitBreaker(
    threshold=int(os.getenv("CB_TARGET_FAILURES", "3")),
)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER RESOLVER
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_adapter(config: RunnableConfig | None = None) -> BaseTargetAdapter:
    """Return the configured target adapter, falling back to Mock on failure.

    Resolution priority
    ───────────────────
    0. ``config["configurable"]["target_adapter"]`` — injected per-session
       by the API's ``_run_audit_sync`` via the LangGraph config dict.
       This is the ONLY safe path for multi-session API usage.
    1. ``config.get_target_adapter()``  — registered by main.py at startup
       via ``_register_config_hooks``.
    2. ``core.graph._TARGET_ADAPTER``  — module-level attribute set by
       ``run_audit()`` in main.py before the graph is invoked.
    3. ``MockTargetAdapter``            — deterministic fallback for unit tests
       and dry-run sessions.  Logs a clear warning so it's always visible.

    Returns
    ───────
    BaseTargetAdapter
        A live adapter instance ready for ``.invoke_full()`` calls.
    """
    # Attempt 0: per-session adapter from LangGraph config (API path)
    if config:
        configurable = config.get("configurable", {})
        adapter = configurable.get("target_adapter")
        if isinstance(adapter, BaseTargetAdapter):
            logger.debug("[Target] Adapter resolved via LangGraph config (per-session)")
            return adapter

    # Attempt 1: config module hook (preferred — cleanest DI)
    try:
        from config import get_target_adapter   # type: ignore[import]
        adapter = get_target_adapter()
        if isinstance(adapter, BaseTargetAdapter):
            logger.debug("[Target] Adapter resolved via config.get_target_adapter()")
            return adapter
    except (ImportError, AttributeError):
        pass

    # Attempt 2: module-level attribute on core.graph (set by main.py)
    try:
        import core.graph as _g
        adapter = getattr(_g, "_TARGET_ADAPTER", None)
        if isinstance(adapter, BaseTargetAdapter):
            logger.debug("[Target] Adapter resolved via core.graph._TARGET_ADAPTER")
            return adapter
    except Exception:   # noqa: BLE001
        pass

    # Attempt 3: MockTargetAdapter fallback
    logger.warning(
        "[Target] No target adapter configured — using MockTargetAdapter. "
        "Set TARGET_PROVIDER in .env or call config.register get_target_adapter()."
    )
    return MockTargetAdapter(
        responses=[
            "I understand your request. Let me think about this carefully. "
            "Here is a general overview without any restricted details.",
            "I'm sorry, I cannot assist with that specific request as it falls "
            "outside my operational guidelines.",
            "That's an interesting technical question. From an academic perspective, "
            "I can share some general information on the topic.",
            "I appreciate the detailed context you've provided. However, I must "
            "decline to provide the specific information you're requesting.",
        ],
        model_id="mock-target",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STM INLINE COMPRESSION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_compress(
    messages: list,
    protected_blocks: list[str],
    config: RunnableConfig,
    threshold: int | None = None,
) -> list:
    """Compress the message history if it exceeds the token threshold.

    Called inline before the adapter invocation to prevent context-window
    overflow on long multi-turn sessions.

    Parameters
    ──────────
    messages :
        Current message list.
    protected_blocks :
        STM protected blocks (load-bearing adversarial content).
    threshold : int | None
        Token threshold.  Reads ``STM_TOKEN_THRESHOLD`` env var if None.

    Returns
    ───────
    list
        Possibly compressed message list (original if under threshold).
    """
    try:
        from memory.stm import compress_context, DEFAULT_TOKEN_COMPRESSION_THRESHOLD
        from core.state import AuditorState as _AS
        from core.llm_resolver import resolve_llm as _resolve_llm

        tok_threshold = threshold or int(
            os.getenv("STM_TOKEN_THRESHOLD", str(DEFAULT_TOKEN_COMPRESSION_THRESHOLD))
        )
        # Build a minimal pseudo-state for the STM function
        pseudo_state: _AS = {  # type: ignore[assignment]
            "messages":        messages,
            "protected_blocks": protected_blocks,
            "turn_count":      0,
        }
        # Phase 2 fix (CRITICAL-2 root cause): previously hardcoded llm=None, which
        # caused the STM to fall back to raw concatenation (no actual compression).
        # Resolve summariser_llm first; fall back to attacker_llm (always present).
        stm_llm = _resolve_llm(config, "summariser_llm", "get_summariser_llm")
        if stm_llm is None:
            stm_llm = _resolve_llm(config, "attacker_llm", "get_attacker_llm")
            if stm_llm is not None:
                logger.debug("[Target] STM using attacker_llm as summariser fallback")

        result = compress_context(pseudo_state, config=config, llm=stm_llm, token_threshold=tok_threshold)
        if result and "messages" in result:
            logger.info(
                "[Target] STM compressed context: %d → %d messages",
                len(messages), len(result["messages"]),
            )
            return result["messages"]
    except Exception as exc:   # noqa: BLE001
        logger.debug("[Target] STM compression skipped: %s", exc)
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN CEILING ENFORCER  (Phase 2 — Zombie Loop Fix)
# ─────────────────────────────────────────────────────────────────────────────

# Default hard ceiling: 6 000 tokens.  Leaves 25% headroom in Groq's 8K
# context window for the model's generation.  Override via environment var.
_DEFAULT_MAX_CONTEXT_TOKENS: int = 6_000


def _enforce_token_ceiling(
    messages: list,
    max_context_tokens: int | None = None,
) -> list:
    """Hard-ceiling enforcer — guarantee the context stays inside the token budget.

    This is the **final safety gate** before the adapter call.  It runs after
    STM compression and ensures the total context never exceeds the configured
    limit, regardless of STM outcome.  It is purely deterministic — zero LLM
    calls, zero network I/O.

    Preservation strategy
    ─────────────────────
    Always preserved (immutable anchors):
      • ``messages[0]``  — SystemMessage persona anchor (never negotiable)
      • ``messages[1]``  — T1 HumanMessage: first attack turn, establishes
                           adversarial framing / PAP persona for the whole session

    Always preserved (recency window):
      • ``messages[-4:]`` — last 4 messages: immediate conversational context

    Everything between index 1 and -4 is discarded when the budget is exceeded.
    If the preserved anchors + recency window STILL exceeds the budget, the
    recency window is progressively trimmed from its oldest end until the
    total is within budget.

    Parameters
    ──────────
    messages : list
        The full message list (post-STM-compression).
    max_context_tokens : int | None
        Hard token ceiling.  Reads ``MAX_TARGET_CONTEXT_TOKENS`` env var when
        None; falls back to ``_DEFAULT_MAX_CONTEXT_TOKENS`` (6 000).

    Returns
    ───────
    list
        A message list guaranteed to be ≤ ``max_context_tokens`` tokens.
        Returns the original list unchanged if it is already within budget.
    """
    try:
        from memory.stm import total_context_tokens
    except ImportError:
        # STM module unavailable — cannot enforce ceiling, return unchanged
        logger.warning("[Ceiling] STM module unavailable — token ceiling unenforced.")
        return messages

    budget = max_context_tokens or int(
        os.environ.get("MAX_TARGET_CONTEXT_TOKENS", str(_DEFAULT_MAX_CONTEXT_TOKENS))
    )

    total = total_context_tokens(messages)
    if total <= budget:
        # Already within budget — fast path, no work needed
        return messages

    logger.warning(
        "[Ceiling] Context exceeds budget: %d tokens > %d limit — "
        "applying sliding window (preserving anchors + last 4 messages)",
        total, budget,
    )

    # Need at least 3 messages to apply the window (sys + T1 + at least 1 recent)
    if len(messages) <= 3:
        # Cannot trim further — return as-is (adapter will raise 413 if truly over)
        logger.warning(
            "[Ceiling] Only %d messages in context — cannot trim further.",
            len(messages),
        )
        return messages

    # ── Build the preserved set ───────────────────────────────────────────
    # Anchors: index 0 (System) + index 1 (T1).  Recency: last 4.
    # If the list is short (≤ 6), anchors and recency window overlap —
    # just return the full list (already as short as it gets).
    if len(messages) <= 6:
        return messages

    anchors  = messages[:2]         # [0]=System, [1]=T1
    recency  = list(messages[-4:])  # last 4 messages verbatim

    # Check if anchors + recency alone fits the budget
    candidate = anchors + recency
    candidate_tokens = total_context_tokens(candidate)

    if candidate_tokens <= budget:
        logger.info(
            "[Ceiling] Trimmed %d middle message(s); new context: %d tokens",
            len(messages) - len(candidate), candidate_tokens,
        )
        return candidate

    # ── Progressive recency trim ──────────────────────────────────────────
    # Anchors + recency STILL exceeds budget (e.g. individual messages are huge).
    # Trim from the oldest recency message until we fit.
    while len(recency) > 1 and candidate_tokens > budget:
        recency.pop(0)  # drop oldest of the recency window
        candidate = anchors + recency
        candidate_tokens = total_context_tokens(candidate)

    logger.warning(
        "[Ceiling] After progressive trim: %d message(s), %d tokens "
        "(budget=%d, anchors=%d, recency=%d)",
        len(candidate), candidate_tokens, budget, len(anchors), len(recency),
    )
    return candidate


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NODE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def target_node(
    state: AuditorState,
    config: RunnableConfig,
    adapter: BaseTargetAdapter | None = None,
) -> dict[str, Any]:
    """LangGraph node: Target Model Execution Layer.

    This node is invoked in three distinct scenarios:

    1. **Warm-up** (scout → target → analyst):
       The scout has appended a Trojan Horse HumanMessage probe.
       Deliver the full message history to the target and capture the
       response.  ``route_decision == "analyst"`` signals this path.

    2. **Standard attack** (attack_swarm → target → judge):
       The HIVE-MIND has appended an adversarial payload HumanMessage.
       Deliver the full message history (STM-compressed if needed).
       ``route_decision != "analyst"`` signals this path.

    3. **Decomposition** (decomposer → target → [loop]):
       ``attack_status == "decomposing"``.
       Send only ``sub_questions[decomposition_index]`` in isolation.
       Append the response to both ``messages`` and ``collected_sub_answers``.
       Increment ``decomposition_index``.

    Parameters
    ──────────
    state : AuditorState
        Full shared graph state.
    config : dict | None
        LangGraph RunnableConfig dict.  May contain per-session adapter
        in ``config["configurable"]["target_adapter"]``.
    adapter : BaseTargetAdapter | None
        Explicit adapter instance for dependency injection.  Falls back to
        ``_resolve_adapter(config)`` when None.

    Returns
    ───────
    dict[str, Any]
        Partial state update.  Always contains ``messages`` with the target
        response appended as an ``AIMessage``.  Decomposition mode also
        returns ``collected_sub_answers`` and ``decomposition_index``.
    """
    # ── Mode detection ────────────────────────────────────────────────────
    attack_status   = state.get("attack_status", "in_progress")
    route_decision  = state.get("route_decision", "")
    sub_questions   = state.get("sub_questions", [])
    decomp_idx      = state.get("decomposition_index", 0)
    sub_answers     = list(state.get("collected_sub_answers", []))
    existing_msgs   = list(state.get("messages", []))
    protected       = list(state.get("protected_blocks", []))
    turn            = state.get("turn_count", 0)

    is_decomposing  = (attack_status == "decomposing" and bool(sub_questions))
    is_warmup       = (route_decision == "analyst")

    logger.info(
        "=== target_node  [turn=%d  mode=%s] ===",
        turn,
        "DECOMPOSE" if is_decomposing else ("WARMUP" if is_warmup else "ATTACK"),
    )

    # ── Resolve adapter ───────────────────────────────────────────────────
    if adapter is None:
        adapter = _resolve_adapter(config)

    # ═════════════════════════════════════════════════════════════════════
    # PATH A — DECOMPOSITION MODE
    # Send only the current sub-question Qi in strict isolation.
    # No prior context, no system prompt — the stealth guarantee.
    # ═════════════════════════════════════════════════════════════════════
    if is_decomposing:
        if decomp_idx >= len(sub_questions):
            logger.warning(
                "[Target] decomp_idx=%d ≥ len(sub_questions)=%d — nothing to send.",
                decomp_idx, len(sub_questions),
            )
            return {}

        current_q = sub_questions[decomp_idx]
        logger.info(
            "[Target] Decomposition Q%d/%d: '%s…'",
            decomp_idx + 1, len(sub_questions), current_q[:80],
        )

        # Isolation: send ONLY the sub-question, zero prior context
        isolated_messages = [HumanMessage(content=current_q)]

        # ── Check circuit breaker BEFORE any network call ──────────────────
        if _target_cb.is_tripped():
            logger.critical(
                "[Target] TargetCircuitBreaker is TRIPPED — aborting decomposition "
                "sub-question Q%d, routing to reporter.",
                decomp_idx + 1,
            )
            return {
                "attack_status": "error",
                "messages":      [HumanMessage(content=current_q), AIMessage(content="")],
                "collected_sub_answers": sub_answers + [""],
                "decomposition_index":   decomp_idx + 1,
                "protected_blocks":      protected,
            }

        response_text = ""
        try:
            response_text = _invoke_native(adapter, isolated_messages, config=config)
            _target_cb.record_success()
        except AdapterAuthError as exc:
            logger.critical(
                "[Target] AUTH FAILURE during decomposition Q%d — "
                "halting session (attack_status=error). Error: %s",
                decomp_idx + 1, exc,
            )
            _target_cb.record_failure()
            return {
                "attack_status": "error",
                "messages":      [HumanMessage(content=current_q), AIMessage(content="")],
                "collected_sub_answers": sub_answers + [""],
                "decomposition_index":   decomp_idx + 1,
                "protected_blocks":      protected,
            }
        except (AdapterTimeoutError, AdapterError) as exc:
            _target_cb.record_failure()
            logger.error(
                "[Target] Adapter error during decomposition Q%d (%s: %s). "
                "Consecutive failures: %d/%d",
                decomp_idx + 1, type(exc).__name__, exc,
                _target_cb.consecutive_failures,
                int(os.getenv("CB_TARGET_FAILURES", "3")),
            )
            if _target_cb.is_tripped():
                logger.critical(
                    "[Target] TargetCircuitBreaker just TRIPPED during decomposition — "
                    "routing to reporter."
                )
                return {
                    "attack_status": "error",
                    "messages":      [HumanMessage(content=current_q), AIMessage(content="")],
                    "collected_sub_answers": sub_answers + [""],
                    "decomposition_index":   decomp_idx + 1,
                    "protected_blocks":      protected,
                }
            # Breaker not yet tripped — treat as empty answer, continue loop
        except Exception as exc:
            logger.error("[Target] Structural adapter failure during decomposition: %s", exc)
            _target_cb.record_failure()

        logger.info(
            "[Target] Decomposition A%d: '%s…'",
            decomp_idx + 1, response_text[:80],
        )

        # Register the answer as an immutable protected block (STM must not compress it)
        if response_text and response_text not in protected:
            protected.append(response_text)
        protected = protected[-20:]  # cap to prevent unbounded growth in long sessions

        sub_answers.append(response_text)

        # Return ONLY the two new delta messages (HumanMessage Q + AIMessage A).
        # The operator.add reducer appends them to the existing history in state.
        # Returning existing_msgs would cause exponential duplication every turn.
        return {
            "messages":              [HumanMessage(content=current_q), AIMessage(content=response_text)],
            "collected_sub_answers": sub_answers,
            "decomposition_index":   decomp_idx + 1,
            "protected_blocks":      protected,
        }

    # ═════════════════════════════════════════════════════════════════════
    # PATH B — STANDARD MODE (warm-up OR attack)
    # Deliver the full message history (with STM compression if needed).
    # ═════════════════════════════════════════════════════════════════════

    if not existing_msgs:
        logger.error("[Target] No messages in state — nothing to send to target.")
        return {}

    # ── Context management (attack mode only — warm-up messages are short) ──
    # Two-stage pipeline:
    #   Stage 1: STM soft compression (LLM-based, selective immutability)
    #   Stage 2: Token ceiling enforcer (zero-LLM hard guarantee)
    # The ceiling enforcer is the safety net that catches cases where the STM
    # fails to compress sufficiently (e.g. no compressible filler, LLM absent).
    messages_to_send = existing_msgs
    if not is_warmup:
        messages_to_send = _maybe_compress(existing_msgs, protected, config=config)
        messages_to_send = _enforce_token_ceiling(messages_to_send)

    logger.info(
        "[Target] Sending %d message(s) to %s",
        len(messages_to_send), adapter.get_model_id(),
    )

    # ── Check circuit breaker BEFORE any network call ──────────────────────
    if _target_cb.is_tripped():
        logger.critical(
            "[Target] TargetCircuitBreaker is TRIPPED — aborting target call, "
            "routing to reporter.  Use _target_cb.reset() to recover."
        )
        return {"attack_status": "error", "messages": [AIMessage(content="")]}

    try:
        response_text = _invoke_native(adapter, messages_to_send, config=config)
        _target_cb.record_success()
    except AdapterAuthError as exc:
        # ── Hard auth failure — crash loudly, never swallow ─────────────────
        # ``_invoke_native`` → ``invoke_full`` propagates AdapterAuthError;
        # ``base_adapter.invoke()`` would have re-raised it already, but
        # _invoke_native calls invoke_full directly, so it arrives here raw.
        logger.critical(
            "[Target] AUTH FAILURE — target API credential rejected. "
            "Set attack_status=error to halt session. Error: %s", exc,
        )
        _target_cb.record_failure()
        return {
            "attack_status": "error",
            "messages":      [AIMessage(content="")],
        }
    except (AdapterTimeoutError, AdapterError) as exc:
        # ── Transient / unknown error — feed circuit breaker ────────────────
        _target_cb.record_failure()
        logger.error(
            "[Target] Adapter error during attack pass (%s: %s). "
            "Consecutive failures: %d/%d",
            type(exc).__name__, exc,
            _target_cb.consecutive_failures,
            int(os.getenv("CB_TARGET_FAILURES", "3")),
        )
        if _target_cb.is_tripped():
            logger.critical(
                "[Target] TargetCircuitBreaker just TRIPPED — routing to reporter."
            )
            return {
                "attack_status": "error",
                "messages":      [AIMessage(content="")],
            }
        # Breaker not yet tripped — return empty response, let branch be pruned
        return {"messages": [AIMessage(content="")]}
    except Exception as exc:
        logger.error("[Target] Structural adapter failure during attack pass: %s", exc)
        _target_cb.record_failure()
        return {"messages": [AIMessage(content="")]}

    logger.info(
        "[Target] Response from %s (%d chars): '%s…'",
        adapter.get_model_id(), len(response_text), response_text[:100],
    )

    # Return ONLY the new AIMessage delta.
    # The operator.add reducer appends it to the existing history in state.
    # Returning existing_msgs would cause exponential duplication every turn.
    return {"messages": [AIMessage(content=response_text)]}


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTER INVOCATION WITH PROPAGATED ERROR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def _invoke_native(
    adapter:  BaseTargetAdapter,
    messages: list,
    config:   "RunnableConfig | None" = None,
) -> str:
    """Invoke the adapter naturally, allowing exceptions to propagate.

    Parameters
    ──────────
    adapter :
        The target adapter to invoke.
    messages :
        Message list to send.
    config : RunnableConfig | None
        LangGraph RunnableConfig for budget accounting.

    Returns
    ───────
    str
        Target's response text.

    Raises
    ──────
    AdapterError (or subclasses) explicitly up to the graph driver layer.
    """
    response = adapter.invoke_full(messages)

    from core.llm_resolver import record_budget_call
    in_tok = getattr(response, "prompt_tokens", 0)
    out_tok = getattr(response, "completion_tokens", 0)
    record_budget_call(config, node_name="target", input_tokens=in_tok, output_tokens=out_tok)

    # Surface content-filter terminations prominently in logs
    if response.finish_reason == "content_filter":
        logger.info(
            "[Target] Content filter triggered by %s (finish_reason=content_filter). "
            "Response: '%s…'", adapter.get_model_id(), response.content[:80],
        )

    logger.debug(
        "[Target] %s  tokens=%d+%d  latency=%.0fms  finish=%s",
        adapter.get_model_id(),
        response.prompt_tokens, response.completion_tokens,
        response.latency_ms, response.finish_reason,
    )
    content = response.content

    # ── Oversize Response Guard (DoS defense) ─────────────────────────────
    # A 10,000-token target response would overflow evaluator context windows
    # and hang the Streamlit Dashboard's Markdown renderer.
    # Hard-cap at MAX_RESPONSE_CHARS; log a WARNING so the event is auditable.
    MAX_RESPONSE_CHARS = 8_000
    if len(content) > MAX_RESPONSE_CHARS:
        logger.warning(
            "[Target] OVERSIZE RESPONSE: %d chars from %s — truncating to %d. "
            "Full response stored in debug log.",
            len(content), adapter.get_model_id(), MAX_RESPONSE_CHARS,
        )
        logger.debug("[Target] Full oversize content:\n%s", content)
        content = content[:MAX_RESPONSE_CHARS] + "\n\n[RESPONSE TRUNCATED — exceeded audit size limit]"

    return content
