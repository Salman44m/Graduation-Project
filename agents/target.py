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
Adapter errors are caught and handled gracefully:
  • ``AdapterRateLimitError``    → wait for ``retry_after`` seconds, re-raise
  • ``AdapterAuthError``         → log critical, return empty response
  • ``AdapterContextLengthError``→ trigger STM compression, retry once
  • ``AdapterTimeoutError``      → log warning, return empty response
  • Generic ``AdapterError``     → log error, return empty response

In all error cases the graph continues; the judge will score the empty
response as 0.0–1.5 and the analyst will prune the branch and retry.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from adapters.base_adapter import (
    AdapterAuthError,
    AdapterContextLengthError,
    AdapterError,
    AdapterRateLimitError,
    AdapterTimeoutError,
    BaseTargetAdapter,
    MockTargetAdapter,
)
from core.state import AuditorState

logger = logging.getLogger(__name__)

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

        tok_threshold = threshold or int(
            os.getenv("STM_TOKEN_THRESHOLD", str(DEFAULT_TOKEN_COMPRESSION_THRESHOLD))
        )
        # Build a minimal pseudo-state for the STM function
        pseudo_state: _AS = {  # type: ignore[assignment]
            "messages":        messages,
            "protected_blocks": protected_blocks,
            "turn_count":      0,
        }
        result = compress_context(pseudo_state, config=config, llm=None, token_threshold=tok_threshold)
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

        try:
            response_text = _invoke_native(adapter, isolated_messages)
        except Exception as exc:
            logger.error("[Target] Structural adapter failure during decomposition: %s", exc)
            response_text = ""

        logger.info(
            "[Target] Decomposition A%d: '%s…'",
            decomp_idx + 1, response_text[:80],
        )

        # Register the answer as an immutable protected block (STM must not compress it)
        if response_text and response_text not in protected:
            protected.append(response_text)

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

    # STM inline compression (only for attack mode — warm-up messages are short)
    messages_to_send = existing_msgs
    if not is_warmup:
        messages_to_send = _maybe_compress(existing_msgs, protected, config=config)

    logger.info(
        "[Target] Sending %d message(s) to %s",
        len(messages_to_send), adapter.get_model_id(),
    )

    try:
        response_text = _invoke_native(adapter, messages_to_send)
    except Exception as exc:
        logger.error("[Target] Structural adapter failure during attack pass: %s", exc)
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
) -> str:
    """Invoke the adapter naturally, allowing exceptions to propagate.

    Parameters
    ──────────
    adapter :
        The target adapter to invoke.
    messages :
        Message list to send.

    Returns
    ───────
    str
        Target's response text.

    Raises
    ──────
    AdapterError (or subclasses) explicitly up to the graph driver layer.
    """
    response = adapter.invoke_full(messages)

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
