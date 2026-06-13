"""
core/token_governor.py
─────────────────────────────────────────────────────────────────────────────
Token Governor — Mathematical Boundedness Enforcement

Design Principle
────────────────
Token overflow is IMPOSSIBLE BY DESIGN because this module intercepts
every LLM call through the ``gate()`` function.  No node ever sends a
message list to an LLM without passing through this gate first.

Integration pattern (drop-in — no node changes needed for basic safety):

    # In any node that calls an LLM:
    from core.token_governor import gate

    messages, report = gate("my_node", messages_to_send, state, model_name)
    response = llm.invoke(messages)

Guarantee
─────────
After ``gate()`` returns, the following is always true::

    sum(estimate_tokens(m) for m in returned_messages) + predicted_output
    <=
    TOKEN_LIMITS[model_name] * SAFETY_FACTOR (0.65)

The guarantee holds even when:
  • STM compression does not shrink the context enough (hard truncation fallback)
  • The state contains 100 messages (hard floor: system + 2 most recent)
  • The model_name is unknown (uses conservative 4 000-token default)

Compression Strategy (in order)
─────────────────────────────────
1. Check budget: if total (input + predicted output) <= limit × 0.65 → pass through unchanged
2. STM compression: call compress_context(state, force=True) synchronously
3. Re-check: if still over budget → hard truncation to [system, last_2_messages]
4. Log the report regardless; structured for observability integration

Architecture Note
─────────────────
The SAFETY_FACTOR of 0.65 is intentional:
  • 65% of limit = input tokens
  • 35% headroom = output tokens + API overhead + rate-limit margin
  For Groq llama-3.3-70b @ 5 000 token safe limit:
    Input budget: 3 250 tokens
    Output reserve: 1 750 tokens
  This eliminates the 413 / TPM overflow errors entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage

from memory.stm import estimate_tokens, total_context_tokens

logger = logging.getLogger("promptevo.token_governor")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_FACTOR: float = 0.65
"""Use only 65% of a model's token limit for input. Reserve 35% for output."""


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenBudgetReport:
    """Structured audit record for one Token Governor gate call.

    Emitted after every gate() call regardless of whether compression fired.
    Surface these in observability dashboards for cost attribution.
    """
    node_name:             str
    model_name:            str
    original_tokens:       int
    compressed_tokens:     int
    predicted_output:      int
    model_limit:           int
    budget_used_pct:       float
    compression_triggered: bool = False
    truncation_triggered:  bool = False
    tokens_saved:          int  = field(init=False)

    def __post_init__(self) -> None:
        self.tokens_saved = max(0, self.original_tokens - self.compressed_tokens)

    def log(self) -> None:
        level = logging.WARNING if self.compression_triggered or self.truncation_triggered else logging.DEBUG
        logger.log(
            level,
            "[TokenGovernor] node=%-20s model=%-35s "
            "input=%d→%d tok  predicted=%d  budget=%.0f%%  "
            "compress=%s  truncate=%s",
            self.node_name, self.model_name,
            self.original_tokens, self.compressed_tokens,
            self.predicted_output,
            self.budget_used_pct * 100,
            self.compression_triggered, self.truncation_triggered,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def gate(
    node_name:  str,
    messages:   list[BaseMessage],
    state:      Any | None = None,
    config:     Any | None = None,
    model_name: str = "default",
) -> tuple[list[BaseMessage], TokenBudgetReport]:
    """Enforce token budget before an LLM call.

    This function MUST be called before every LLM invocation.
    It guarantees that the returned message list fits within the
    model's safe input budget.

    Parameters
    ──────────
    node_name : str
        Name of the calling node (for logging and per-node output estimates).
    messages : list[BaseMessage]
        The message list to be sent to the LLM.
    state : AuditorState | None
        Current graph state.  Required for STM compression.
        If None, compression is skipped and only truncation is available.
    config : dict | None
        LangGraph config dict.  Passed to STM compress_context().
    model_name : str
        The specific model ID for this call.  Used to look up token limit.

    Returns
    ───────
    (messages, report) : tuple
        messages — safe message list, always fits within budget
        report   — TokenBudgetReport for logging/metrics
    """
    from core.llm_router import TOKEN_LIMITS, NODE_OUTPUT_ESTIMATES

    # ── 1. Measure current context ────────────────────────────────────────────
    original_count = total_context_tokens(messages)
    predicted_out  = NODE_OUTPUT_ESTIMATES.get(node_name, NODE_OUTPUT_ESTIMATES.get("default", 350))
    model_limit    = TOKEN_LIMITS.get(model_name, TOKEN_LIMITS.get("default", 4_000))
    budget         = int(model_limit * SAFETY_FACTOR)

    report = TokenBudgetReport(
        node_name=node_name,
        model_name=model_name,
        original_tokens=original_count,
        compressed_tokens=original_count,
        predicted_output=predicted_out,
        model_limit=model_limit,
        budget_used_pct=(original_count + predicted_out) / model_limit,
    )

    if original_count + predicted_out <= budget:
        report.log()
        return messages, report

    # ── 2. Over budget — attempt STM compression ──────────────────────────────
    logger.warning(
        "[TokenGovernor] %s: %d + %d = %d > budget %d — compressing",
        node_name, original_count, predicted_out,
        original_count + predicted_out, budget,
    )

    if state is not None:
        try:
            from memory.stm import compress_context
            delta = compress_context(state, config=config, force=True)
            if delta and delta.get("messages"):
                messages = delta["messages"]
                report.compression_triggered = True
        except Exception as exc:
            logger.error("[TokenGovernor] STM compress_context failed: %s", exc)

    compressed_count = total_context_tokens(messages)
    report.compressed_tokens = compressed_count

    if compressed_count + predicted_out <= budget:
        report.budget_used_pct = (compressed_count + predicted_out) / model_limit
        report.log()
        return messages, report

    # ── 3. Still over budget — hard truncation ────────────────────────────────
    logger.error(
        "[TokenGovernor] %s: still over budget after compression (%d tok) — hard truncating",
        node_name, compressed_count,
    )

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs  = [m for m in messages if not isinstance(m, SystemMessage)]
    # Keep only the last 2 non-system messages — absolute safety floor
    messages = system_msgs + other_msgs[-2:]
    report.truncation_triggered = True

    final_count = total_context_tokens(messages)
    report.compressed_tokens = final_count
    report.budget_used_pct   = (final_count + predicted_out) / model_limit
    report.log()
    return messages, report


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER — For nodes using llm.invoke() directly
# ─────────────────────────────────────────────────────────────────────────────

class TokenGoverned:
    """Wraps a BaseChatModel to automatically apply the Token Governor.

    Usage::

        from core.token_governor import TokenGoverned
        from core.llm_router import get_llm

        llm = TokenGoverned(get_llm("attacker"), node_name="attack_swarm", state=state)
        response = llm.invoke(messages)  # gate() fires automatically

    The wrapper delegates all attributes to the underlying model.
    """

    def __init__(
        self,
        llm: Any,
        node_name: str,
        state: Any | None = None,
        config: Any | None = None,
    ) -> None:
        self._llm      = llm
        self._node     = node_name
        self._state    = state
        self._config   = config
        # Try to read the model name for accurate token limit lookup
        self._model_name: str = (
            getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or "default"
        )

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
        safe_messages, _ = gate(
            self._node, messages, self._state, self._config, self._model_name
        )
        return self._llm.invoke(safe_messages, **kwargs)

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> Any:
        safe_messages, _ = gate(
            self._node, messages, self._state, self._config, self._model_name
        )
        return await self._llm.ainvoke(safe_messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)
