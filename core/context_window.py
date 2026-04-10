"""
core/context_window.py
─────────────────────────────────────────────────────────────────────────────
JIT Sliding Window Utility — Evaluator Context Adapter

Architectural Role
──────────────────
The ``AuditorState.messages`` field is an append-only audit ledger that grows
indefinitely throughout a session.  Passing the full history to internal LLMs
(Judge, Analyst, Hive-Mind) causes two problems:

  1. **Token exhaustion** — evaluator LLMs breach their context window on long
     sessions, causing truncation errors or "lost in the middle" degradation.
  2. **Signal dilution** — distant turns from the warm-up phase confuse the
     Judge and Hive-Mind, which care only about *recent* performance.

This module provides a single utility function, ``get_evaluator_context``,
that each internal agent calls *just before* invoking its LLM.  The full
``messages`` list in state is never modified — the Dashboard and Reporter
always read the complete, unclipped audit trail.

Usage
─────
    from core.context_window import get_evaluator_context

    # Inside an agent node (e.g. hive_mind_node, red_debate_judge_swarm):
    full_messages = list(state.get("messages", []))
    context = get_evaluator_context(full_messages, max_pairs=3)
    response = llm.invoke([system_msg] + context + [user_msg])

    # The returned `context` is a *read-only slice* — do NOT write it back
    # to state["messages"].  Only return the new delta message.

Design Decisions
────────────────
* **SystemMessage preservation**: If the session begins with a SystemMessage
  (e.g., the target's system prompt or a PAP persona), it is always kept at
  index 0 of the returned context.  Stripping it would corrupt the target's
  identity framing for the evaluator.

* **Pair-based windowing**: A "pair" is one (HumanMessage, AIMessage) exchange.
  Slicing by pairs is more semantically correct than slicing by raw message
  count, since the ratio of Human to AI messages is not always 1:1 (especially
  during decomposition mode where each sub-question sends one HumanMessage).

* **Safe short-history handling**: If fewer pairs exist than ``max_pairs``,
  the function returns all available messages without raising IndexError.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage


def get_evaluator_context(
    messages: list[BaseMessage],
    max_pairs: int = 3,
) -> list[BaseMessage]:
    """Return a bounded, evaluator-safe slice of the full message history.

    Extracts at most the last ``max_pairs`` (Human + AI) exchange pairs from
    ``messages``, always preserving any leading ``SystemMessage``.  This gives
    internal LLMs (Judge, Hive-Mind, Analyst) a focused, token-bounded context
    without altering the permanent audit ledger stored in ``AuditorState``.

    Parameters
    ──────────
    messages : list[BaseMessage]
        The full ``state["messages"]`` audit ledger.  This list is never
        mutated — the function only reads from it.
    max_pairs : int
        Maximum number of (Human, AI) exchange pairs to include.  Each pair
        is approximately 2 messages, so ``max_pairs=3`` allows up to 6
        messages (plus the optional leading SystemMessage).
        Defaults to 3, which covers the Hive-Mind's reflexive exploitation
        window without overwhelming the evaluator LLM.

    Returns
    ───────
    list[BaseMessage]
        A new list containing:
          • The leading ``SystemMessage`` (if present), always at index 0.
          • At most the last ``max_pairs * 2`` non-System messages.
        Always returns a valid list — never raises IndexError even if
        ``messages`` is empty or shorter than the requested window.

    Examples
    ────────
    >>> from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    >>> msgs = [
    ...     SystemMessage(content="You are a helpful assistant."),
    ...     HumanMessage(content="Turn 1 question"),
    ...     AIMessage(content="Turn 1 answer"),
    ...     HumanMessage(content="Turn 2 question"),
    ...     AIMessage(content="Turn 2 answer"),
    ...     HumanMessage(content="Turn 3 question"),
    ...     AIMessage(content="Turn 3 answer"),
    ...     HumanMessage(content="Turn 4 question"),
    ...     AIMessage(content="Turn 4 answer"),
    ... ]
    >>> ctx = get_evaluator_context(msgs, max_pairs=2)
    >>> len(ctx)  # SystemMessage + 4 messages (2 pairs)
    5
    >>> isinstance(ctx[0], SystemMessage)
    True
    >>> ctx[1].content
    'Turn 3 question'
    """
    if not messages:
        return []

    # ── 1. Split off leading SystemMessage ────────────────────────────────
    system_prefix: list[BaseMessage] = []
    conversation: list[BaseMessage] = []

    for msg in messages:
        if not conversation and isinstance(msg, SystemMessage):
            system_prefix.append(msg)
        else:
            conversation.append(msg)

    # ── 2. Slice the tail of the conversation ─────────────────────────────
    # Each pair is 2 messages; cap at the available history length.
    max_messages = max_pairs * 2
    windowed = conversation[-max_messages:] if len(conversation) > max_messages else conversation

    # ── 3. Reassemble: system prefix (if any) + windowed conversation ──────
    return system_prefix + windowed
