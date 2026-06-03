import json
import re
from typing import Annotated, Any, Callable
import operator
import logging

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

class CorrectionState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    parsed_json: Any
    error_count: int
    max_retries: int

def build_self_correction_graph(llm: Any) -> Callable[[list[BaseMessage], int], Any]:
    """Builds a LangGraph-native self-correction loop for JSON parsing.
    
    Returns a callable that takes initial messages and max_retries, and returns the parsed JSON
    or raises ValueError if it fails after max_retries.
    """
    def _llm_node(state: CorrectionState):
        logger.debug("[Self-Correction] Invoking LLM for formatting correction.")
        resp = llm.invoke(state["messages"])
        return {"messages": [resp]}

    def _parse_node(state: CorrectionState):
        last_msg = state["messages"][-1]
        raw = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            parsed = json.loads(raw)
            logger.debug("[Self-Correction] Parse successful.")
            return {"parsed_json": parsed}
        except Exception as exc:
            logger.warning("[Self-Correction] Parse failed: %s", exc)
            err_msg = f"Failed to parse JSON: {exc}. Please fix the formatting and return ONLY valid JSON."
            return {"error_count": state.get("error_count", 0) + 1, "messages": [HumanMessage(content=err_msg)]}

    def _route(state: CorrectionState):
        if state.get("parsed_json") is not None:
            return END
        if state.get("error_count", 0) >= state.get("max_retries", 3):
            return END
        return "llm"

    builder = StateGraph(CorrectionState)
    builder.add_node("llm", _llm_node)
    builder.add_node("parse", _parse_node)
    builder.add_edge("llm", "parse")
    builder.add_conditional_edges("parse", _route, {"llm": "llm", END: END})
    builder.set_entry_point("llm")
    compiled = builder.compile()

    def run_correction(initial_messages: list[BaseMessage], max_retries: int = 3) -> Any:
        # Create initial state
        initial_state = {
            "messages": initial_messages,
            "parsed_json": None,
            "error_count": 0,
            "max_retries": max_retries
        }
        res = compiled.invoke(initial_state)
        
        parsed = res.get("parsed_json")
        if parsed is None:
            raise ValueError(f"Self-correction failed to produce valid JSON after {max_retries} attempts.")
        return parsed

    return run_correction
