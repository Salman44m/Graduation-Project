import json
from infra.persistence import _json_loads, _json_fallback
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage

# == Test 1: False Positive - does arbitrary dict get converted? ==
print("=== Test 1: False Positive Deserialization ===")

# A realistic HITL payload dict that contains "type" and "content"
hitl_payload = {
    "type": "human",
    "content": "Please approve this payload for submission",
    "payload": "JAILBREAK ATTEMPT HERE",
    "session_id": "abc123"
}
result = _json_loads(json.dumps(hitl_payload))
print("HITL dict with type=human, restored as:", type(result).__name__)
print("  Was coerced into HumanMessage:", isinstance(result, HumanMessage))

# == Test 2: RemoveMessage - does it survive round-trip? ==
print("\n=== Test 2: RemoveMessage Round-Trip ===")
rm = RemoveMessage(id="msg-to-delete")
for fmt, serialized in [
    ("to_json", json.dumps(rm.to_json())),
    ("model_dump", json.dumps(rm.model_dump())),
    ("_json_fallback", json.dumps(rm, default=_json_fallback))
]:
    result = _json_loads(serialized)
    print(f"  {fmt}: type={type(result).__name__}, is RemoveMessage={isinstance(result, RemoveMessage)}")

# == Test 3: AIMessage with tool_calls, empty content ==
print("\n=== Test 3: AIMessage with tool_calls ===")
ai = AIMessage(content="", tool_calls=[{"name":"fn","args":{},"id":"x"}])
serialized = json.dumps(ai.model_dump())
result = _json_loads(serialized)
print(f"  type={type(result).__name__}, is AIMessage={isinstance(result, AIMessage)}")
if isinstance(result, AIMessage):
    print(f"  tool_calls count: {len(result.tool_calls)}")

# == Test 4: What happens when get_request returns a dict with type=human ==
print("\n=== Test 4: Simulated get_request false-positive ===")
request_body = {
    "objective": "Test the model",
    "type": "human",  # e.g. request type field
    "content": "Probe the target with adversarial prompts",
    "model": "gpt-4o"
}
result = _json_loads(json.dumps(request_body))
print(f"  type={type(result).__name__}, is HumanMessage={isinstance(result, HumanMessage)}")
print(f"  Can call .get('model')?", end=" ")
try:
    val = result.get("model") if isinstance(result, dict) else getattr(result, "model", "ATTR_NOT_FOUND")
    print(val)
except Exception as e:
    print(f"ERROR: {e}")

# == Test 5: Full State Snapshot with RemoveMessage ==
print("\n=== Test 5: Full State Snapshot with RemoveMessage ===")
state_snapshot = {
    "messages": [
        HumanMessage(content="probe"),
        AIMessage(content="response"),
        RemoveMessage(id="probe-msg-id"),
    ],
    "status": "running",
}
serialized_state = json.dumps(state_snapshot, default=_json_fallback)
restored = _json_loads(serialized_state)
msgs = restored.get("messages", [])
print(f"  Messages count: {len(msgs)}")
for i, m in enumerate(msgs):
    print(f"  [{i}] type={type(m).__name__}, isinstance_RemoveMessage={isinstance(m, RemoveMessage)}")

# == Test 6: What does RemoveMessage model_dump 'type' field equal? ==
print("\n=== Test 6: RemoveMessage type field in model_dump ===")
rm2 = RemoveMessage(id="test-id")
d = rm2.model_dump()
print(f"  model_dump type field = '{d['type']}'")
print(f"  Does branch 2 of hook catch it? (type in human/ai/system):", d['type'] in ("human","ai","system"))
