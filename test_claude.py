
from core.llm_factory import create_chat_model, Provider

try:
    print("Testing Claude via AgentRouter...")
    model = create_chat_model(
        provider=Provider.ANTHROPIC,
        model_name="claude-haiku-4-5-20251001",
        temperature=0.0,
        api_key="sk-WVG9y0jb5GaeqyLlHXnCgoPOnQDP9s2WgWOj2tm1Jl5ZQqS0",
        base_url="https://agentrouter.org"
    )
    response = model.invoke("reply with just the word OK")
    print("Claude Pass!")
    print("Response:", response.content)
except Exception:
    print("Claude Fail!")
    import traceback
    traceback.print_exc()
