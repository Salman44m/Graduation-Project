from anthropic import Anthropic

print("Testing Raw Anthropic client via AgentRouter...")
try:
    client = Anthropic(
        api_key="sk-WVG9y0jb5GaeqyLlHXnCgoPOnQDP9s2WgWOj2tm1Jl5ZQqS0",
        base_url="https://agentrouter.org"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "reply with just the word OK"}
        ]
    )
    print("Raw Claude Pass!")
    print("Response:", response.content[0].text)
except Exception:
    print("Raw Claude Fail!")
    import traceback
    traceback.print_exc()
