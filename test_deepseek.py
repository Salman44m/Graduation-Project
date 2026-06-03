
from core.llm_factory import create_chat_model, Provider

try:
    print("Testing DeepSeek via AgentRouter...")
    model = create_chat_model(
        provider=Provider.DEEPSEEK,
        model_name="deepseek-v3.2",
        temperature=0.0,
        api_key="sk-TC5OJV9x5HpT0SQTGZjYhbDmG6mO3QofCSvt05UUQIDajxZS",
        base_url="https://agentrouter.org/v1"
    )
    response = model.invoke("reply with just the word OK")
    print("DeepSeek Pass!")
    print("Response:", response.content)
except Exception:
    print("DeepSeek Fail!")
    import traceback
    traceback.print_exc()
