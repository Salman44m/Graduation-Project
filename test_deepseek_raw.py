from openai import OpenAI

print("Testing Raw DeepSeek client via AgentRouter...")
try:
    client = OpenAI(
        api_key="sk-TC5OJV9x5HpT0SQTGZjYhbDmG6mO3QofCSvt05UUQIDajxZS",
        base_url="https://agentrouter.org/v1"
    )
    response = client.chat.completions.create(
        model="deepseek-v3.2",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "reply with just the word OK"}
        ]
    )
    print("Raw DeepSeek Pass!")
    print("Response:", response.choices[0].message.content)
except Exception:
    print("Raw DeepSeek Fail!")
    import traceback
    traceback.print_exc()
