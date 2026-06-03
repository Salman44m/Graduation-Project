import os
from dotenv import load_dotenv

load_dotenv()

from core.llm_factory import create_chat_model, Provider

try:
    print("Testing Groq...")
    model = create_chat_model(
        provider=Provider.GROQ,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    response = model.invoke("reply with just the word OK")
    print("Groq Pass!")
    print("Response:", response.content)
except Exception:
    print("Groq Fail!")
    import traceback
    traceback.print_exc()
