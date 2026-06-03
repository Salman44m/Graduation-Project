
import config

# Manually clear the keys loaded from .env
config.settings.groq_api_key = None
config.settings.target_groq_key = None

try:
    print("Testing Target Adapter with missing GROQ_API_KEY...")
    llm = config.get_target_adapter()
    print("FAIL: Expected MissingAPIKeyError, but got an LLM!")
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")

try:
    print("\nTesting Attacker LLM with missing GROQ_API_KEY...")
    llm = config.get_attacker_llm()
    print("FAIL: Expected MissingAPIKeyError, but got an LLM!")
except Exception as e:
    print(f"Caught Exception: {type(e).__name__}: {e}")
