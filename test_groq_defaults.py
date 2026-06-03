import config

try:
    print("Testing Attacker LLM out-of-the-box defaults...")
    attacker = config.get_attacker_llm()
    response = attacker.invoke("reply with just the word OK")
    print(f"Attacker OK! ({response.content})")

    print("\nTesting Target Adapter out-of-the-box defaults...")
    target = config.get_target_adapter()
    response2 = target._model.invoke("reply with just the word OK")
    print(f"Target OK! ({response2.content})")

except Exception:
    print("FAIL!")
    import traceback
    traceback.print_exc()
