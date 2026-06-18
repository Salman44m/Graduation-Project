def extract_text(content: any) -> str:
    """Extract text from LangChain message content, handling Gemini list-of-dicts."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "".join(parts)
    else:
        return str(content)
