import re

with open("dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix unescaped braces inside f-string for the new CSS blocks
# Let's find the exact primary button block and replace single { and } with {{ and }}
# Note: we need to be careful not to replace already escaped {{ or }}

def fix_braces(match):
    text = match.group(0)
    # temporarily convert {{ and }} to a placeholder, replace { with {{, etc
    text = text.replace("{{", "@@L@@").replace("}}", "@@R@@")
    text = text.replace("{", "{{").replace("}", "}}")
    text = text.replace("@@L@@", "{{").replace("@@R@@", "}}")
    return text

content = re.sub(r'/\* ── Primary button ───────────────────────────────────────────────────── \*/.*?/\* ── Metric cards', fix_braces, content, flags=re.DOTALL)

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)
