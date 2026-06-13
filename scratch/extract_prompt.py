import json

path = r"C:\Users\Mahmoud Salman\.gemini\antigravity\brain\f4c82f7a-f1b7-4e11-a844-31c1d26f86ad\.system_generated\logs\transcript_full.jsonl"
with open(path, "r", encoding="utf-8") as f:
    first_line = f.readline()

data = json.loads(first_line)
prompt_content = data["content"]

with open("scratch/original_prompt.txt", "w", encoding="utf-8") as out:
    out.write(prompt_content)

print("Prompt written to scratch/original_prompt.txt successfully.")
