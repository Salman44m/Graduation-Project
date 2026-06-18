import os
import re

regex = re.compile(r'([a-zA-Z0-9_.]+)\s+if\s+isinstance\(\1,\s*str\)\s+else\s+str\(\1\)')

for root, dirs, files in os.walk('.'):
    if 'venv' in root or '.git' in root or '__pycache__' in root or 'scratch' in root or '.venv' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content, count = regex.subn(r'extract_text(\1)', content)
            if count > 0:
                if 'from core.utils import extract_text' not in new_content:
                    lines = new_content.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            lines.insert(i, 'from core.utils import extract_text')
                            break
                    new_content = '\n'.join(lines)
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated {path} ({count} replacements)")
