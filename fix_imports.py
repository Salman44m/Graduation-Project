import os

for root, dirs, files in os.walk('.'):
    if 'venv' in root or '.git' in root or '__pycache__' in root or 'scratch' in root or '.venv' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check if we have the issue
            has_extract = any('from core.utils import extract_text' in line for line in lines)
            has_future = any('from __future__ import' in line for line in lines)
            
            if not (has_extract and has_future):
                continue
            
            extract_idx = -1
            future_idx = -1
            for i, line in enumerate(lines):
                if 'from core.utils import extract_text' in line:
                    extract_idx = i
                if 'from __future__ import' in line:
                    future_idx = max(future_idx, i)
                    
            if extract_idx != -1 and future_idx != -1 and extract_idx < future_idx:
                # Need to fix
                lines.pop(extract_idx)
                # Recalculate future index after pop
                future_idx = -1
                for i, line in enumerate(lines):
                    if 'from __future__ import' in line:
                        future_idx = i
                
                lines.insert(future_idx + 1, 'from core.utils import extract_text\n')
                
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                print(f"Fixed import order in {path}")
