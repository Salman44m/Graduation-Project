import os
import ast

def analyze_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        classes = []
        functions = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                doc = ast.get_docstring(node)
                classes.append({'name': node.name, 'methods': methods, 'doc': doc[:100] if doc else None})
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return {'classes': classes, 'functions': functions}
    except Exception as e:
        return str(e)

project_root = r'c:\Users\Mahmoud Salman\Downloads\prompt_evo - Grad project'
dirs_to_check = ['core', 'agents', 'memory', 'intelligence', 'evaluators', 'reporters', 'hitl', 'infra', 'remediation', 'adapters', 'schemas']

with open(r'c:\Users\Mahmoud Salman\Downloads\prompt_evo - Grad project\scratch\arch_dump.txt', 'w', encoding='utf-8') as out:
    for d in dirs_to_check:
        dp = os.path.join(project_root, d)
        if not os.path.exists(dp): continue
        for root, _, files in os.walk(dp):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    fp = os.path.join(root, file)
                    rel_fp = os.path.relpath(fp, project_root)
                    res = analyze_file(fp)
                    out.write(f'--- {rel_fp} ---\n')
                    if isinstance(res, dict):
                        for c in res['classes']:
                            name = c['name']
                            doc = c['doc']
                            methods = ", ".join(c['methods'])
                            out.write(f'Class: {name}\n')
                            if doc: out.write(f'  Doc: {doc}\n')
                            out.write(f'  Methods: {methods}\n')
                        if res['functions']:
                            funcs = ", ".join(res['functions'])
                            out.write(f'Functions: {funcs}\n')
                    else:
                        out.write(f'Error: {res}\n')
                    out.write('\n')
