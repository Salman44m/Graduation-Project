import ast
import os
import subprocess
import sys

def get_imports(directory):
    imports = set()
    for root, dirs, files in os.walk(directory):
        if '.venv' in dirs:
            dirs.remove('.venv')
        if 'tests' in dirs:
            dirs.remove('tests')
            
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read(), filename=path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.level == 0:
                                imports.add(node.module.split('.')[0])
                except Exception as e:
                    print(f"Error parsing {path}: {e}")
    return imports

first_party = {'core', 'adapters', 'agents', 'evaluators', 'infra', 'memory', 'remediation', 'reports', 'config', 'api', 'main', 'dashboard'}
std_lib = sys.stdlib_module_names

def get_installed_packages():
    try:
        pip_path = os.path.join('.venv', 'Scripts', 'pip.exe')
        result = subprocess.run([pip_path, 'freeze'], capture_output=True, text=True, check=True)
        packages = {}
        for line in result.stdout.splitlines():
            line = line.strip()
            if '==' in line:
                name, version = line.split('==', 1)
                packages[name.lower()] = version
            elif '@' in line:
                name = line.split('@')[0].strip()
                packages[name.lower()] = "any"
        return packages
    except Exception as e:
        print(f"Error running pip freeze: {e}")
        return {}

def main():
    detected_imports = get_imports('.')
    installed = get_installed_packages()
    
    # Mapping exact pip names for commonly mismatched ones
    # (e.g. dotenv -> python-dotenv, yaml -> pyyaml, PIL -> pillow)
    mapping = {
        'dotenv': 'python-dotenv',
        'yaml': 'pyyaml',
        'langchain_openai': 'langchain-openai',
        'langchain_anthropic': 'langchain-anthropic',
        'langchain_groq': 'langchain-groq',
        'langchain_ollama': 'langchain-ollama',
        'langchain_core': 'langchain-core',
        'faiss': 'faiss-cpu',
        'pydantic': 'pydantic',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'starlette': 'starlette',
        'streamlit': 'streamlit',
        'pytest': 'pytest',
        'httpx': 'httpx',
        'rich': 'rich',
        'tiktoken': 'tiktoken',
        'numpy': 'numpy'
    }

    # We also have implicit runtime dependencies like uvicorn for api.py, streamlit for dashboard.py
    implicit = ['uvicorn', 'streamlit', 'pytest']
    
    all_needed = detected_imports.union(implicit)
    
    requirements = []
    
    print("Dependencies Found via AST:")
    for imp in sorted(all_needed):
        if imp in first_party or imp in std_lib or imp.startswith('_'):
            continue
            
        pip_name = mapping.get(imp, imp)
        pip_name_lower = pip_name.lower().replace('_', '-')
        
        # Check if we have it in installed
        version = installed.get(pip_name_lower)
        if version:
            requirements.append(f"{pip_name}=={version}")
            print(f" - {imp} -> {pip_name}=={version}")
        else:
            # Let's try some variations or check if it's part of another package
            print(f" - [!] {imp} (mapped as {pip_name}) not found in pip freeze")

    print("\n[requirements.txt format]")
    for req in sorted(requirements):
        print(req)

if __name__ == '__main__':
    main()
