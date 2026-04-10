import logging
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

# Configuration to suppress noise if needed, but we want to see UserWarnings
logging.basicConfig(level=logging.INFO)

print("--- Starting Build Graph Check ---")
try:
    from core.graph import build_graph
    app = build_graph()
    if app is not None:
        print("Success: Graph compiled.")
    else:
        print("Failure: Graph is None.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
print("--- End Build Graph Check ---")
