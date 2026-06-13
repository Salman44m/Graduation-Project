"""
preflight_check.py
PromptEvo Pre-Flight Sanity Check

Verifies four pillars before running the full graph:
  1. Environment & Config     — .env loaded, TARGET_PROVIDER=gemini, key present
  2. Dependency Resolution    — langchain-google-genai & all required packages
  3. LLM Factory / API Ping   — discovery chain resolves + actual .invoke("ping")
  4. Graph Compilation        — build_graph() succeeds without syntax/import errors

Exit codes:
  0 — All systems go
  1 — One or more checks FAILED
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import traceback
import io

# Force UTF-8 stdout so Unicode emojis don't crash on Windows cp1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from typing import Any

# ── Rich or plain text output ─────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table
    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False

# Colour helpers for plain-text fallback
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _ok(msg: str)  -> None: print(f"{GREEN}  [PASS]  {msg}{RESET}")
def _fail(msg: str) -> None: print(f"{RED}  [FAIL]  {msg}{RESET}")
def _warn(msg: str) -> None: print(f"{YELLOW}  [WARN]  {msg}{RESET}")
def _info(msg: str) -> None: print(f"{CYAN}  [INFO]  {msg}{RESET}")
def _head(msg: str) -> None: print(f"\n{BOLD}{CYAN}{'-'*60}\n  {msg}\n{'-'*60}{RESET}")


results: list[tuple[str, bool, str]] = []   # (pillar_name, passed, detail)


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    if passed:
        _ok(f"{name}: {detail}")
    else:
        _fail(f"{name}: {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 1 — Environment & Config
# ─────────────────────────────────────────────────────────────────────────────
_head("PILLAR 1 — Environment & Config")

try:
    from dotenv import load_dotenv
    loaded = load_dotenv(override=False)
    _info(f".env load_dotenv() → loaded={loaded}")
    record("dotenv loaded", True, ".env file parsed successfully")
except Exception as exc:
    record("dotenv loaded", False, f"load_dotenv() raised: {exc}")

# TARGET_PROVIDER check
target_provider = os.getenv("TARGET_PROVIDER", "")
if target_provider.lower() == "gemini":
    record("TARGET_PROVIDER", True, f"TARGET_PROVIDER='{target_provider}' ✓")
elif target_provider == "":
    record("TARGET_PROVIDER", False, "TARGET_PROVIDER is not set in .env")
else:
    record("TARGET_PROVIDER", False, f"TARGET_PROVIDER='{target_provider}' — expected 'gemini'")

# Google API key presence — check multiple known env vars
key_candidates = [
    "TARGET_GEMINI_KEY",
    "Gemini_Summarize_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
]
found_key_name: str | None = None
for kname in key_candidates:
    val = os.getenv(kname, "")
    if val.strip():
        found_key_name = kname
        break

if found_key_name:
    raw = os.getenv(found_key_name, "")
    masked = raw[:4] + "..." + raw[-4:] if len(raw) >= 8 else "***"
    record("Google API Key", True, f"Found via {found_key_name}= {masked} (length={len(raw)})")
else:
    record("Google API Key", False,
           f"No Google API key found. Checked: {', '.join(key_candidates)}")

# Print full config snapshot (safe — no raw keys)
try:
    import config  # uses load_dotenv internally too
    s = config.settings
    _info(f"settings.target_provider       = '{s.target_provider}'")
    _info(f"settings.target_model          = '{s.target_model}'")
    _info(f"settings.attacker_provider     = '{s.attacker_provider}'")
    _info(f"settings.attacker_model        = '{s.attacker_model}'")
    _info(f"settings.judge_provider        = '{s.judge_provider}'")
    _info(f"settings.target_gemini_key set = {bool(s.target_gemini_key)}")
    _info(f"settings.gemini_summarize_key  = {bool(s.gemini_summarize_key)}")
    record("config.py import", True, "PromptEvoSettings loaded without errors")
except Exception as exc:
    record("config.py import", False, f"config.py failed to import: {exc}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2 — Dependency Resolution
# ─────────────────────────────────────────────────────────────────────────────
_head("PILLAR 2 — Dependency Resolution")

REQUIRED_PACKAGES: list[tuple[str, str]] = [
    # (import_name,            pip_name)
    ("langchain_google_genai", "langchain-google-genai"),
    ("langchain",              "langchain"),
    ("langchain_core",         "langchain-core"),
    ("langchain_anthropic",    "langchain-anthropic"),
    ("langchain_openai",       "langchain-openai"),
    ("langchain_groq",         "langchain-groq"),
    ("langchain_community",    "langchain-community"),
    ("langgraph",              "langgraph"),
    ("dotenv",                 "python-dotenv"),
    ("pydantic",               "pydantic"),
    ("anthropic",              "anthropic"),
    ("openai",                 "openai"),
    ("faiss",                  "faiss-cpu"),
    ("rich",                   "rich"),
    ("yaml",                   "pyyaml"),
    ("tenacity",               "tenacity"),
    ("numpy",                  "numpy"),
]

all_deps_ok = True
for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        # Get version safely
        try:
            import importlib.metadata
            ver = importlib.metadata.version(pip_name)
        except Exception:
            ver = getattr(mod, "__version__", "?")
        _ok(f"  {pip_name:35s} → v{ver}")
    except ImportError as exc:
        _fail(f"  {pip_name:35s} → MISSING ({exc})")
        all_deps_ok = False

record("Dependency Resolution", all_deps_ok,
       "All required packages importable" if all_deps_ok
       else "One or more packages are missing — run: pip install -r requirements.txt")


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 3 — LLM Factory & Connectivity Ping
# ─────────────────────────────────────────────────────────────────────────────
_head("PILLAR 3 — LLM Factory & Connectivity Ping")

# Determine which Google API key to use
google_api_key: str = ""
for kname in key_candidates:
    v = os.getenv(kname, "")
    if v.strip():
        google_api_key = v.strip()
        _info(f"Using key from env var: {kname}")
        break

if not google_api_key:
    record("Gemini Model Discovery", False, "No Google API key available — skipping API ping")
    record("Gemini API Ping", False, "Skipped — no key")
else:
    # Step 3a: Model discovery via llm_factory._resolve_gemini_model
    try:
        from core.llm_factory import _resolve_gemini_model, LLMFactoryError

        _info("Running _resolve_gemini_model() discovery chain (no preferred model)…")
        t0 = time.time()
        llm = _resolve_gemini_model(
            preferred_model=os.getenv("TARGET_MODEL", None),
            api_key=google_api_key,
            temperature=0.0,
            max_tokens=64,
        )
        elapsed = time.time() - t0

        # Extract which model was resolved
        resolved_model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
        record("Gemini Model Discovery", True,
               f"Resolved model: '{resolved_model}' in {elapsed:.2f}s")

        # Step 3b: Actual .invoke() ping to Google API
        _info(f"Sending .invoke('Say PONG') to '{resolved_model}'…")
        t1 = time.time()
        response = llm.invoke("Say only the word PONG and nothing else.")
        elapsed2 = time.time() - t1

        response_text = getattr(response, "content", str(response)).strip()
        _info(f"API response: '{response_text}' ({elapsed2:.2f}s)")
        record("Gemini API Ping", True,
               f"Successful round-trip in {elapsed2:.2f}s → model='{resolved_model}', response='{response_text[:80]}'")

    except Exception as exc:
        record("Gemini Model Discovery", False, f"{type(exc).__name__}: {exc}")
        record("Gemini API Ping", False, "Could not complete ping — see discovery error above")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 4 — Graph Compilation
# ─────────────────────────────────────────────────────────────────────────────
_head("PILLAR 4 — Graph Compilation")

try:
    _info("Importing core.graph…")
    import core.graph as graph_module
    _ok("core.graph imported successfully")

    _info("Calling build_graph()…")
    t0 = time.time()
    compiled_graph = graph_module.build_graph()
    elapsed = time.time() - t0

    # Verify it's a compiled LangGraph object
    graph_type = type(compiled_graph).__name__
    has_nodes = hasattr(compiled_graph, "nodes") or hasattr(compiled_graph, "get_graph")

    if has_nodes:
        try:
            # Try to get node count for extra info
            g = compiled_graph.get_graph()
            node_names = list(g.nodes.keys())
            _info(f"Graph nodes ({len(node_names)}): {node_names}")
            record("Graph Compilation", True,
                   f"build_graph() completed in {elapsed:.2f}s · {len(node_names)} nodes · type={graph_type}")
        except Exception:
            record("Graph Compilation", True,
                   f"build_graph() completed in {elapsed:.2f}s · type={graph_type}")
    else:
        record("Graph Compilation", True,
               f"build_graph() returned {graph_type} in {elapsed:.2f}s")

except Exception as exc:
    record("Graph Compilation", False, f"{type(exc).__name__}: {exc}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# -------------------------------------------------------------------------------
print(f"\n{BOLD}{'='*60}")
print(f"  PRE-FLIGHT CHECK RESULTS")
print(f"{'='*60}{RESET}")

all_passed = True
for name, passed, detail in results:
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{status}]  {name}")
    if not passed:
        print(f"         {RED}-> {detail}{RESET}")
        all_passed = False

print(f"\n{'='*60}")
if all_passed:
    print(f"{GREEN}{BOLD}  >>> ALL SYSTEMS GO - Safe to run PromptEvo! <<<{RESET}")
else:
    failed_count = sum(1 for _, p, _ in results if not p)
    print(f"{RED}{BOLD}  >>> {failed_count} CHECK(S) FAILED - Resolve above issues before running. <<<{RESET}")
print(f"{'='*60}\n")

sys.exit(0 if all_passed else 1)
