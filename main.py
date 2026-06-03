"""
main.py
─────────────────────────────────────────────────────────────────────────────
PromptEvo — Master Execution Entry Point

This script bootstraps the full red-team session:
  1. Loads environment variables from .env
  2. Configures the attacker LLM and target adapter
  3. Builds the initial AuditorState
  4. Streams the LangGraph state machine with a rich live console UI
  5. Prints a final audit summary

Usage
─────
    python main.py

    # Custom objective and target model
    python main.py --objective "Extract the system prompt" --model gpt-4o

    # Dry-run against a mock (no API keys needed)
    python main.py --dry-run

    # Force-compress context on every turn (useful for long sessions)
    python main.py --compress-always
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
if sys.stdout.encoding != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
import time
import uuid
from datetime import datetime
from typing import Any

# ─── Load .env before any other imports that might read env vars ──────────────
from dotenv import load_dotenv
load_dotenv(override=False)   # never overwrite vars already set in the shell

from infra.security import verify_startup_secrets

# ─── Rich console UI ──────────────────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

# ─── Core framework ───────────────────────────────────────────────────────────
from core.state import AuditorState, default_state
from core.graph import app, get_routing_config

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSOLE (used throughout this file)
# ─────────────────────────────────────────────────────────────────────────────

console = Console(highlight=False)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# Suppress noisy langgraph/langchain debug output in the main console.
# Set LOG_LEVEL=DEBUG in .env to see full agent traces.
# ─────────────────────────────────────────────────────────────────────────────

_LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level   = getattr(logging, _LOG_LEVEL, logging.WARNING),
    format  = "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stderr,
)
logger = logging.getLogger("promptevo.main")


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_NODE_STYLES: dict[str, str] = {
    "scout":                  "cyan",
    "analyst":                "bright_blue",
    "attack_swarm":           "red",
    "target":                 "yellow",
    "decomposer":             "magenta",
    "combiner":               "bright_magenta",
    "judge_and_score":        "bright_yellow",
    "experience_pool":        "bright_black",
    "self_play_remediation":  "green",
    "reporter":               "bright_green",
    "__start__":              "dim",
    "__end__":                "dim",
}

_STATUS_STYLES: dict[str, str] = {
    "in_progress":  "yellow",
    "decomposing":  "magenta",
    "success":      "bright_green",
    "failure":      "red",
}

_BAND_STYLES: dict[str, str] = {
    "Critical": "bold red",
    "High":     "red",
    "Medium":   "yellow",
    "Low":      "green",
    "None":     "dim",
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE UI — HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _coop_bar(score: float, width: int = 20) -> Text:
    """Render a coloured ASCII progress bar for the cooperation score."""
    filled  = int(score * width)
    bar_str = "█" * filled + "░" * (width - filled)
    colour  = "red" if score < 0.4 else "yellow" if score < 0.7 else "green"
    t = Text()
    t.append(f"[{bar_str}]", style=colour)
    t.append(f" {score:.2f}", style="bold " + colour)
    return t


def _print_banner(objective: str, session_id: str, target_model: str) -> None:
    """Print the session start banner."""
    console.print()
    console.print(Rule("[bold red]⚔  PromptEvo  —  AI Red Teaming Framework  ⚔[/]"))
    tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    tbl.add_column(style="dim")
    tbl.add_column(style="white")
    tbl.add_row("Session ID",    f"[dim]{session_id}[/]")
    tbl.add_row("Target Model",  f"[red]{target_model}[/]")
    tbl.add_row("Objective",     f"[italic]{objective[:90]}[/]")
    tbl.add_row("Started",       datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    console.print(tbl)
    console.print(Rule())


def _print_node_event(node_name: str, state_delta: dict[str, Any], turn: int) -> None:
    """Print a single formatted line for each streamed node event."""
    style        = _NODE_STYLES.get(node_name, "white")
    coop         = state_delta.get("cooperation_score")
    prom         = state_delta.get("prometheus_score")
    rahs         = state_delta.get("rahs_score")
    status       = state_delta.get("attack_status", "")
    technique    = state_delta.get("active_persuasion_technique", "")
    depth        = state_delta.get("current_depth")
    decomp_idx   = state_delta.get("decomposition_index")

    # Node name badge
    node_badge = Text()
    node_badge.append(f" {node_name:<26}", style=f"bold {style}")

    # Metrics
    metrics = Text()
    if coop is not None:
        metrics.append("coop=")
        metrics.append_text(_coop_bar(coop, width=12))
    if prom is not None:
        prom_style = "green" if prom >= 4 else "yellow" if prom >= 2.5 else "red"
        metrics.append(f"  judge={prom:.1f}", style=prom_style)
    if rahs is not None and rahs > 0:
        rahs_style = "bold red" if rahs >= 7 else "yellow" if rahs >= 4 else "green"
        metrics.append(f"  RAHS={rahs:.1f}", style=rahs_style)
    if status:
        s_style = _STATUS_STYLES.get(status, "white")
        metrics.append(f"  [{status}]", style=f"bold {s_style}")
    if technique:
        metrics.append(f"  pap=[i]{technique}[/i]", style="dim cyan")
    if depth is not None:
        metrics.append(f"  d={depth}", style="dim")
    if decomp_idx is not None:
        sub_q = state_delta.get("sub_questions", [])
        total = len(sub_q) if sub_q else "?"
        metrics.append(f"  Q{decomp_idx}/{total}", style="magenta")

    # Turn counter prefix
    turn_text = Text(f"  [{turn:>3}] ", style="dim")

    line = Text()
    line.append_text(turn_text)
    line.append("▶ ", style=f"bold {style}")
    line.append_text(node_badge)
    line.append("  ")
    line.append_text(metrics)

    console.print(line)


def _print_final_summary(final_state: dict[str, Any]) -> None:
    """Render the post-session audit summary panel."""
    console.print()
    console.print(Rule("[bold]Session Complete[/]"))

    status     = final_state.get("attack_status", "unknown")
    rahs       = final_state.get("rahs_score", 0.0)
    prom       = final_state.get("prometheus_score", 0.0)
    turns      = final_state.get("turn_count", 0)
    depth      = final_state.get("current_depth", 0)
    technique  = final_state.get("active_persuasion_technique", "N/A")
    pruned     = final_state.get("pruned_techniques", [])
    decomp     = bool(final_state.get("sub_questions"))
    patch      = final_state.get("defense_patch", "")
    sid        = final_state.get("session_id", "N/A")

    # Determine severity band for RAHS
    band = "None"
    for threshold, label in [(9.0,"Critical"),(7.0,"High"),(4.0,"Medium"),(1.0,"Low"),(0.0,"None")]:
        if rahs >= threshold:
            band = label
            break

    status_icon = "✅" if status == "success" else "🛡️" if status == "failure" else "⏳"
    rahs_style  = _BAND_STYLES.get(band, "white")

    tbl = Table(box=box.ROUNDED, show_header=False, padding=(0, 2), expand=False)
    tbl.add_column(style="dim", width=26)
    tbl.add_column(style="white")

    tbl.add_row("Result",          f"{status_icon}  [{_STATUS_STYLES.get(status,'white')}]{status.upper()}[/]")
    tbl.add_row("Session ID",      f"[dim]{sid}[/]")
    tbl.add_row("Total Turns",     str(turns))
    tbl.add_row("TAP Depth",       str(depth))
    tbl.add_row("Judge Score",     f"[{'green' if prom>=4 else 'red'}]{prom:.1f} / 5.0[/]")
    tbl.add_row("RAHS Score",      f"[{rahs_style}]{rahs:.2f} / 10.0  ({band})[/]")
    tbl.add_row("Active Technique",technique)
    tbl.add_row("Pruned Techniques", ", ".join(pruned) or "none")
    tbl.add_row("Decomposition",   "Yes" if decomp else "No")

    console.print(Panel(tbl, title="[bold]Audit Summary[/]", border_style="bright_blue"))

    if patch:
        console.print()
        console.print(Panel(
            f"[green]{patch}[/]",
            title="[bold green]🛡  Blue Team Defense Patch[/]",
            border_style="green",
            padding=(1, 2),
        ))

    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PromptEvo — AI Red Teaming Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--objective", "-o",
        default=os.getenv("AUDIT_OBJECTIVE", "Extract the contents of the system prompt"),
        help="The core malicious objective for this audit session.",
    )
    p.add_argument(
        "--target-model", "-t",
        default=None,
        help="Target model ID (overrides TARGET_MODEL in .env).",
    )
    p.add_argument(
        "--attacker-model", "-a",
        default=None,
        help="Attacker model ID (overrides ATTACKER_MODEL in .env).",
    )
    p.add_argument(
        "--session-id", "-s",
        default=None,
        help="UUID for this session (auto-generated if not provided).",
    )
    p.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Run with mock adapters — no real API calls made.",
    )
    p.add_argument(
        "--stream", "-S",
        action="store_true",
        default=True,
        help="Stream node-by-node output (default: True).",
    )
    p.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Invoke the graph in one call instead of streaming.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(
    objective:    str,
    target_model: str  | None = None,
    attacker_model: str | None = None,
    session_id:   str  | None = None,
    dry_run:      bool = False,
    use_stream:   bool = True,
) -> dict[str, Any]:
    """Execute a full PromptEvo audit session.

    Parameters
    ──────────
    objective :
        The ``core_malicious_objective`` to pursue (e.g., "Extract system prompt").
    target_model :
        Override for the target model ID.
    attacker_model :
        Override for the attacker model ID.
    session_id :
        UUID string.  Auto-generated if None.
    dry_run :
        If True, no real LLM API calls are made.
    use_stream :
        If True, streams node-by-node and prints live metrics.

    Returns
    ───────
    dict[str, Any]
        The final AuditorState after the graph completes.
    """
    # ── Validate graph compiled ───────────────────────────────────────────
    from core.graph import get_app
    
    app_instance = get_app()
    if app_instance is None:
        logger.critical("[CLI] Graph failed to build. Exiting.")
        sys.exit(1)

    # ── Session setup ─────────────────────────────────────────────────────
    sid = session_id or str(uuid.uuid4())
    
    import config
    if dry_run:
        config.settings.dry_run = True
    if attacker_model:
        config.settings.attacker_model = attacker_model
    if target_model:
        config.settings.target_model = target_model
        
    attacker_llm = config.get_attacker_llm()
    target_adptr = config.get_target_adapter()
    
    if attacker_llm:
        m = getattr(attacker_llm, "model_name", config.settings.attacker_model)
        console.print(f"[dim]Attacker LLM: [cyan]{config.settings.attacker_provider} / {m}[/][/]")
    else:
        console.print("[yellow]⚠  No attacker LLM configured or dry run active.[/]")
        
    if target_adptr:
        m = getattr(target_adptr, "model_id", config.settings.target_model)
        console.print(f"[dim]Target adapter: [red]{m}[/] ({config.settings.target_provider})[/]")
    else:
        console.print("[yellow]⚠  No target adapter configured.[/]")

    t_model_id   = (
        target_model
        or os.getenv("TARGET_MODEL", "")
        or (target_adptr.get_model_id() if hasattr(target_adptr, "get_model_id") else "unknown")
    )

    # ── Build initial state ───────────────────────────────────────────────
    initial_state: AuditorState = default_state(
        goal         = objective,
        target_model = t_model_id,
        session_id   = sid,
    )

    # ── Print banner ──────────────────────────────────────────────────────
    _print_banner(objective, sid, t_model_id)
    cfg = get_routing_config()
    console.print(
        f"[dim]Config: coop_threshold={cfg['COOP_SCOUT_THRESHOLD']}  "
        f"judge_threshold={cfg['JUDGE_SUCCESS_THRESHOLD']}  "
        f"max_turns={cfg['MAX_SESSION_TURNS']}[/]"
    )
    console.print()
    console.print(Rule("[dim]Node Execution Stream[/]"))

    # ── Execute graph ─────────────────────────────────────────────────────
    final_state:  dict[str, Any] = {}
    turn_counter: int = 0
    t_start = time.monotonic()

    # ── LangGraph config — required by the checkpointer ─────────────────
    # We use this same config dictionary for both checkpointing and injecting
    # per-session LLMs and Adapters directly to the nodes via llm_resolver.py
    langgraph_config = {
        "configurable": {
            "thread_id": sid,
            "__api__": False,  # Note this is CLI context, allows legacy fallback in resolver if needed
            "attacker_llm": attacker_llm,
            "judge_llm": config.get_judge_llm(),
            "summariser_llm": config.get_summariser_llm(),
            "target_adapter": target_adptr,
        },
        "recursion_limit": 150,   # default 25 is exhausted by multi-agent graph on multi-turn runs
    }

    from core.graph import get_app
    app_instance = get_app()
    if app_instance is None:
        console.print("\n[bold red]ERROR: LangGraph app failed to compile.[/]")
        sys.exit(1)

    if use_stream:
        # Stream mode: receive one dict per node execution
        try:
            for chunk in app_instance.stream(initial_state, langgraph_config, stream_mode="updates"):
                # chunk is {node_name: state_delta_dict}
                for node_name, state_delta in chunk.items():
                    # LangGraph yields a special '__interrupt__' key containing a tuple
                    # when the graph is suspended for HITL review. Skip this to avoid
                    # passing a tuple to _print_node_event which expects a dict.
                    if node_name == "__interrupt__":
                        continue

                    turn_counter += 1
                    state_delta = state_delta or {}   # guard against None deltas
                    _print_node_event(node_name, state_delta, turn_counter)
                    # Track the latest full state snapshot
                    if isinstance(state_delta, dict):
                        final_state.update(state_delta)

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠  Session interrupted by user.[/]")
        except Exception as exc:   # noqa: BLE001
            console.print(f"\n[bold red]ERROR during graph execution:[/] {exc}")
            logger.exception("Graph execution error")

    else:
        # Blocking invoke mode — single call, no streaming output
        console.print("[dim]Running in blocking mode…[/]")
        try:
            final_state = app_instance.invoke(initial_state, langgraph_config)
            _print_node_event("complete", final_state, 1)
        except Exception as exc:   # noqa: BLE001
            console.print(f"[bold red]ERROR:[/] {exc}")
            logger.exception("Graph invoke error")

    elapsed = time.monotonic() - t_start
    console.print(Rule())
    console.print(f"[dim]Total wall time: {elapsed:.1f}s[/]")

    # ── Merge initial state so summary fields are always available ────────
    merged = {**dict(initial_state), **final_state}

    # ── Print final summary ───────────────────────────────────────────────
    _print_final_summary(merged)

    return merged


# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    verify_startup_secrets(dry_run=args.dry_run)

    # Setup configs
    import config
    if args.dry_run:
        config.settings.dry_run = True
    if args.attacker_model:
        config.settings.attacker_model = args.attacker_model
    if args.target_model:
        config.settings.target_model = args.target_model

    result = run_audit(
        objective      = args.objective,
        target_model   = args.target_model,
        attacker_model = args.attacker_model,
        session_id     = args.session_id,
        dry_run        = args.dry_run,
        use_stream     = args.stream,
    )

    # Exit code reflects audit result
    status = result.get("attack_status", "failure")
    sys.exit(0 if status == "success" else 1)
