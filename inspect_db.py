"""
inspect_db.py
─────────────────────────────────────────────────────────────────────────────
QA Verification Tool — Phase 2 Database Persistence
====================================================
Connects to audit_reports.db and prints a formatted summary of the most
recent audit session, including JSON-column integrity checks.

Usage
─────
    python inspect_db.py                    # default: audit_reports.db
    python inspect_db.py --db my.db         # custom path
    python inspect_db.py --all              # show all sessions, not just latest
    python inspect_db.py --session <uuid>   # look up a specific session ID
"""

from __future__ import annotations  # must be the very first statement

# ── Force UTF-8 on Windows terminals (cp1252 can't encode some Unicode) ──────
import sys as _sys
import io as _io
if hasattr(_sys.stdout, "buffer"):
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os
import sys
from datetime import datetime

# ── Colour helpers (no external deps — pure ANSI) ─────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"
GREY   = "\033[90m"


def c(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes."""
    return "".join(codes) + str(text) + RESET


def _severity_colour(band: str) -> str:
    return {
        "Critical": RED,
        "High":     YELLOW,
        "Medium":   MAGENTA,
        "Low":      GREEN,
        "None":     GREY,
    }.get(band, WHITE)


def _status_icon(status: str) -> str:
    return {
        "success":     c("✅  BREACHED",    GREEN,  BOLD),
        "failure":     c("🛡  DEFENDED",    CYAN,   BOLD),
        "in_progress": c("⏳  PARTIAL",     YELLOW, BOLD),
    }.get(status, c(f"❓  {status.upper()}", GREY))


def _rahs_bar(score: float, width: int = 30) -> str:
    """Render a simple ASCII progress bar for the RAHS score (0-10)."""
    filled = int(round(score / 10.0 * width))
    bar    = "█" * filled + "░" * (width - filled)
    col    = RED if score >= 7 else YELLOW if score >= 4 else GREEN
    return c(bar, col) + c(f"  {score:.2f}/10.0", BOLD)


def _techniques_tree(techniques: list[str]) -> str:
    """Pretty-print the evolved_techniques list as a tree."""
    if not techniques:
        return c("  (none recorded)", GREY)
    lines = []
    for i, t in enumerate(techniques):
        prefix = "  └─" if i == len(techniques) - 1 else "  ├─"
        lines.append(c(prefix, GREY) + c(f" {t}", CYAN))
    return "\n".join(lines)


def _divider(char: str = "─", width: int = 68) -> str:
    return c(char * width, GREY)


def _section(title: str) -> str:
    pad   = (66 - len(title)) // 2
    return (
        "\n" + _divider() + "\n"
        + c(" " * pad + title, BOLD, WHITE) + "\n"
        + _divider()
    )


# ── Database helpers ──────────────────────────────────────────────────────────

def _connect(db_path: str):
    """Return a SQLAlchemy engine for the given SQLite file."""
    try:
        from sqlalchemy import create_engine
    except ImportError:
        print(c("ERROR: SQLAlchemy is not installed.", RED, BOLD))
        print(c("       pip install sqlalchemy", GREY))
        sys.exit(1)

    if not os.path.exists(db_path):
        print(c(f"ERROR: Database file not found: {db_path}", RED, BOLD))
        print(c("       Run an audit from the dashboard first to generate data.", GREY))
        sys.exit(1)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    return engine


def _fetch_rows(engine, session_id: str | None, all_rows: bool) -> list[dict]:
    """Query audit_sessions and return a list of row dicts."""
    from sqlalchemy import text

    if session_id:
        sql = "SELECT * FROM audit_sessions WHERE session_id = :sid"
        params: dict = {"sid": session_id}
    elif all_rows:
        sql    = "SELECT * FROM audit_sessions ORDER BY end_time DESC"
        params = {}
    else:
        sql    = "SELECT * FROM audit_sessions ORDER BY end_time DESC LIMIT 1"
        params = {}

    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        cols   = list(result.keys())
        rows   = [dict(zip(cols, row)) for row in result.fetchall()]

    return rows


def _deserialise_row(row: dict) -> dict:
    """Deserialise JSON columns stored as text in SQLite."""
    evolved = row.get("evolved_techniques")
    if isinstance(evolved, str):
        try:
            row["evolved_techniques"] = json.loads(evolved)
        except (json.JSONDecodeError, TypeError):
            row["evolved_techniques"] = []
    elif evolved is None:
        row["evolved_techniques"] = []
    return row


# ── Printer ───────────────────────────────────────────────────────────────────

def _print_report(row: dict, index: int = 1, total: int = 1) -> None:
    row = _deserialise_row(row)

    sid      = row.get("session_id") or "—"
    model    = row.get("target_model") or "—"
    status   = row.get("attack_status") or "unknown"
    rahs     = float(row.get("rahs_score") or 0.0)
    prom     = float(row.get("prometheus_score") or 0.0)
    band     = row.get("severity_band") or "None"
    turns    = int(row.get("turn_count") or 0)
    duration = float(row.get("duration_seconds") or 0.0)
    technique= row.get("final_technique") or "—"
    patch    = (row.get("defense_patch") or "").strip()
    obj_text = (row.get("objective") or "").strip()
    evolved  = row.get("evolved_techniques") or []
    start_t  = row.get("start_time") or "—"
    end_t    = row.get("end_time") or "—"

    sev_col  = _severity_colour(band)

    header_label = f"AUDIT REPORT {index}/{total}" if total > 1 else "LATEST AUDIT REPORT"
    print(_section(header_label))

    # ── Identity block ──────────────────────────────────────────────────────
    print(f"\n  {c('Session ID   :', GREY)}  {c(sid, BOLD, WHITE)}")
    print(f"  {c('Target Model :', GREY)}  {c(model, BOLD, CYAN)}")
    print(f"  {c('Started      :', GREY)}  {c(start_t, DIM)}")
    print(f"  {c('Completed    :', GREY)}  {c(end_t, DIM)}")

    # ── Outcome ─────────────────────────────────────────────────────────────
    print(f"\n  {c('Outcome      :', GREY)}  {_status_icon(status)}")
    print(f"  {c('Severity     :', GREY)}  {c(f'■ {band}', sev_col, BOLD)}")

    # ── Scores ──────────────────────────────────────────────────────────────
    print(f"\n  {c('RAHS Score   :', GREY)}")
    print(f"    {_rahs_bar(rahs)}")

    prom_col = RED if prom >= 4 else YELLOW if prom >= 2 else GREEN
    prom_line = f"\n  {c('Prometheus   :', GREY)}  {c(f'{prom:.1f}/5.0', prom_col, BOLD)}"
    if prom >= 4:
        prom_line += c("  (jailbroken)", RED)
    print(prom_line)

    # ── Session stats ────────────────────────────────────────────────────────
    print(f"\n  {c('Turns        :', GREY)}  {c(str(turns), BOLD)}")
    print(f"  {c('Duration     :', GREY)}  {c(f'{duration:.1f}s', BOLD)}")
    print(f"  {c('Final PAP    :', GREY)}  {c(technique, MAGENTA)}")

    # ── Objective (truncated) ────────────────────────────────────────────────
    if obj_text:
        preview = obj_text[:120] + ("…" if len(obj_text) > 120 else "")
        print(f"\n  {c('Objective    :', GREY)}")
        print(f"    {c(preview, DIM)}")

    # ── evolved_techniques — the critical JSON column ────────────────────────
    print(f"\n  {c('Evolved Techniques  :', GREY, BOLD)}  "
          + c(f"({len(evolved)} mutation(s) recorded)", DIM))
    print(_techniques_tree(evolved))

    # ── JSON integrity verdict ───────────────────────────────────────────────
    if isinstance(evolved, list) and len(evolved) > 0:
        print(f"\n  {c('JSON column  :', GREY)}  {c('✔ Deserialised correctly', GREEN, BOLD)}")
    elif isinstance(evolved, list) and len(evolved) == 0:
        print(f"\n  {c('JSON column  :', GREY)}  {c('⚠  Empty list (session may have had 0 pruned techniques)', YELLOW)}")
    else:
        print(f"\n  {c('JSON column  :', GREY)}  {c('✘ Unexpected type: ' + type(evolved).__name__, RED, BOLD)}")

    # ── Defense patch (truncated) ────────────────────────────────────────────
    if patch:
        preview = patch[:300] + ("…" if len(patch) > 300 else "")
        print(f"\n  {c('Defense Patch:', GREY)}")
        for line in preview.splitlines()[:8]:
            print(f"    {c(line, DIM)}")

    print("\n" + _divider())


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PromptEvo Phase 2 — DB inspection tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        default="audit_reports.db",
        metavar="PATH",
        help="Path to the SQLite database file (default: audit_reports.db)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all sessions, ordered by most recent first",
    )
    parser.add_argument(
        "--session",
        metavar="UUID",
        default=None,
        help="Inspect a specific session by its UUID",
    )
    args = parser.parse_args()

    print(c("\n[*] PromptEvo -- Audit Database Inspector", BOLD, CYAN))
    print(c(f"    Database : {os.path.abspath(args.db)}", GREY))
    print(c(f"    Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", GREY))

    engine = _connect(args.db)

    try:
        rows = _fetch_rows(engine, session_id=args.session, all_rows=args.all)
    except Exception as exc:
        # Table may not exist yet (no audit run yet)
        if "no such table" in str(exc).lower():
            print(c("\nERROR: 'audit_sessions' table does not exist yet.", RED, BOLD))
            print(c("       Complete an audit from the dashboard to populate the DB.", GREY))
        else:
            print(c(f"\nERROR: {exc}", RED, BOLD))
        sys.exit(1)

    if not rows:
        print(c("\n  No audit records found.", YELLOW, BOLD))
        if args.session:
            print(c(f"  Session '{args.session}' does not exist in the database.", GREY))
        else:
            print(c("  Run an audit from the dashboard and then re-run this script.", GREY))
        print()
        sys.exit(0)

    total = len(rows)
    for i, row in enumerate(rows, start=1):
        _print_report(row, index=i, total=total)

    print(c(f"\n  ✔  Inspected {total} session(s) from {os.path.abspath(args.db)}\n", GREEN, BOLD))


if __name__ == "__main__":
    main()
