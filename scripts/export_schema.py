#!/usr/bin/env python3
"""Export and verify frozen architecture schema artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = ROOT / "schemas"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def export_state_schema() -> dict:
    from core.state import ALL_FIELDS, INTELLIGENCE_FIELDS, GROOMING_FIELDS

    return {
        "version": (SCHEMAS / "SCHEMA_VERSION").read_text(encoding="utf-8").strip(),
        "all_fields": sorted(ALL_FIELDS),
        "intelligence_fields": sorted(INTELLIGENCE_FIELDS),
        "grooming_fields": sorted(GROOMING_FIELDS),
        "field_count": len(ALL_FIELDS),
    }


def export_topology_spec() -> dict:
    from core.graph import (
        _GRAPH_CONDITIONAL_ROUTES,
        _GRAPH_UNCONDITIONAL_EDGES,
        _GRAPH_USER_NODES,
        compute_topology_hash,
        get_app,
    )

    routes = _GRAPH_CONDITIONAL_ROUTES
    if isinstance(routes, dict):
        conditional = {k: sorted(v) for k, v in routes.items()}
    else:
        conditional = list(routes)

    return {
        "nodes": sorted(_GRAPH_USER_NODES),
        "unconditional_edges": _GRAPH_UNCONDITIONAL_EDGES,
        "conditional_routes": conditional,
        "topology_hash": compute_topology_hash(get_app()),
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_all() -> None:
    _write_json(SCHEMAS / "state_schema_v1.json", export_state_schema())
    _write_json(SCHEMAS / "topology_spec_v1.json", export_topology_spec())
    print(f"Exported schemas to {SCHEMAS}")


def check_all() -> int:
    errors: list[str] = []
    live_state = json.dumps(export_state_schema(), sort_keys=True)
    live_topo = json.dumps(export_topology_spec(), sort_keys=True)

    state_path = SCHEMAS / "state_schema_v1.json"
    topo_path = SCHEMAS / "topology_spec_v1.json"

    if state_path.exists():
        committed = state_path.read_text(encoding="utf-8")
        if json.dumps(json.loads(committed), sort_keys=True) != live_state:
            errors.append("state_schema_v1.json drift — run export_schema.py or bump SCHEMA_VERSION")
    else:
        errors.append("state_schema_v1.json missing — run export_schema.py")

    if topo_path.exists():
        committed = topo_path.read_text(encoding="utf-8")
        if json.dumps(json.loads(committed), sort_keys=True) != live_topo:
            errors.append("topology_spec_v1.json drift — topology or node list changed")
    else:
        errors.append("topology_spec_v1.json missing — run export_schema.py")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("Schema check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export or verify PromptEvo schema artifacts")
    parser.add_argument("--check", action="store_true", help="Verify committed schemas match live code")
    parser.add_argument("--export", action="store_true", help="Write schema JSON files")
    args = parser.parse_args()

    if args.check:
        return check_all()
    if args.export or not args.check:
        export_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
