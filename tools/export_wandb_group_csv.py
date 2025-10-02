#!/usr/bin/env -S uv run
"""Export WandB runs for a group to CSV using WandbStore.

Usage:
  uv run ./tools/export_wandb_group_csv.py --group ak.vit_sweep.10012346 --output out.csv

Notes:
  - Requires WandB API access (ensure you are logged in: `wandb login` or have WANDB_API_KEY set).
  - Defaults to the repo's configured project/entity; override via CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

from metta.adaptive.stores.wandb import WandbStore
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dict using dotted keys.

    Lists and non-scalar values are JSON-encoded so they fit in CSV cells.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            if isinstance(v, (list, tuple)):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
    return dict(items)


def _collect_summary_keys(rows: Iterable[Dict[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for r in rows:
        for k in r.keys():
            keys.add(k)
    return sorted(keys)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export WandB runs for a group to CSV")
    parser.add_argument("--group", required=True, help="WandB group name to export (e.g., ak.vit_sweep.10012346)")
    parser.add_argument("--entity", default=METTA_WANDB_ENTITY, help="WandB entity (default from repo constants)")
    parser.add_argument("--project", default=METTA_WANDB_PROJECT, help="WandB project (default from repo constants)")
    parser.add_argument("--output", default="wandb_group_export.csv", help="Output CSV filepath")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of runs to fetch")

    args = parser.parse_args()

    store = WandbStore(entity=args.entity, project=args.project)

    # Fetch runs by group using the store API
    runs = store.fetch_runs({"group": args.group}, limit=args.limit)

    # Base fields from RunInfo
    base_fields = [
        "run_id",
        "group",
        "status",
        "has_started_training",
        "has_completed_training",
        "has_started_eval",
        "has_been_evaluated",
        "has_failed",
        "runtime",
        "cost",
        "total_timesteps",
        "current_steps",
        "created_at",
        "last_updated_at",
    ]

    # Build row dicts and collect flattened summary keys
    summary_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for r in runs:
        # Ensure datetimes serialize nicely
        created_at = r.created_at.isoformat() if isinstance(r.created_at, datetime) else r.created_at
        last_updated_at = r.last_updated_at.isoformat() if isinstance(r.last_updated_at, datetime) else r.last_updated_at

        base: Dict[str, Any] = {
            "run_id": r.run_id,
            "group": r.group,
            "status": str(r.status),
            "has_started_training": r.has_started_training,
            "has_completed_training": r.has_completed_training,
            "has_started_eval": r.has_started_eval,
            "has_been_evaluated": r.has_been_evaluated,
            "has_failed": r.has_failed,
            "runtime": r.runtime,
            "cost": r.cost,
            "total_timesteps": r.total_timesteps,
            "current_steps": r.current_steps,
            "created_at": created_at,
            "last_updated_at": last_updated_at,
        }

        summary = r.summary or {}
        if not isinstance(summary, dict):
            summary = {}
        summary_flat = _flatten_dict(summary)
        # Prefix summary keys to avoid collisions
        summary_prefixed = {f"summary.{k}": v for k, v in summary_flat.items()}

        summary_rows.append(summary_prefixed)
        csv_rows.append({**base, **summary_prefixed})

    # Determine CSV columns: base fields + all summary keys
    summary_columns = _collect_summary_keys(summary_rows)
    fieldnames = base_fields + summary_columns

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Exported {len(csv_rows)} runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

