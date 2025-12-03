"""
Launch a targeted learning-progress grid sweep from the shell.

Primarily intended for quick local experiments where the orchestration cost of
`metta.tools.sweep` is unnecessary. The script enumerates a Cartesian product of
hyperparameter overrides, builds `uv run ./tools/run.py …` commands, and runs
them sequentially (or prints them with `--dry-run`).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_VARIANTS = ("heart_chorus", "lonely_heart")

MAP_RECIPES = {
    "proc": "recipes.experiment.cvc.proc_maps.train",
    "fixed": "recipes.prod.cvc.fixed_maps.train",
}

LAUNCHER_PREFIXES = {
    "devops": ["uv", "run", "./devops/run.sh"],
    "tools": ["uv", "run", "./tools/run.py"],
}


@dataclass(frozen=True)
class SweepDimension:
    """Simple container describing a sweep axis."""

    key: str
    values: Sequence[object]


SWEEP_DIMENSIONS: Sequence[SweepDimension] = (
    SweepDimension("ema_timescale", [0.001, 0.005, 0.01]),
    SweepDimension("progress_smoothing", [0.01, 0.05, 0.1]),
    SweepDimension("exploration_bonus", [0.05, 0.1]),
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _format_value(value: object) -> str:
    """Format override values for CLI consumption."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _format_override(key: str, value: object) -> str:
    return f"{key}={_format_value(value)}"


def _build_base_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {
        "group": args.run_group,
        "trainer.total_timesteps": args.total_timesteps,
        "checkpointer.epoch_interval": args.epoch_interval,
    }
    if args.num_cogs is not None:
        overrides["num_cogs"] = args.num_cogs
    if args.variant_override is not None:
        overrides["variants"] = args.variant_override
    return overrides


def _enumerate_commands(args: argparse.Namespace) -> Iterable[list[str]]:
    """Yield fully materialized commands for each sweep point."""
    axes = [dim.values for dim in SWEEP_DIMENSIONS]
    base_overrides = _build_base_overrides(args)
    for idx, combo in enumerate(itertools.product(*axes), start=1):
        if args.max_runs is not None and idx > args.max_runs:
            break

        overrides = dict(base_overrides)
        for dim, value in zip(SWEEP_DIMENSIONS, combo, strict=False):
            overrides[dim.key] = value

        run_name = f"lp_grid_{idx:03d}"
        overrides["run"] = run_name

        cmd = list(LAUNCHER_PREFIXES[args.launcher])
        cmd.append(args.recipe)
        cmd.extend(_format_override(k, v) for k, v in overrides.items())
        yield cmd


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a local LP sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Cap the number of configurations to launch (defaults to the full grid).",
    )
    parser.add_argument(
        "--maps",
        choices=sorted(MAP_RECIPES.keys()),
        default="proc",
        help="Select which CVC recipe to target.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Override recipe path (takes precedence over --maps).",
    )
    parser.add_argument(
        "--launcher",
        choices=sorted(LAUNCHER_PREFIXES.keys()),
        default="devops",
        help="Choose whether to launch via devops/run.sh or the direct TrainTool.",
    )
    parser.add_argument("--num-cogs", type=int, default=4, help="Override num_cogs when supported by the recipe.")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="JSON list of variants to train on (default: heart_chorus + lonely_heart).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total timesteps per run (trainer.total_timesteps).",
    )
    parser.add_argument(
        "--epoch-interval",
        type=int,
        default=10,
        help="Checkpoint epoch interval (checkpointer.epoch_interval).",
    )
    parser.add_argument("--run-group", default="lp_local_grid", help="Group label for WandB/stats.")
    args = parser.parse_args()

    args.recipe = args.recipe or MAP_RECIPES[args.maps]
    if args.variants is None:
        variant_list = list(DEFAULT_VARIANTS)
    else:
        loaded = json.loads(args.variants)
        if not isinstance(loaded, list):
            raise ValueError("--variants must decode to a JSON list")
        variant_list = [str(item) for item in loaded]
    args.variant_override = json.dumps(variant_list) if variant_list else None
    return args


def main() -> None:
    args = parse_args()
    for cmd in _enumerate_commands(args):
        print("Launching:", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
