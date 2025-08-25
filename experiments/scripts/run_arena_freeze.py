#!/usr/bin/env -S uv run
import argparse
import subprocess
import sys
from typing import List


def run_freeze_experiment(
    freeze_values: List[int], group_reward_pct: float | None, dry_run: bool
) -> int:
    base = [
        "./tools/run.py",
        "experiments.recipes.arena_freeze.train",
        "--args",
        "run=arena_freeze",
    ]

    ret = 0
    for val in freeze_values:
        cmd = base + [
            "--overrides",
            f"trainer.curriculum.task_generator.overrides.freeze_duration={val}",
        ]
        if group_reward_pct is not None:
            cmd.append(
                f"trainer.curriculum.task_generator.overrides.group_reward_pct={group_reward_pct}"
            )

        if dry_run:
            cmd.append("--dry-run")

        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            ret = proc.returncode
    return ret


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", nargs="*", type=int, default=[10, 50, 100])
    p.add_argument("--group-reward", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    return run_freeze_experiment(args.freeze, args.group_reward, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
