#!/usr/bin/env -S uv run
"""
Run a set of evaluations over multiple policy versions and seeds, and print a
concise summary at the end. Also writes a CSV of results.

Usage example:

  ./tools/ct_seed_sweep.py \
    --run-name-base jacke.colortree_recurrent_jacke_2color_precise_random_bptt64_easy_lstm_20250808_074647 \
    --versions v7 v8 v9 v10 \
    --seeds 1 2 3 4 5 \
    --env env/mettagrid/colortree_easy \
    --episodes 1 \
    --device cpu \
    --vectorization serial \
    --out ./train_dir/ct_easy_seed_sweep/results.csv

This tool captures and parses the JSON block printed by tools.sim and prints a
final table of scores per (version, seed) and per-version averages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed sweep evaluator with concise end-of-run summary")
    p.add_argument(
        "--run-name-base",
        required=True,
        help="Base run name for the wandb artifact (without :version)",
    )
    p.add_argument(
        "--versions",
        nargs="+",
        required=True,
        help="Artifact versions to evaluate (e.g., v7 v8 v9)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Seeds to evaluate (e.g., 1 2 3 4 5)",
    )
    p.add_argument(
        "--env",
        default="env/mettagrid/colortree_easy",
        help="Env path used in the sim suite (defaults to colortree_easy)",
    )
    p.add_argument("--episodes", type=int, default=1, help="Episodes per run")
    p.add_argument("--device", default="cpu", help="Device for evaluation")
    p.add_argument(
        "--vectorization",
        default="serial",
        choices=["serial", "multiprocessing"],
        help="Vectorization backend",
    )
    p.add_argument(
        "--out",
        default="./train_dir/ct_seed_sweep/results.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--sim-name",
        default="ct_seed_sweep",
        help="Name for the sim suite (appears in metrics key)",
    )
    return p.parse_args()


def run_one(version: str, seed: int, env: str, episodes: int, device: str, vectorization: str, run_uri_base: str, sim_name: str) -> float | None:
    run_name = f"{sim_name}_{version}_s{seed}"
    policy_uri = f"wandb://metta-research/metta/model/{run_uri_base}:{version}"

    cmd: list[str] = [
        sys.executable,
        "-m",
        "tools.sim",
        "sim=sim_suite",
        f"+sim.name={sim_name}",
        f"+sim.simulations.task.env={env}",  # use a neutral key name 'task'
        f"sim.num_episodes={episodes}",
        f"seed={seed}",
        f"vectorization={vectorization}",
        f"device={device}",
        f"run={run_name}",
        f"policy_uri={policy_uri}",
    ]

    # Capture output and parse the JSON block only
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout

    start = out.find("===JSON_OUTPUT_START===")
    end = out.find("===JSON_OUTPUT_END===")
    if start == -1 or end == -1:
        return None
    payload = out[start + len("===JSON_OUTPUT_START===") : end].strip()
    try:
        data = json.loads(payload)
        ckpt = data["policies"][0]["checkpoints"][0]
        score = ckpt["metrics"].get("reward_avg")
        return float(score) if score is not None else None
    except Exception:
        return None


def print_summary(rows: list[dict[str, object]], versions: Iterable[str]) -> None:
    # Compute per-version averages
    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        v = str(r["version"])  # type: ignore[arg-type]
        s = r["score"]
        if isinstance(s, (int, float)):
            sums[v] += float(s)
            counts[v] += 1

    # Pretty print
    print()
    print("Results (version, seed, score):")
    for r in rows:
        print(f"  {r['version']}, {r['seed']}, {r['score']}")

    print()
    print("Averages by version:")
    for v in versions:
        n = counts[v]
        avg = (sums[v] / n) if n else None
        print(f"  {v}: {avg if avg is not None else 'NA'} (n={n})")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for v in args.versions:
        for seed in args.seeds:
            score = run_one(
                version=v,
                seed=seed,
                env=args.env,
                episodes=args.episodes,
                device=args.device,
                vectorization=args.vectorization,
                run_uri_base=args.run_name_base,
                sim_name=args.sim_name,
            )
            rows.append({"version": v, "seed": seed, "score": score})

    # Write CSV
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "seed", "score"])
        for r in rows:
            writer.writerow([r["version"], r["seed"], r["score"]])

    # Print concise summary to stdout at the end
    print_summary(rows, args.versions)
    print()
    print(f"CSV written to: {out_path}")


if __name__ == "__main__":
    main()




