#!/usr/bin/env -S uv run
"""
Evaluate a ColorTree policy on each of the 16 possible two-color sequences (length 4)
and generate a heatmap of average scores per sequence.

This script mirrors tools/ct_seed_sweep.py but sweeps over explicit sequences
instead of seeds only. It supports multiple artifact versions and seeds; results
are aggregated per (version, sequence).

Examples:

  ./tools/ct_sequence_heatmap.py \
    --run-name-base <WANDB_RUN_NAME_BASE> \
    --versions v7 v8 v9 v10 \
    --seeds 1 2 3 4 5 \
    --env env/mettagrid/colortree_easy \
    --episodes 1 \
    --device cpu \
    --vectorization serial \
    --out-dir ./train_dir/ct_seq_heatmap

Outputs:
  - CSV: results.csv with columns [version, seed, sequence, score]
  - CSV: results_aggregated.csv with columns [version, sequence, avg_score, n]
  - PNG: heatmap_<version>.png (4x4 grid; rows are first two bits, cols are last two bits)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-sequence evaluation and heatmap for 2-color ColorTree")
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
        "--sim-name",
        default="ct_sequence_eval",
        help="Name for the sim suite (appears in metrics key)",
    )
    p.add_argument(
        "--out-dir",
        default="./train_dir/ct_sequence_heatmap",
        help="Directory to write CSVs and figures",
    )
    return p.parse_args()


def all_binary_sequences(length: int = 4) -> List[List[int]]:
    return [[(i >> k) & 1 for k in reversed(range(length))] for i in range(2**length)]


def sequence_to_str(seq: List[int]) -> str:
    return "".join(str(x) for x in seq)


def run_one(
    *,
    sim_name: str,
    run_uri_base: str,
    version: str,
    seed: int,
    env: str,
    episodes: int,
    device: str,
    vectorization: str,
    target_sequence: List[int],
) -> float | None:
    """Run one evaluation for a specific (version, seed, target_sequence)."""
    run_name = f"{sim_name}_{version}_s{seed}_seq{sequence_to_str(target_sequence)}"
    policy_uri = f"wandb://metta-research/metta/model/{run_uri_base}:{version}"

    # Hydra overrides to set the sequence; num_trials=1 makes base target active
    seq_list = ",".join(str(x) for x in target_sequence)
    cmd: list[str] = [
        sys.executable,
        "-m",
        "tools.sim",
        "sim=sim_suite",
        f"+sim.name={sim_name}",
        f"+sim.simulations.task.env={env}",
        f"sim.num_episodes={episodes}",
        f"seed={seed}",
        f"vectorization={vectorization}",
        f"device={device}",
        f"run={run_name}",
        f"policy_uri={policy_uri}",
        # ColorTree-specific overrides
        "+sim.simulations.task.env_overrides.game.actions.color_tree.num_trials=1",
        f"+sim.simulations.task.env_overrides.game.actions.color_tree.target_sequence=[{seq_list}]",
    ]

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


@dataclass
class ResultRow:
    version: str
    seed: int
    sequence: str
    score: float | None


def aggregate_by_sequence(rows: List[ResultRow]) -> list[tuple[str, str, float | None, int]]:
    """Aggregate results by (version, sequence). Returns tuples of
    (version, sequence, avg_score_or_None, n)."""
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.score is None:
            continue
        key = (r.version, r.sequence)
        by_key.setdefault(key, []).append(r.score)

    out: list[tuple[str, str, float | None, int]] = []
    versions = sorted({r.version for r in rows})
    sequences = sorted({r.sequence for r in rows})
    for v in versions:
        for s in sequences:
            scores = by_key.get((v, s), [])
            if scores:
                avg = sum(scores) / len(scores)
                out.append((v, s, avg, len(scores)))
            else:
                out.append((v, s, None, 0))
    return out


def write_csvs(out_dir: Path, rows: List[ResultRow]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "results.csv"
    with raw_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "seed", "sequence", "score"])
        for r in rows:
            w.writerow([r.version, r.seed, r.sequence, r.score])

    agg = aggregate_by_sequence(rows)
    agg_path = out_dir / "results_aggregated.csv"
    with agg_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "sequence", "avg_score", "n"])
        for v, s, avg, n in agg:
            w.writerow([v, s, avg if avg is not None else "NA", n])

    return raw_path


def plot_heatmaps(out_dir: Path, agg_rows: list[tuple[str, str, float | None, int]]) -> None:
    """Create one 4x4 heatmap per version.

    Rows correspond to the first two bits of the sequence (00..11), columns to the last two bits.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        try:
            import seaborn as sns  # optional
        except Exception:
            sns = None
    except Exception:
        print("matplotlib not available; skipping heatmap generation")
        return

    versions = sorted({v for (v, _s, _avg, _n) in agg_rows})

    # Organize by version -> matrix
    for v in versions:
        # Initialize with NaNs so missing entries are visible
        mat = np.full((4, 4), np.nan, dtype=float)
        for _v, s, avg, _n in agg_rows:
            if _v != v:
                continue
            if len(s) != 4 or any(ch not in "01" for ch in s):
                continue
            r = int(s[:2], 2)
            c = int(s[2:], 2)
            if avg is not None:
                mat[r, c] = avg

        fig, ax = plt.subplots(figsize=(6, 5))
        if "sns" in locals() and sns is not None:
            sns.heatmap(
                mat,
                ax=ax,
                cmap="viridis",
                annot=True,
                fmt=".2f",
                xticklabels=["00", "01", "10", "11"],
                yticklabels=["00", "01", "10", "11"],
                cbar_kws={"label": "avg score"},
            )
        else:
            im = ax.imshow(mat, cmap="viridis")
            ax.set_xticks([0, 1, 2, 3], ["00", "01", "10", "11"])
            ax.set_yticks([0, 1, 2, 3], ["00", "01", "10", "11"])
            for i in range(4):
                for j in range(4):
                    val = mat[i, j]
                    txt = "" if (val != val) else f"{val:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", color="w", fontsize=9)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("avg score")

        ax.set_xlabel("last two bits")
        ax.set_ylabel("first two bits")
        ax.set_title(f"Sequence heatmap · {v}")
        fig.tight_layout()
        fig_path = out_dir / f"heatmap_{v}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {fig_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = all_binary_sequences(4)  # 16 sequences for 2 colors

    rows: list[ResultRow] = []
    for v in args.versions:
        for seed in args.seeds:
            for seq in sequences:
                score = run_one(
                    sim_name=args.sim_name,
                    run_uri_base=args.run_name_base,
                    version=v,
                    seed=seed,
                    env=args.env,
                    episodes=args.episodes,
                    device=args.device,
                    vectorization=args.vectorization,
                    target_sequence=seq,
                )
                rows.append(
                    ResultRow(
                        version=v,
                        seed=seed,
                        sequence=sequence_to_str(seq),
                        score=score,
                    )
                )

    raw_csv_path = write_csvs(out_dir, rows)
    print(f"Wrote: {raw_csv_path}")

    agg_rows = aggregate_by_sequence(rows)
    plot_heatmaps(out_dir, agg_rows)


if __name__ == "__main__":
    sys.exit(main())
