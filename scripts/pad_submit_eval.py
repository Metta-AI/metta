#!/usr/bin/env python
"""
Pad narrow-action checkpoints, submit them to CoGames, and optionally run a local eval.

Usage:
  AWS_PROFILE=softmax uv run python scripts/pad_submit_eval.py --submit --eval

Flags:
  --submit    Perform cogames submit for each padded checkpoint.
  --eval      Run local v0_leaderboard.evaluate against each padded checkpoint.
  --runs ...  Override the default run list (space-separated).

Notes:
  - Padding adds zero weights and very negative bias (-1e9) for extra actions,
    so padded actions stay effectively disabled.
  - Submissions upload padded checkpoints to S3 and submit using the S3 URI.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import boto3
import torch
from botocore.config import Config

from mettagrid.policy.mpt_artifact import load_mpt, save_mpt

TARGET_ACTIONS = 157
PAD_BIAS = -1e9
S3_BUCKET = "softmax-public"
S3_PREFIX = "policies"

# Default runs to process (latest checkpoint per run will be used)
DEFAULT_RUNS: list[str] = [
    "relh.machina1_bc_dinky_sup.hc.1209.0.5.hc.1",
    "relh.machina1_bc_thinky_sup.1209.0_5.10",
    "relh.machina1_bc_dinky_sliced.hc.1209.13",
    "relh.machina1_bc_thinky_sliced.hc.1209.11",
    "relh.machina1_bc_dinky_sliced.hc.1209.12",
]


def _find_latest_mpt_key(s3, run: str) -> str:
    prefix = f"{S3_PREFIX}/{run}/"
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    if resp.get("KeyCount", 0) == 0:
        raise RuntimeError(f"No checkpoints found for {run}")
    best_version: Tuple[int, str] | None = None
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".mpt"):
            continue
        # Expect ...:v<number>.mpt
        match = re.search(r":v(\d+)\.mpt$", key)
        if not match:
            continue
        v = int(match.group(1))
        if best_version is None or v > best_version[0]:
            best_version = (v, key)
    if not best_version:
        raise RuntimeError(f"No versioned .mpt found for {run}")
    return best_version[1]


def pad_checkpoint(src_uri: str, out_path: Path) -> Path:
    artifact = load_mpt(src_uri)
    state = artifact.state_dict

    weight_key = None
    bias_key = None
    for k, v in state.items():
        if "actor_head" in k and v.ndim == 2:
            weight_key = k
        if "actor_head" in k and v.ndim == 1:
            bias_key = k
    if weight_key is None or bias_key is None:
        raise RuntimeError("Could not locate actor_head weight/bias in checkpoint")

    cur_w = state[weight_key]
    cur_b = state[bias_key]

    if cur_w.shape[0] == TARGET_ACTIONS:
        # Already at target; just copy
        return Path(save_mpt(out_path, architecture=artifact.architecture, state_dict=state))
    if cur_w.shape[0] > TARGET_ACTIONS:
        raise RuntimeError(f"Checkpoint has more actions ({cur_w.shape[0]}) than target {TARGET_ACTIONS}")

    pad = TARGET_ACTIONS - cur_w.shape[0]
    pad_w = torch.zeros((pad, cur_w.shape[1]), dtype=cur_w.dtype)
    pad_b = torch.full((pad,), PAD_BIAS, dtype=cur_b.dtype)

    state[weight_key] = torch.cat([cur_w, pad_w], dim=0)
    state[bias_key] = torch.cat([cur_b, pad_b], dim=0)

    saved_uri = save_mpt(out_path, architecture=artifact.architecture, state_dict=state)
    return Path(saved_uri.replace("file://", ""))


def submit_checkpoint(path_or_uri: str | Path, name: str) -> None:
    # CoGames submit accepts URIs; add file:// for local paths.
    path_str = str(path_or_uri)
    uri = path_str if "://" in path_str else f"file://{path_str}"
    cmd = ["uv", "run", "cogames", "submit", "-p", uri, "-n", name]
    subprocess.run(cmd, check=True)


def local_eval(path: Path, seed: int = 50) -> None:
    uri = f"file://{path}"
    cmd = [
        "uv",
        "run",
        "tools/run.py",
        "recipes.experiment.v0_leaderboard.evaluate",
        f"policy_uri={uri}",
        f"seed={seed}",
    ]
    subprocess.run(cmd, check=True)


def upload_padded_checkpoint(s3, local_path: Path, run: str, version: str) -> str:
    key = f"{S3_PREFIX}/{run}/{run}:v{version}.padded_neginf.mpt"
    dest_uri = f"s3://{S3_BUCKET}/{key}"
    print(f"[{run}] uploading padded checkpoint to {dest_uri}")
    s3.upload_file(str(local_path), S3_BUCKET, key)
    return dest_uri


def main(runs: Iterable[str], do_submit: bool, do_eval: bool) -> None:
    s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
    workdir = Path(tempfile.mkdtemp(prefix="padded_mpt_"))

    for run in runs:
        try:
            key = _find_latest_mpt_key(s3, run)
        except RuntimeError as e:
            print(f"[{run}] skipping: {e}", file=sys.stderr)
            continue
        src_uri = f"s3://{S3_BUCKET}/{key}"
        version = key.split(":v")[-1].split(".mpt")[0]
        out_path = workdir / f"{run}.v{version}.padded_neginf.mpt"

        print(f"[{run}] latest: {src_uri} -> padding to {out_path}")
        padded = pad_checkpoint(src_uri, out_path)
        padded_s3_uri = upload_padded_checkpoint(s3, padded, run, version) if do_submit else None

        if do_submit:
            name = f"{run}-neginf"
            print(f"[{run}] submitting as {name}")
            submit_checkpoint(padded_s3_uri or padded, name)

        if do_eval:
            print(f"[{run}] running local eval")
            local_eval(padded)

    print(f"Done. Padded files in {workdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Submit each padded checkpoint")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run local v0_leaderboard evaluate on each padded checkpoint",
    )
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="Run names to process")
    args = parser.parse_args()

    try:
        main(args.runs, do_submit=args.submit, do_eval=args.eval)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        sys.exit(e.returncode)
