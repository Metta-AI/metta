#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from metta.common.util.lazypath import LazyPath

# Repo root path calculation
REPO_ROOT = Path(__file__).resolve().parents[5]  # Navigate up to repo root

PROD_STATS_SERVER_URI = "https://api.observatory.softmax-research.net"
DEV_STATS_SERVER_URI = "http://localhost:8000"
PROD_OBSERVATORY_FRONTEND_URL = "https://observatory.softmax-research.net"
DEV_OBSERVATORY_FRONTEND_URL = "http://localhost:5173"
METTA_WANDB_PROJECT = "metta"
METTA_WANDB_ENTITY = "metta-research"
METTA_GITHUB_ORGANIZATION = "Metta-AI"
METTA_GITHUB_REPO = "metta"
DEV_METTASCOPE_FRONTEND_URL = "http://localhost:8000"
METTASCOPE_REPLAY_URL = "https://metta-ai.github.io/metta"
METTA_AWS_ACCOUNT_ID = "751442549699"
METTA_AWS_REGION = "us-east-1"
METTA_SKYPILOT_URL = "skypilot-api.softmax-research.net"
SKYPILOT_LAUNCH_PATH = str(REPO_ROOT / "devops" / "skypilot" / "launch.py")
METTA_ENV_FILE = LazyPath(os.path.expanduser("~/.metta_env_path"))
SOFTMAX_S3_BUCKET = "softmax-public"
SOFTMAX_S3_BASE = f"s3://{SOFTMAX_S3_BUCKET}"
SOFTMAX_S3_POLICY_PREFIX = f"{SOFTMAX_S3_BASE}/policies"
SOFTMAX_S3_DATASET_PREFIX = f"{SOFTMAX_S3_BASE}/datasets"
SOFTMAX_S3_REPLAYS_PREFIX = f"{SOFTMAX_S3_DATASET_PREFIX}/replays"
RANK_ENV_VARS = [
    "SKYPILOT_NODE_RANK",  # SkyPilot clusters
    "RANK",  # PyTorch DDP
    "OMPI_COMM_WORLD_RANK",  # OpenMPI
]


def main():
    if len(sys.argv) != 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"Usage: {script_name} <CONSTANT_KEY>", file=sys.stderr)
        print("\nRetrieve Metta configuration constants.", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print(f"  {script_name} METTA_ENV_FILE", file=sys.stderr)
        print(f"  {script_name} METTA_AWS_REGION", file=sys.stderr)
        sys.exit(1)

    key = sys.argv[1]

    # Get all available constants (uppercase variables that don't start with _)
    available_keys = [k for k in globals() if k.isupper() and not k.startswith("_")]

    val = globals().get(key)
    if val is None:
        print(f"Error: Unknown constant '{key}'", file=sys.stderr)
        print("\nAvailable constants:", file=sys.stderr)
        for k in sorted(available_keys):
            print(f"  - {k}", file=sys.stderr)
        sys.exit(2)

    # If it's callable (e.g. LazyPath), call it
    if callable(val):
        val = val()

    # If it's a Path, cast to string
    if isinstance(val, Path):
        print(str(val))
    else:
        print(val)


if __name__ == "__main__":
    main()
