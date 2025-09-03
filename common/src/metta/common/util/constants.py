#!/usr/bin/env python3
import os
import sys
from pathlib import Path

from metta.common.util.lazypath import LazyPath

PROD_STATS_SERVER_URI = "https://api.observatory.softmax-research.net"
DEV_STATS_SERVER_URI = "http://localhost:8000"

PROD_OBSERVATORY_FRONTEND_URL = "https://observatory.softmax-research.net"
DEV_OBSERVATORY_FRONTEND_URL = "http://localhost:5173"

METTA_WANDB_PROJECT = "metta"
METTA_WANDB_ENTITY = "metta-research"


DEV_METTASCOPE_FRONTEND_URL = "http://localhost:8000"
METTASCOPE_REPLAY_URL = "https://metta-ai.github.io/metta"

METTA_AWS_ACCOUNT_ID = "751442549699"
METTA_AWS_REGION = "us-east-1"

METTA_SKYPILOT_URL = "skypilot-api.softmax-research.net"

METTA_ENV_FILE = LazyPath(os.path.expanduser("~/.metta_env_path"))

SOFTMAX_S3_BUCKET = "softmax-public"
SOFTMAX_S3_BASE = f"s3://{SOFTMAX_S3_BUCKET}"

RANK_ENV_VARS = [
    "SKYPILOT_NODE_RANK",  # SkyPilot clusters
    "RANK",  # PyTorch DDP
    "OMPI_COMM_WORLD_RANK",  # OpenMPI
]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <KEY>", file=sys.stderr)
        sys.exit(1)

    key = sys.argv[1]
    val = globals().get(key)

    if val is None:
        print(f"Error: no such key '{key}'", file=sys.stderr)
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
