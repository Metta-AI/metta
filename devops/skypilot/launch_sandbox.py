#!/usr/bin/env -S uv run
"""Run a Metta training/eval job inside an existing multi-node SkyPilot sandbox.

This utility wraps `sky exec` for a running sandbox cluster and executes a
distributed job across N nodes using the repo's `./devops/run.sh` wrapper.

Typical usage:

  ./devops/skypilot/launch_sandbox.py subho-sandbox-3 \
    experiments.recipes.arena_basic_easy_shaped.train \
    --nodes 4 --run my_dist_run -- trainer.total_timesteps=1_000_000

It will:
  - Activate the repo venv in the container (creating it if missing)
  - Derive NUM_NODES, NODE_INDEX, MASTER_ADDR from SkyPilot env vars
  - Unset CUDA_VISIBLE_DEVICES (to expose GPUs) and set NUM_GPUS per node
  - Invoke `./devops/run.sh <module> [args...]` on all nodes

Notes:
  - This does not provision a cluster. Use `./devops/skypilot/sandbox.py --new` first.
  - Prefer the full launcher (`./devops/skypilot/launch.py`) for non-sandbox runs.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from typing import List

from metta.common.tool.tool_path import validate_module_path
from metta.tools.utils.auto_config import auto_run_name


def _build_remote_script(
    module_path: str,
    tool_args: List[str],
    unset_cuda_visible_devices: bool,
    bootstrap: bool,
) -> str:
    """Build the bash script executed on each sandbox node.

    The script ensures a usable venv, wires SkyPilot env into torchrun
    variables, exposes GPUs, and launches `./devops/run.sh` with the provided
    module path and tool arguments.
    """

    # Join tool args safely for bash; they are later consumed by run.py
    args_str = " ".join(shlex.quote(a) for a in tool_args)

    lines: list[str] = [
        "set -euo pipefail",
        "REPO=/workspace/metta",
        # Ensure repo exists (in case sandbox was created without setup run)
        "if [ ! -d \"$REPO/devops\" ]; then",
        "  mkdir -p /workspace",
        "  cd /workspace",
        "  if [ ! -d metta ]; then",
        "    echo \"[BOOTSTRAP] Cloning metta repo...\"",
        "    git clone https://github.com/Metta-AI/metta.git metta",
        "  fi",
        "fi",
        "cd /workspace/metta",
        # Ensure uv exists in PATH on the sandbox
        "if ! command -v uv >/dev/null 2>&1; then",
        "  echo \"[BOOTSTRAP] Installing uv...\"",
        "  curl -LsSf https://astral.sh/uv/install.sh | sh",
        "  export PATH=\"$HOME/.local/bin:$PATH\"",
        "fi",
        # Create venv if missing
        "if [ ! -d .venv ]; then",
        "  echo \"[BOOTSTRAP] Creating venv with uv sync...\"",
        "  uv sync --locked || uv sync",
        "fi",
        ". .venv/bin/activate",
        # Wire SkyPilot distributed env into torchrun
        "export NUM_NODES=\"${SKYPILOT_NUM_NODES:-1}\"",
        "export NODE_INDEX=\"${SKYPILOT_NODE_RANK:-0}\"",
        # Prefer explicit head IP if present; fall back to first node IP
        "export MASTER_ADDR=\"${SKYPILOT_RAY_HEAD_IP:-}\"",
        "if [ -z \"$MASTER_ADDR\" ]; then MASTER_ADDR=\"$(echo \"${SKYPILOT_NODE_IPS:-127.0.0.1}\" | head -n1)\"; fi",
        # Use a stable default master port unless overridden
        "export MASTER_PORT=\"${MASTER_PORT:-29501}\"",
    ]

    if unset_cuda_visible_devices:
        lines.append("unset CUDA_VISIBLE_DEVICES")

    # Detect GPUs per node reliably via nvidia-smi; default to 1 if unavailable
    lines.extend(
        [
            "export NUM_GPUS=\"$(nvidia-smi -L | wc -l || echo 1)\"",
            "echo \"[DEBUG] $(hostname) NUM_NODES=$NUM_NODES NODE_INDEX=$NODE_INDEX MASTER_ADDR=$MASTER_ADDR NUM_GPUS=$NUM_GPUS\"",
            # Launch the distributed job
            f"./devops/run.sh {shlex.quote(module_path)} {args_str}".rstrip(),
        ]
    )

    script = "\n".join(lines)
    if not bootstrap:
        # If user opts out of bootstrap, strip repo/venv creation lines
        # Keep activation and execution; sandbox setup should have prepared the env
        keep_prefixes = (
            "set -euo pipefail",
            "REPO=/workspace/metta",  # harmless
            "cd /workspace/metta",
            ". .venv/bin/activate",
            "export NUM_NODES=",
            "export NODE_INDEX=",
            "export MASTER_ADDR=",
            "if [ -z \"$MASTER_ADDR\" ]",
            "unset CUDA_VISIBLE_DEVICES" if unset_cuda_visible_devices else "",
            "export NUM_GPUS=",
            "echo \"[DEBUG]",
            f"./devops/run.sh {shlex.quote(module_path)}",
        )
        script = "\n".join(l for l in lines if any(l.startswith(p) for p in keep_prefixes if p))
    return script


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a Metta job on an existing SkyPilot sandbox across N nodes.\n"
            "Wraps `sky exec` and launches ./devops/run.sh with the provided module."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s my-sandbox --nodes 4 arena.train -- run=test trainer.total_timesteps=1_000_000\n"
            "  %(prog)s subho-sandbox-3 --nodes 4 experiments.recipes.arena_basic_easy_shaped.train -- --verbose"
        ),
    )

    parser.add_argument("cluster", type=str, help="SkyPilot cluster (sandbox) name, e.g., user-sandbox-1")
    parser.add_argument("module_path", type=str, help="Tool module path (e.g., arena.train or experiments.recipes.arena.train)")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to exec across (integer)")
    parser.add_argument("--run", type=str, default=None, help="Run ID (auto-generated if omitted)")
    parser.add_argument("--master-port", type=int, default=29501, help="Master port for torchrun rendezvous (default: 29501)")
    parser.add_argument("--unset-cuda-visible-devices", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--keep-cuda-visible-devices", action="store_true", help="Do not unset CUDA_VISIBLE_DEVICES")
    parser.add_argument("--bootstrap", action="store_true", help="Create venv/repo on nodes if missing")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit")
    parser.add_argument("--confirm", action="store_true", help="Prompt before launching")

    # Robustly split tool args: everything after the first "--" goes to tools/run.py
    argv = sys.argv[1:]
    if "--" in argv:
        sep_idx = argv.index("--")
        base_argv = argv[:sep_idx]
        tool_args = argv[sep_idx + 1 :]
    else:
        base_argv = argv
        tool_args = []

    args = parser.parse_args(base_argv)

    # Normalize module path and validate it exists
    module_path = args.module_path
    if not validate_module_path(module_path):
        print(f"Invalid module path: {module_path}", file=sys.stderr)
        return 2

    # Inject run= if provided or auto-generate one
    tool_args = list(tool_args)
    # Be defensive: strip any mistakenly provided launcher flags from tool args
    sanitized: list[str] = []
    skip_next = False
    for i, tok in enumerate(tool_args):
        if skip_next:
            skip_next = False
            continue
        if tok == "--nodes" and i + 1 < len(tool_args):
            skip_next = True
            continue
        sanitized.append(tok)
    tool_args = sanitized
    has_run_arg = any(a.startswith("run=") for a in tool_args)
    run_id = args.run or auto_run_name()
    if not has_run_arg:
        tool_args.append(f"run={run_id}")

    unset_cvd = not args.keep_cuda_visible_devices
    # Inject MASTER_PORT into environment via preamble line in remote script; we set a default here but allow override
    # Build remote script body
    remote_script = _build_remote_script(
        module_path=module_path,
        tool_args=tool_args,
        unset_cuda_visible_devices=unset_cvd,
        bootstrap=args.bootstrap,
    )
    # Prepend explicit MASTER_PORT export so it overrides defaults inside the script
    remote_script = f"MASTER_PORT={args.master_port}\n" + remote_script

    exec_cmd = [
        "uv",
        "run",
        "sky",
        "exec",
        args.cluster,
        "--num-nodes",
        str(args.nodes),
        "--",
        "bash",
        "-lc",
        remote_script,
    ]

    if args.dry_run:
        print("Command to run:\n" + " ".join(shlex.quote(c) for c in exec_cmd))
        return 0

    if args.confirm:
        yn = input(f"Launch on cluster '{args.cluster}' across {args.nodes} nodes? [y/N] ").strip().lower()
        if yn not in {"y", "yes"}:
            return 0

    # Execute and stream output
    try:
        subprocess.run(exec_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"launch_sandbox: sky exec failed with code {e.returncode}", file=sys.stderr)
        return e.returncode or 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
