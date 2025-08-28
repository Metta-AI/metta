#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

from metta.common.util.constants import METTA_ENV_FILE, PROD_OBSERVATORY_FRONTEND_URL, PROD_STATS_SERVER_URI


def run_command(cmd, capture_output=True):
    """Run a shell command and return the output."""
    if isinstance(cmd, str):
        cmd = cmd.split()

    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    if result.returncode != 0 and not capture_output:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result.stdout.strip() if capture_output else None


def setup_job_metadata():
    """Setup job metadata tracking (restart count, accumulated runtime)."""
    data_dir = os.environ.get("DATA_DIR", "./train_dir")
    metta_run_id = os.environ.get("METTA_RUN_ID", "default")
    job_metadata_dir = Path(data_dir) / ".job_metadata" / metta_run_id
    job_metadata_dir.mkdir(parents=True, exist_ok=True)

    restart_count_file = job_metadata_dir / "restart_count"
    accumulated_runtime_file = job_metadata_dir / "accumulated_runtime"
    heartbeat_file = Path(os.environ.get("HEARTBEAT_FILE", job_metadata_dir / "heartbeat_file"))

    # Calculate IS_MASTER based on node rank
    node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
    is_master = node_index == 0

    # Handle restart count
    if restart_count_file.exists():
        restart_count = int(restart_count_file.read_text())
        restart_count += 1
    else:
        restart_count = 0

    # Only update restart count on master node
    if is_master:
        restart_count_file.write_text(str(restart_count))
    else:
        print(f"[INFO] Skipping RESTART_COUNT_FILE updates on non-master node {node_index}")

    # Read accumulated runtime
    if accumulated_runtime_file.exists():
        accumulated_runtime = int(accumulated_runtime_file.read_text())
    else:
        accumulated_runtime = 0

    if is_master:
        print("============= RESTART INFO =============")
        print(f"  METTA_RUN_ID: {metta_run_id}")
        print(f"  RESTART_COUNT: {restart_count}")
        print(f"  ACCUMULATED_RUNTIME: {accumulated_runtime}s ({accumulated_runtime // 60}m)")
        print("========================================")

    return {
        "restart_count": restart_count,
        "accumulated_runtime": accumulated_runtime,
        "accumulated_runtime_file": str(accumulated_runtime_file),
        "heartbeat_file": str(heartbeat_file),
    }


def write_environment_variables(metta_env_file, metadata):
    """Write environment variables to METTA_ENV_FILE."""
    env_vars = f'''export PYTHONUNBUFFERED=1
export PYTHONPATH="${{PYTHONPATH:+$PYTHONPATH:}}$(pwd)"
export PYTHONOPTIMIZE=1
export HYDRA_FULL_ERROR=1

export WANDB_DIR="./wandb"
export WANDB_API_KEY="${{WANDB_PASSWORD}}"
export DATA_DIR="${{DATA_DIR:-./train_dir}}"

# Datadog configuration
export DD_ENV="production"
export DD_SERVICE="skypilot-worker"
export DD_AGENT_HOST="localhost"
export DD_TRACE_AGENT_PORT="8126"

export NUM_GPUS="${{SKYPILOT_NUM_GPUS_PER_NODE}}"
export NUM_NODES="${{SKYPILOT_NUM_NODES}}"
export MASTER_ADDR="$(echo "$SKYPILOT_NODE_IPS" | head -n1)"
export MASTER_PORT="${{MASTER_PORT:-29501}}"
export NODE_INDEX="${{SKYPILOT_NODE_RANK}}"

# Job metadata exports
export RESTART_COUNT="{metadata["restart_count"]}"
export ACCUMULATED_RUNTIME="{metadata["accumulated_runtime"]}"

# File path exports for monitors
export ACCUMULATED_RUNTIME_FILE="{metadata["accumulated_runtime_file"]}"
export HEARTBEAT_FILE="{metadata["heartbeat_file"]}"

# NCCL Configuration
export NCCL_PORT_RANGE="${{NCCL_PORT_RANGE:-43000-43063}}"
export NCCL_SOCKET_FAMILY="${{NCCL_SOCKET_FAMILY:-AF_INET}}"

# Debug
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${{TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}}"
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=""

# NCCL Mode
export NCCL_P2P_DISABLE="${{NCCL_P2P_DISABLE:-0}}"
export NCCL_SHM_DISABLE="${{NCCL_SHM_DISABLE:-0}}"
export NCCL_IB_DISABLE="${{NCCL_IB_DISABLE:-1}}"
'''

    # Append to file
    with open(metta_env_file, "a") as f:
        f.write(env_vars)

    print(f"Environment variables written to: {metta_env_file}")


def create_job_secrets(profile, wandb_password, observatory_token):
    """Create ~/.netrc and ~/.metta/observatory_tokens.yaml files."""
    # Run metta configure if profile is provided
    if profile:
        subprocess.run(["uv", "run", "metta", "configure", "--profile", profile])

    # Create ~/.netrc for wandb
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        print("~/.netrc already exists")
    else:
        netrc_content = f"machine api.wandb.ai\n  login user\n  password {wandb_password}\n"
        netrc_path.write_text(netrc_content)
        netrc_path.chmod(0o600)  # Restrict to owner read/write only
        print("~/.netrc created")

    # Create observatory tokens file if token is provided
    if observatory_token:
        observatory_path = Path.home() / ".metta" / "observatory_tokens.yaml"
        if observatory_path.exists():
            print("~/.metta/observatory_tokens.yaml already exists")
        else:
            observatory_path.parent.mkdir(exist_ok=True)
            observatory_content = (
                f"{PROD_STATS_SERVER_URI}: {observatory_token}\n"
                + f"{PROD_OBSERVATORY_FRONTEND_URL}/api: {observatory_token}\n"
            )
            observatory_path.write_text(observatory_content)
            print("~/.metta/observatory_tokens.yaml created")


def main():
    """Main function to configure the runtime environment."""
    node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
    is_master = node_index == 0

    # Print initial environment info
    if is_master:
        print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
        print(f"Which python: {run_command('which python')}")
        python_exec = run_command([sys.executable, "-c", "import sys; print(sys.executable)"])
        print(f"Python executable: {python_exec}")

    print(f"Configuring runtime environment [node {node_index}]...")

    # Create wandb directory
    Path("./wandb").mkdir(exist_ok=True)

    # Get METTA_ENV_FILE path and create parent directories
    metta_env_file = str(METTA_ENV_FILE)
    Path(metta_env_file).parent.mkdir(parents=True, exist_ok=True)
    print(f"Persisting env vars into: {metta_env_file} [node {node_index}]")

    # Setup job metadata
    metadata = setup_job_metadata()

    # Write environment variables
    write_environment_variables(metta_env_file, metadata)

    # Check for required WANDB_PASSWORD
    wandb_password = os.environ.get("WANDB_PASSWORD")
    if not wandb_password:
        print("ERROR: WANDB_PASSWORD environment variable is required but not set")
        print("Please ensure WANDB_PASSWORD is set in your Skypilot environment variables")
        sys.exit(1)

    # Get optional OBSERVATORY_TOKEN
    observatory_token = os.environ.get("OBSERVATORY_TOKEN")

    print(f"Creating/updating job secrets [node {node_index}]...")

    # Create job secrets
    try:
        create_job_secrets("softmax-docker", wandb_password, observatory_token)
    except Exception as e:
        print(f"ERROR: Failed to create job secrets: {e}")
        sys.exit(1)

    print("Runtime environment configuration completed [node {node_index}]")


if __name__ == "__main__":
    main()
