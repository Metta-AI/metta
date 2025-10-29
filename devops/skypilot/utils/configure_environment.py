#!/usr/bin/env -S uv run python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

from metta.common.util.constants import METTA_ENV_FILE, PROD_OBSERVATORY_FRONTEND_URL, PROD_STATS_SERVER_URI
from metta.common.util.log_config import getRankAwareLogger, init_logging

logger = getRankAwareLogger(__name__)


def _setup_job_metadata():
    """Setup job metadata tracking (restart count, accumulated runtime).

    Note: We track restart count separately from SkyPilot's job recovery config because:
    1. We need the count for runtime monitoring and accumulated runtime tracking
    2. We persist it to shared storage (DATA_DIR) so it survives across job restarts
    3. It's used by runtime monitors to make decisions about timeouts and testing
    4. SkyPilot's max_restarts_on_errors controls the restart policy, while this tracks actual restarts
    """
    data_dir = os.environ.get("DATA_DIR", "./train_dir")
    metta_run_id = os.environ.get("METTA_RUN_ID", "default")

    shared_metadata_dir = Path(data_dir) / ".job_metadata" / metta_run_id
    shared_metadata_dir.mkdir(parents=True, exist_ok=True)
    restart_count_file = shared_metadata_dir / "restart_count"
    accumulated_runtime_file = shared_metadata_dir / "accumulated_runtime"

    local_metadata_dir = Path("/tmp") / ".job_metadata" / metta_run_id
    local_metadata_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_file = Path(os.environ.get("HEARTBEAT_FILE", local_metadata_dir / "heartbeat_file"))

    # Calculate IS_MASTER based on node rank
    node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
    is_master = node_index == 0

    # Handle restart count
    if restart_count_file.exists():
        restart_count = int(restart_count_file.read_text())
        logger.info_master(f"read restart_count = {restart_count}")
        restart_count += 1
    else:
        logger.info_master("restart count file not found, setting restart_count to 0")
        restart_count = 0

    # Only update restart count on master node
    if is_master:
        restart_count_file.write_text(str(restart_count))
        logger.info_master(f"Updated restart count file to {restart_count}")
    else:
        logger.info("Skipping RESTART_COUNT_FILE updates on non-master node")

    # Read accumulated runtime
    if accumulated_runtime_file.exists():
        accumulated_runtime_sec = int(accumulated_runtime_file.read_text())
    else:
        accumulated_runtime_sec = 0

    # Log restart info only on master
    logger.info_master("=" * 40 + " RESTART INFO " + "=" * 40)
    logger.info_master(f"  METTA_RUN_ID: {metta_run_id}")
    logger.info_master(f"  RESTART_COUNT: {restart_count}")
    logger.info_master(f"  ACCUMULATED_RUNTIME_SEC: {accumulated_runtime_sec}s ({accumulated_runtime_sec // 60}m)")
    logger.info_master("=" * 94)

    return {
        "restart_count": restart_count,
        "accumulated_runtime_sec": accumulated_runtime_sec,
        "accumulated_runtime_file": str(accumulated_runtime_file),
        "heartbeat_file": str(heartbeat_file),
    }


def _write_environment_variables(metta_env_file, metadata=None):
    """Write environment variables to METTA_ENV_FILE from template."""
    # Read the base template
    template_path = Path(__file__).parent.parent / "config" / "job_env.template.sh"
    env_vars = template_path.read_text()

    # Add job metadata exports only if metadata is provided (non-sandbox mode)
    if metadata:
        env_vars += f'''
# Job metadata exports
export RESTART_COUNT="{metadata["restart_count"]}"
export ACCUMULATED_RUNTIME_SEC="{metadata["accumulated_runtime_sec"]}"

# File path exports for monitors
export ACCUMULATED_RUNTIME_FILE="{metadata["accumulated_runtime_file"]}"
export HEARTBEAT_FILE="{metadata["heartbeat_file"]}"
'''

    # Append to file
    with open(metta_env_file, "a") as f:
        f.write(env_vars)

    logger.info(f"Environment variables written to: {metta_env_file}")


def _create_job_secrets(profile, wandb_password, observatory_token):
    """Create ~/.netrc and ~/.metta/observatory_tokens.yaml files."""
    # Run metta configure if profile is provided
    if profile:
        logger.info(f"Running metta configure with profile: {profile}")
        subprocess.run(["uv", "run", "metta", "configure", "--profile", profile])

    # Create ~/.netrc for wandb
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        logger.info("~/.netrc already exists")
    else:
        netrc_content = f"machine api.wandb.ai\n  login user\n  password {wandb_password}\n"
        netrc_path.write_text(netrc_content)
        netrc_path.chmod(0o600)  # Restrict to owner read/write only
        logger.info("~/.netrc created")

    # Create observatory tokens file if token is provided
    if observatory_token:
        observatory_path = Path.home() / ".metta" / "observatory_tokens.yaml"
        if observatory_path.exists():
            logger.info("~/.metta/observatory_tokens.yaml already exists")
        else:
            observatory_path.parent.mkdir(exist_ok=True)
            observatory_content = (
                f"{PROD_STATS_SERVER_URI}: {observatory_token}\n"
                + f"{PROD_OBSERVATORY_FRONTEND_URL}/api: {observatory_token}\n"
            )
            observatory_path.write_text(observatory_content)
            logger.info("~/.metta/observatory_tokens.yaml created")


def main():
    parser = argparse.ArgumentParser(description="Configure runtime environment for Skypilot jobs")
    parser.add_argument(
        "--sandbox", action="store_true", help="Configure for sandbox environment (skip job metadata setup)"
    )
    args = parser.parse_args()

    # Log mode
    if args.sandbox:
        logger.info("Running in SANDBOX mode - skipping job metadata setup")
    else:
        logger.info("Running in MANAGED JOB mode")

    # Print initial environment info only on master
    logger.info_master(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")

    # Get python path
    which_result = subprocess.run(["which", "python"], capture_output=True, text=True)
    if which_result.returncode == 0:
        logger.info_master(f"Which python: {which_result.stdout.strip()}")

    python_result = subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.executable)"], capture_output=True, text=True
    )
    if python_result.returncode == 0:
        logger.info_master(f"Python executable: {python_result.stdout.strip()}")

    logger.info("Configuring runtime environment...")

    # Create wandb directory
    Path("./wandb").mkdir(exist_ok=True)

    # Get METTA_ENV_FILE path and create parent directories
    metta_env_file = str(METTA_ENV_FILE)
    Path(metta_env_file).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Persisting env vars into: {metta_env_file}")

    # Setup job metadata only in non-sandbox mode
    metadata = None
    if not args.sandbox:
        metadata = _setup_job_metadata()

    # Write environment variables
    _write_environment_variables(metta_env_file, metadata)

    # Check for required WANDB_PASSWORD
    wandb_password = os.environ.get("WANDB_PASSWORD")
    if not wandb_password:
        logger.error("WANDB_PASSWORD environment variable is required but not set")
        logger.error("Please ensure WANDB_PASSWORD is set in your Skypilot environment variables")
        sys.exit(1)

    # Get optional OBSERVATORY_TOKEN
    observatory_token = os.environ.get("OBSERVATORY_TOKEN")

    logger.info("Creating/updating job secrets...")

    # Create job secrets
    _create_job_secrets("softmax-docker", wandb_password, observatory_token)

    logger.info("Runtime environment configuration completed")


if __name__ == "__main__":
    init_logging()
    main()
