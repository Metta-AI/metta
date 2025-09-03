"""Skypilot dispatcher implementation for distributed job execution."""

import logging
import os
import subprocess
import uuid

from metta.sweep.models import JobDefinition, JobTypes
from metta.sweep.protocols import Dispatcher

logger = logging.getLogger(__name__)


class SkypilotDispatcher(Dispatcher):
    """Dispatches jobs to cloud resources using Skypilot."""

    def __init__(self):
        """Initialize the Skypilot dispatcher."""
        pass  # No initialization needed

    def dispatch(self, job: JobDefinition) -> str:
        """Dispatch job using Skypilot launcher"""

        # Build command parts starting with the launcher script
        cmd_parts = [
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "devops", "skypilot", "launch.py")
            )
        ]

        # Add Skypilot flags (in order)
        # 1. Always add --no-spot
        cmd_parts.append("--no-spot")

        # 2. Add --gpus if specified and > 0
        if job.gpus and job.gpus > 0:
            cmd_parts.append(f"--gpus={job.gpus}")

        # 3. Add --nodes if specified and > 1
        if job.nodes and job.nodes > 1:
            cmd_parts.append(f"--nodes={job.nodes}")

        # 4. Add the actual command (e.g., experiments.recipes.arena.train)
        cmd_parts.append(job.cmd)

        # Add positional arguments first (if any)
        cmd_parts.extend(job.args)

        # Collect all args (exactly like LocalDispatcher)
        all_args = []

        # Add run_id for training jobs only (not for eval)
        if job.type == JobTypes.LAUNCH_TRAINING:
            all_args.append(f"run={job.run_id}")

        # Add metadata fields as args (used for evaluation jobs)
        for key, value in job.metadata.items():
            all_args.append(f"{key}={value}")

        # Add all args with --args flag
        if all_args:
            cmd_parts.append("--args")
            cmd_parts.extend(all_args)

        # Collect all overrides (from both overrides and config)
        all_overrides = []

        # Add explicit overrides
        for key, value in job.overrides.items():
            all_overrides.append(f"{key}={value}")

        # Add config from optimizer as additional overrides
        for key, value in job.config.items():
            all_overrides.append(f"{key}={value}")

        # Add all overrides with --overrides flag
        if all_overrides:
            cmd_parts.append("--overrides")
            cmd_parts.extend(all_overrides)

        # Extract trial portion for cleaner display (like LocalDispatcher)
        display_id = job.run_id.split("_trial_")[-1] if "_trial_" in job.run_id else job.run_id
        display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id

        logger.info(f"Dispatching {display_id}: {' '.join(cmd_parts)}")

        # Generate a UUID for this dispatch
        dispatch_id = str(uuid.uuid4())

        try:
            # Launch the process with fire-and-forget approach
            # Use DEVNULL for stdout/stderr (no output capture)
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )

            logger.info(f"Launched {display_id} with PID {process.pid}")

            # Return the UUID as dispatch_id
            return dispatch_id

        except Exception as e:
            logger.error(f"Failed to launch job {job.run_id}: {e}")
            raise
