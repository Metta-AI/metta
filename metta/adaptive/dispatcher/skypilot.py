"""Skypilot dispatcher implementation for distributed job execution."""

import logging
import subprocess
import uuid

from metta.adaptive.models import JobDefinition, JobTypes
from metta.adaptive.protocols import Dispatcher
from metta.adaptive.utils import get_display_id
from metta.common.util.constants import SKYPILOT_LAUNCH_PATH

logger = logging.getLogger(__name__)


class SkypilotDispatcher(Dispatcher):
    """Dispatches jobs to cloud resources using Skypilot.

    Training jobs are dispatched to cloud resources via Skypilot.
    Evaluation jobs are run locally to avoid unnecessary cloud costs.
    """

    def __init__(self):
        """Initialize the Skypilot dispatcher."""
        pass  # No initialization needed

    def dispatch(self, job: JobDefinition) -> str:
        """Dispatch job based on type.

        Training jobs go to Skypilot for cloud execution.
        Evaluation jobs run locally to avoid cloud costs.
        """
        display_id = get_display_id(job.run_id)

        # Check if this is an evaluation job
        if job.type == JobTypes.LAUNCH_EVAL:
            # Run evaluations locally
            logger.info(f"Running evaluation {display_id} locally (not via Skypilot)")
            return self._dispatch_local(job)
        else:
            # Dispatch training jobs to Skypilot
            logger.info(f"Dispatching training {display_id} to Skypilot")
            return self._dispatch_skypilot(job)

    def _dispatch_local(self, job: JobDefinition) -> str:
        """Dispatch evaluation job locally."""
        # Build command for local execution
        cmd_parts = ["uv", "run", "./tools/run.py", job.cmd]

        # Add all arguments
        for k, v in job.args.items():
            cmd_parts.append(f"{k}={v}")

        for k, v in job.overrides.items():
            cmd_parts.append(f"{k}={v}")

        display_id = get_display_id(job.run_id)
        logger.info(f"Local eval command: {' '.join(cmd_parts)}")

        try:
            # Launch the process locally with no output capture
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )

            # Use PID as dispatch ID for local processes
            dispatch_id = str(process.pid)
            logger.info(f"Started local eval {display_id} with PID {dispatch_id}")
            return dispatch_id

        except Exception as e:
            logger.error(f"Failed to launch local eval {job.run_id}: {e}")
            raise

    def _dispatch_skypilot(self, job: JobDefinition) -> str:
        """Dispatch training job via Skypilot."""
        # Build command parts starting with the launcher script
        cmd_parts = [SKYPILOT_LAUNCH_PATH]

        # Add Skypilot flags (in order)
        # 1. Always add --no-spot
        cmd_parts.append("--no-spot")

        # 2. Add --gpus if specified and > 0
        if job.gpus and job.gpus > 0:
            cmd_parts.append(f"--gpus={job.gpus}")

        # 3. Add --nodes if specified and > 1
        if job.nodes and job.nodes > 1:
            cmd_parts.append(f"--nodes={job.nodes}")

        # 3.1 Always add long timeout
        cmd_parts.append("--heartbeat-timeout=10000")

        # 4. Add the actual command (e.g., experiments.recipes.arena.train)
        cmd_parts.append(job.cmd)

        # Add all arguments directly (no --args or --overrides flags)
        # First add job args, then overrides
        for k, v in job.args.items():
            cmd_parts.append(f"{k}={v}")

        for k, v in job.overrides.items():
            cmd_parts.append(f"{k}={v}")

        display_id = get_display_id(job.run_id)
        logger.info(f"Skypilot command: {' '.join(cmd_parts)}")

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

            logger.info(f"Launched {display_id} on Skypilot with PID {process.pid}")

            # Return the UUID as dispatch_id
            return dispatch_id

        except Exception as e:
            logger.error(f"Failed to launch Skypilot job {job.run_id}: {e}")
            raise
