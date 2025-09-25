"""Skypilot dispatcher implementation for distributed job execution."""

import logging
import subprocess
import uuid

from metta.adaptive.dispatcher.local import LocalDispatcher
from metta.adaptive.models import JobDefinition, JobTypes
from metta.adaptive.protocols import Dispatcher
from metta.adaptive.utils import get_display_id
from metta.common.util.constants import SKYPILOT_LAUNCH_PATH

logger = logging.getLogger(__name__)


class SkypilotDispatcher(Dispatcher):
    """Dispatches jobs to cloud resources using Skypilot.

    Training jobs are dispatched to cloud resources via Skypilot.
    Evaluation jobs are delegated to LocalDispatcher to avoid unnecessary cloud costs.
    """

    def __init__(self):
        """Initialize the Skypilot dispatcher with a LocalDispatcher for eval jobs."""
        # Create a LocalDispatcher instance for handling eval jobs
        # Use capture_output=True to see evaluation output
        self._local_dispatcher = LocalDispatcher(capture_output=True)

    def dispatch(self, job: JobDefinition) -> str:
        """Dispatch job based on type.

        Training jobs go to Skypilot for cloud execution.
        Evaluation jobs are delegated to LocalDispatcher.
        """
        display_id = get_display_id(job.run_id)

        # Check if this is an evaluation job
        if job.type == JobTypes.LAUNCH_EVAL:
            # Delegate evaluations to LocalDispatcher
            logger.info(f"Delegating evaluation {display_id} to LocalDispatcher")
            return self._local_dispatcher.dispatch(job)
        else:
            # Dispatch training jobs to Skypilot
            logger.info(f"Dispatching training {display_id} to Skypilot")
            return self._dispatch_skypilot(job)

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
            # Launch the process and capture output for logging
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for the launch script to complete and capture output
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout for launch

            # Log the output
            if stdout:
                logger.info(f"Skypilot launch output for {display_id}:\n{stdout}")
            if stderr:
                logger.warning(f"Skypilot launch stderr for {display_id}:\n{stderr}")

            # Check if the launch was successful
            if process.returncode != 0:
                error_msg = f"Skypilot launch failed with return code {process.returncode}"
                if stderr:
                    error_msg += f": {stderr}"
                logger.error(f"Failed to launch {display_id} on Skypilot: {error_msg}")
                raise RuntimeError(error_msg)

            logger.info(f"Successfully launched {display_id} on Skypilot")

            # Return the UUID as dispatch_id
            return dispatch_id

        except subprocess.TimeoutExpired as e:
            logger.error(f"Skypilot launch timed out for {job.run_id}")
            process.kill()
            raise RuntimeError(f"Skypilot launch timed out after 5 minutes for {job.run_id}") from e
        except Exception as e:
            logger.error(f"Failed to launch Skypilot job {job.run_id}: {e}")
            raise

    def check_local_processes(self) -> int:
        """Check status of local evaluation processes.

        Returns:
            Number of active local processes
        """
        return self._local_dispatcher.check_processes()
