"""Skypilot dispatcher implementation for distributed job execution."""

import logging
import subprocess
import uuid

from softmax.training.adaptive.dispatcher.local import LocalDispatcher
from softmax.training.adaptive.models import JobDefinition, JobTypes
from softmax.training.adaptive.protocols import Dispatcher
from softmax.training.adaptive.utils import get_display_id
from metta.common.util.constants import SKYPILOT_LAUNCH_PATH

logger = logging.getLogger(__name__)


class SkypilotDispatcher(Dispatcher):
    """Dispatch training via Skypilot while keeping evals local."""

    def __init__(self) -> None:
        self._local_dispatcher = LocalDispatcher(capture_output=True)

    def dispatch(self, job: JobDefinition) -> str:
        display_id = get_display_id(job.run_id)

        if job.type == JobTypes.LAUNCH_EVAL:
            logger.info("Delegating evaluation %s to LocalDispatcher", display_id)
            return self._local_dispatcher.dispatch(job)

        logger.info("Dispatching training %s to Skypilot", display_id)
        return self._dispatch_skypilot(job, display_id)

    def _dispatch_skypilot(self, job: JobDefinition, display_id: str) -> str:
        cmd_parts = [
            SKYPILOT_LAUNCH_PATH,
            "--no-spot",
            "--heartbeat-timeout=10000",
        ]

        if job.gpus and job.gpus > 0:
            cmd_parts.append(f"--gpus={job.gpus}")

        if job.nodes and job.nodes > 1:
            cmd_parts.append(f"--nodes={job.nodes}")

        cmd_parts.append(job.cmd)
        cmd_parts.extend(f"{k}={v}" for k, v in job.args.items())
        cmd_parts.extend(f"{k}={v}" for k, v in job.overrides.items())

        logger.info("Skypilot command: %s", " ".join(cmd_parts))

        dispatch_id = str(uuid.uuid4())

        try:
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            stdout, stderr = process.communicate(timeout=300)

            for label, stream, log in (
                ("output", stdout, logger.info),
                ("stderr", stderr, logger.warning),
            ):
                if stream:
                    log("Skypilot launch %s for %s:\n%s", label, display_id, stream)

            if process.returncode != 0:
                error_msg = f"Skypilot launch failed with return code {process.returncode}"
                if stderr:
                    error_msg = f"{error_msg}: {stderr}"
                logger.error("Failed to launch %s on Skypilot: %s", display_id, error_msg)
                raise RuntimeError(error_msg)

            logger.info("Successfully launched %s on Skypilot", display_id)
            return dispatch_id

        except subprocess.TimeoutExpired as exc:
            logger.error("Skypilot launch timed out for %s", job.run_id)
            process.kill()
            raise RuntimeError(f"Skypilot launch timed out after 5 minutes for {job.run_id}") from exc
        except Exception as exc:  # pragma: no cover - surface the underlying error
            logger.error("Failed to launch Skypilot job %s: %s", job.run_id, exc)
            raise

    def check_local_processes(self) -> int:
        """Return number of active local evaluation processes."""
        return self._local_dispatcher.check_processes()
