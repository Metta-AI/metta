"""Local dispatcher implementation for running jobs as subprocesses."""

import logging
import subprocess
import threading

from metta.sweep.models import JobDefinition, JobTypes

logger = logging.getLogger(__name__)


class LocalDispatcher:
    """Runs jobs as local subprocesses."""

    def __init__(self, capture_output: bool = True):
        self._processes: dict[str, subprocess.Popen] = {}  # pid -> process
        self._run_to_pid: dict[str, str] = {}  # run_id -> pid for debugging
        self._capture_output = capture_output
        self._output_threads: dict[str, threading.Thread] = {}  # pid -> output thread

    def _reap_finished_processes(self):
        """Clean up finished subprocesses."""
        finished_pids = []
        for pid, process in self._processes.items():
            # poll() returns None if process is still running, returncode otherwise
            if process.poll() is not None:
                finished_pids.append(pid)
                logger.debug(f"Process {pid} finished with return code {process.returncode}")

        # Clean up finished processes
        for pid in finished_pids:
            del self._processes[pid]
            # Clean up run_id mapping
            run_id = next((rid for rid, p in self._run_to_pid.items() if p == pid), None)
            if run_id:
                del self._run_to_pid[run_id]
            # Clean up output thread if exists
            if pid in self._output_threads:
                thread = self._output_threads[pid]
                if thread.is_alive():
                    thread.join(timeout=1.0)  # Wait briefly for thread to finish
                del self._output_threads[pid]

    def check_processes(self):
        """Check status of all processes."""
        self._reap_finished_processes()
        active_count = len(self._processes)
        if active_count > 0:
            logger.debug(f"Active subprocesses: {active_count}")
        return active_count

    def _stream_output(self, process: subprocess.Popen, run_id: str, pid: str):
        """Stream output from subprocess to logger."""
        try:
            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    # Check if process is still running
                    if process.poll() is not None:
                        break
                    continue

                line = line.strip()

                # Log to logger with appropriate prefix
                # Extract trial portion for cleaner display
                display_id = run_id.split("_trial_")[-1] if "_trial_" in run_id else run_id
                display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id
                logger.info(f"[{display_id}] {line}")

        except Exception as e:
            logger.error(f"Error streaming output for PID {pid}: {e}")

    def dispatch(self, job: JobDefinition) -> str:
        """Dispatch job locally as subprocess."""
        # Reap any finished processes first to prevent zombie accumulation
        self._reap_finished_processes()

        # Build command
        cmd_parts = ["uv", "run", "./tools/run.py", job.cmd]

        # Add positional arguments first (if any)
        cmd_parts.extend(job.args)

        # Collect all args
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

        # Extract trial portion for cleaner display
        display_id = job.run_id.split("_trial_")[-1] if "_trial_" in job.run_id else job.run_id
        display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id

        logger.info(f"Dispatching {display_id}: {' '.join(cmd_parts)}")

        try:
            # Configure subprocess output handling
            if self._capture_output:
                # Capture output for streaming and logging
                process = subprocess.Popen(
                    cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Combine stderr with stdout
                    text=True,
                    bufsize=1,  # Line buffered
                )
            else:
                # Production mode - discard output to avoid potential deadlock
                process = subprocess.Popen(
                    cmd_parts,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )

            # Use PID as the dispatch_id
            pid = str(process.pid)
            self._processes[pid] = process
            self._run_to_pid[job.run_id] = pid

            # Start output streaming thread if capturing output
            if self._capture_output:
                output_thread = threading.Thread(
                    target=self._stream_output,
                    args=(process, job.run_id, pid),
                    daemon=True,  # Daemon thread will be killed when main process exits
                )
                output_thread.start()
                self._output_threads[pid] = output_thread

            logger.info(f"Started {display_id} with PID {pid}")
            return pid

        except Exception as e:
            logger.error(f"Failed to start local run {job.run_id}: {e}")
            raise
