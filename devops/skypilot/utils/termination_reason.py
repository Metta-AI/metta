"""Termination reason types for SkyPilot jobs."""

from enum import Enum


class TerminationReason(str, Enum):
    """Enum for job termination reasons."""

    # Normal completion
    JOB_COMPLETED = "job_completed"

    # Timeouts and limits
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    MAX_RUNTIME_REACHED = "max_runtime_reached"

    # Heartbeat errors
    HEARTBEAT_FILE_MISSING = "heartbeat_file_missing"
    HEARTBEAT_DIRECTORY_MISSING = "heartbeat_directory_missing"
    HEARTBEAT_PERMISSION_DENIED = "heartbeat_permission_denied"

    # Testing
    FORCE_RESTART_TEST = "force_restart_test"
    NCCL_TESTS_FAILED = "nccl_tests_failed"
    RAPID_RESTARTS = "rapid_restarts"

    # Failures
    JOB_FAILED = "job_failed"  # Generic failure, can be suffixed with exit code

    def with_exit_code(self, exit_code: int) -> str:
        """Create a failure reason with an exit code."""
        return f"{self.value}_{exit_code}"

    @staticmethod
    def parse_os_error(errno: int | str) -> str:
        """Create a heartbeat OS error reason."""
        return f"heartbeat_os_error_{errno}"

    @staticmethod
    def parse_unexpected_error(error_type: str) -> str:
        """Create an unexpected error reason."""
        return f"heartbeat_unexpected_error_{error_type}"
