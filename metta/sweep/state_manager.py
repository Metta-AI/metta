"""Robust state management for sweep runs.

This module provides comprehensive state tracking for sweep runs with:
- Clear lifecycle states (initializing, running, evaluating, success, failure, etc.)
- Heartbeat mechanism to detect stuck/crashed runs
- Timeout detection
- Error tracking and recovery
"""

import enum
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SweepRunState(enum.Enum):
    """Comprehensive sweep run states."""

    INITIALIZING = "initializing"  # Setting up, loading previous runs
    RUNNING = "running"  # Training in progress
    EVALUATING = "evaluating"  # Evaluation in progress
    SUCCESS = "success"  # Completed successfully
    FAILURE = "failure"  # Failed with error
    TIMEOUT = "timeout"  # Timed out (no heartbeat)
    CRASHED = "crashed"  # Unexpected termination
    CANCELLED = "cancelled"  # User cancelled


class RobustSweepStateManager:
    """
    Manages sweep state with proper lifecycle tracking and recovery.

    This class provides:
    - Consistent state updates with timestamps
    - Heartbeat mechanism to indicate liveness
    - Timeout detection for stuck runs
    - Error tracking for debugging
    - Extra fields for custom metrics

    Example usage:
        state_manager = RobustSweepStateManager(wandb_run, timeout_minutes=30)
        state_manager.set_state(SweepRunState.INITIALIZING)

        # During training loop
        state_manager.heartbeat()

        # On completion
        state_manager.set_state(SweepRunState.SUCCESS,
                               **{"score": 0.95, "train_time": 120.0})
    """

    def __init__(self, wandb_run, timeout_minutes: int = 30):
        """
        Initialize the state manager.

        Args:
            wandb_run: The WandB run object to update
            timeout_minutes: Timeout in minutes before considering run stuck
        """
        self.wandb_run = wandb_run
        self.timeout_seconds = timeout_minutes * 60
        self.start_time = time.time()
        self._last_heartbeat = time.time()

    def set_state(self, state: SweepRunState, error: Optional[str] = None, **extra_fields):
        """
        Update sweep state with timestamp and optional error.

        Args:
            state: The new state to set
            error: Optional error message (typically for FAILURE state)
            **extra_fields: Additional fields to include in the update
        """
        update_dict = {
            "protein.state": state.value,
            "protein.last_update": datetime.now(timezone.utc).isoformat(),
            "protein.runtime_seconds": time.time() - self.start_time,
        }

        if error:
            update_dict["protein.error"] = error

        # Add any extra fields with protein prefix
        for key, value in extra_fields.items():
            if not key.startswith("protein."):
                key = f"protein.{key}"
            update_dict[key] = value

        try:
            self.wandb_run.summary.update(update_dict)
            logger.info(f"Sweep state updated to: {state.value}")
        except Exception as e:
            logger.error(f"Failed to update sweep state: {e}")

    def heartbeat(self):
        """
        Send heartbeat to indicate the run is still alive.

        This should be called periodically during long-running operations
        to prevent the run from being marked as timed out.
        """
        self._last_heartbeat = time.time()

        try:
            self.wandb_run.summary.update(
                {
                    "protein.heartbeat": datetime.now(timezone.utc).isoformat(),
                    "protein.runtime_seconds": time.time() - self.start_time,
                }
            )
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    def check_timeout(self) -> bool:
        """
        Check if run has exceeded timeout.

        Returns:
            True if the run has exceeded the timeout, False otherwise
        """
        elapsed = time.time() - self._last_heartbeat
        return elapsed > self.timeout_seconds

    def get_runtime_seconds(self) -> float:
        """
        Get the total runtime in seconds.

        Returns:
            Runtime in seconds since initialization
        """
        return time.time() - self.start_time

    def transition_to_terminal_state(self, success: bool = True, error: Optional[str] = None, **extra_fields):
        """
        Transition to a terminal state (SUCCESS or FAILURE).

        Args:
            success: If True, transition to SUCCESS; if False, to FAILURE
            error: Error message for failure cases
            **extra_fields: Additional fields to include
        """
        if success:
            self.set_state(SweepRunState.SUCCESS, **extra_fields)
        else:
            self.set_state(SweepRunState.FAILURE, error=error, **extra_fields)

    def handle_exception(self, exception: Exception, state: Optional[SweepRunState] = None):
        """
        Handle an exception by setting appropriate failure state.

        Args:
            exception: The exception that occurred
            state: Optional specific state to set (defaults to FAILURE)
        """
        error_msg = f"{type(exception).__name__}: {str(exception)}"

        if state is None:
            # Determine appropriate state based on exception type
            if isinstance(exception, TimeoutError):
                state = SweepRunState.TIMEOUT
            elif isinstance(exception, KeyboardInterrupt):
                state = SweepRunState.CANCELLED
            else:
                state = SweepRunState.FAILURE

        self.set_state(state, error=error_msg)

    def __enter__(self):
        """Context manager entry - set RUNNING state."""
        self.set_state(SweepRunState.RUNNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - handle exceptions or set SUCCESS."""
        if exc_type is not None:
            self.handle_exception(exc_val)
        else:
            # Only set SUCCESS if not already in a terminal state
            current_state = self.wandb_run.summary.get("protein.state")
            if current_state not in ["success", "failure", "timeout", "crashed", "cancelled"]:
                self.set_state(SweepRunState.SUCCESS)
        return False  # Don't suppress exceptions
