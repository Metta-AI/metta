from typing import Any, Optional, Tuple

import numpy as np

from mettagrid.util.stopwatch import Stopwatch


class TimedVecenv:
    """Wraps a PufferLib vectorized environment to track timing."""

    def __init__(self, vecenv: Any, timer: Stopwatch):
        self._vecenv = vecenv
        self._timer = timer
        self._in_user_code = False
        self._overhead_timer_running = False

        # Start timing overhead immediately - we're in "framework" code
        # until we enter one of our wrapped methods
        self._start_overhead_timer()

    def __getattr__(self, name: str) -> Any:
        """Delegate all non-overridden attributes to the wrapped vecenv.

        This allows the wrapper to be a drop-in replacement for the original
        vecenv, only intercepting the methods we explicitly override.
        """
        return getattr(self._vecenv, name)

    # High-level methods (these call async_reset + recv internally)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment with timing.

        This high-level method internally calls async_reset + recv.
        """
        self._enter_user_code()
        try:
            with self._timer("vecenv.reset"):
                return self._vecenv.reset(seed)
        finally:
            self._exit_user_code()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step the environment with timing.

        This high-level method internally calls send + recv.
        """
        self._enter_user_code()
        try:
            with self._timer("vecenv.step"):
                return self._vecenv.step(actions)
        finally:
            self._exit_user_code()

    # Low-level async methods

    def async_reset(self, seed: Optional[int] = None) -> None:
        """Asynchronously reset the environment with timing."""
        self._enter_user_code()
        try:
            with self._timer("vecenv.async_reset"):
                return self._vecenv.async_reset(seed)
        finally:
            self._exit_user_code()

    def send(self, actions: np.ndarray) -> None:
        """Send actions to the environment with timing."""
        self._enter_user_code()
        try:
            with self._timer("vecenv.send"):
                return self._vecenv.send(actions)
        finally:
            self._exit_user_code()

    def recv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
        """Receive results from the environment with timing.

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos, env_ids, masks)
        """
        self._enter_user_code()
        try:
            with self._timer("vecenv.recv"):
                return self._vecenv.recv()
        finally:
            self._exit_user_code()

    def close(self) -> None:
        """Close the environment and stop timing."""
        self._enter_user_code()
        try:
            with self._timer("vecenv.close"):
                return self._vecenv.close()
        finally:
            # Don't call _exit_user_code() since we're shutting down
            # Just reset the state flags
            self._in_user_code = False
            # Overhead timer is already stopped by _enter_user_code()
            self._overhead_timer_running = False

    def notify(self) -> None:
        """Notify the environment (used for async coordination)."""
        self._enter_user_code()
        try:
            with self._timer("vecenv.notify"):
                return self._vecenv.notify()
        finally:
            self._exit_user_code()

    def _start_overhead_timer(self) -> None:
        """Start the overhead timer if not already running."""
        assert not self._overhead_timer_running
        self._timer.start("pufferlib_overhead")
        self._overhead_timer_running = True

    def _stop_overhead_timer(self) -> None:
        """Stop the overhead timer if running."""
        assert self._overhead_timer_running
        self._timer.stop("pufferlib_overhead")
        self._overhead_timer_running = False

    def _enter_user_code(self) -> None:
        """Stop overhead timer when entering vecenv methods.

        This is called at the start of each wrapped method to pause
        the overhead timer while we're executing vecenv code.
        """
        assert not self._in_user_code
        self._stop_overhead_timer()
        self._in_user_code = True

    def _exit_user_code(self) -> None:
        """Restart overhead timer when exiting vecenv methods.

        This is called at the end of each wrapped method to resume
        the overhead timer when we return to framework code.
        """
        assert self._in_user_code
        self._in_user_code = False
        self._start_overhead_timer()
