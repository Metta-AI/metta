from typing import Any, Optional, Tuple

import numpy as np

from mettagrid.util.stopwatch import Stopwatch


class TimedVecenv:
    """Wraps a PufferLib vectorized environment to track timing."""

    def __init__(self, vecenv: Any, timer: Stopwatch):
        self._vecenv = vecenv
        self._timer = timer

    def __getattr__(self, name: str) -> Any:
        """Delegate all non-overridden attributes to the wrapped vecenv.

        This allows the wrapper to be a drop-in replacement for the original
        vecenv, only intercepting the methods we explicitly override.
        """
        return getattr(self._vecenv, name)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        with self._timer("vecenv.reset"):
            return self._vecenv.reset(seed)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        with self._timer("vecenv.step"):
            return self._vecenv.step(actions)

    # def async_reset(self, seed: Optional[int] = None) -> None:
    #     with self._timer("vecenv.async_reset"):
    #         return self._vecenv.async_reset(seed)

    # def send(self, actions: np.ndarray) -> None:
    #     with self._timer("vecenv.send"):
    #         return self._vecenv.send(actions)

    def recv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
        with self._timer("vecenv.recv"):
            return self._vecenv.recv()

    def close(self) -> None:
        with self._timer("vecenv.close"):
            return self._vecenv.close()

    def notify(self) -> None:
        with self._timer("vecenv.notify"):
            return self._vecenv.notify()
