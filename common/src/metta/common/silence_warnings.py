"""Suppress noisy third-party warnings for developer tooling."""

from __future__ import annotations

import logging
import warnings


def silence_gym_warnings() -> None:
    """Mute legacy Gym deprecation chatter in logs and stderr."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger("gym.error").setLevel(logging.ERROR)


silence_gym_warnings()

__all__ = ["silence_gym_warnings"]
