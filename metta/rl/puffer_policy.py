"""Backward-compatible shim for PufferLib policy helpers."""

from __future__ import annotations

from metta.agent.puffer_policy import *  # noqa: F401,F403
from metta.agent.puffer_policy import (  # noqa: F401
    _create_metta_agent,
    _is_puffer_state_dict,
    load_pufferlib_checkpoint,
)

__all__ = [
    "_create_metta_agent",
    "_is_puffer_state_dict",
    "load_pufferlib_checkpoint",
]
