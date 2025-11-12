"""Fast Nim-based policy wrappers."""

from .agents import (
    RaceCarAgentPolicy,
    RaceCarAgentsMultiPolicy,
    RandomAgentPolicy,
    RandomAgentsMultiPolicy,
    ThinkyAgentPolicy,
    ThinkyAgentsMultiPolicy,
)

__all__ = [
    "RandomAgentPolicy",
    "RandomAgentsMultiPolicy",
    "ThinkyAgentPolicy",
    "ThinkyAgentsMultiPolicy",
    "RaceCarAgentPolicy",
    "RaceCarAgentsMultiPolicy",
]
