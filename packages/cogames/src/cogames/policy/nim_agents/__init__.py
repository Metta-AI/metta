"""Nim-based agent policies for CoGames."""

from cogames.policy.nim_agents import agents  # noqa: F401

__all__ = [
    "RandomAgentsMultiPolicy",
    "ThinkyAgentsMultiPolicy",
    "RaceCarAgentsMultiPolicy",
    "LadyBugAgentsMultiPolicy",
]

# Re-export the policy classes for convenience
from cogames.policy.nim_agents.agents import (  # noqa: F401
    LadyBugAgentsMultiPolicy,
    RaceCarAgentsMultiPolicy,
    RandomAgentsMultiPolicy,
    ThinkyAgentsMultiPolicy,
)
