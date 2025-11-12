"""Fast Nim-based policy wrappers."""

from .agents import (
    LadybugAgentPolicy,
    LadybugAgentsMultiPolicy,
    RaceCarAgentPolicy,
    RaceCarAgentsMultiPolicy,
    RandomAgentPolicy,
    RandomAgentsMultiPolicy,
    ThinkyAgentPolicy,
    ThinkyAgentsMultiPolicy,
)

# Backwards compatibility for older scripted_baseline references
ScriptedBaselineAgentPolicy = LadybugAgentPolicy
ScriptedBaselineMultiPolicy = LadybugAgentsMultiPolicy

__all__ = [
    "RandomAgentPolicy",
    "RandomAgentsMultiPolicy",
    "ThinkyAgentPolicy",
    "ThinkyAgentsMultiPolicy",
    "RaceCarAgentPolicy",
    "RaceCarAgentsMultiPolicy",
    "LadybugAgentPolicy",
    "LadybugAgentsMultiPolicy",
    "ScriptedBaselineAgentPolicy",
    "ScriptedBaselineMultiPolicy",
]
