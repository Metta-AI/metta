"""Scripted agent policy implementation with visual discovery and phase-based control."""

from cogames.policy.scripted_agent.agent import ScriptedAgentPolicy, ScriptedAgentPolicyImpl
from cogames.policy.scripted_agent.constants import C, FeatureNames, StationMaps
from cogames.policy.scripted_agent.hyperparameter_presets import HYPERPARAMETER_PRESETS
from cogames.policy.scripted_agent.hyperparameters import Hyperparameters
from cogames.policy.scripted_agent.navigator import Navigator
from cogames.policy.scripted_agent.phase_controller import GamePhase, create_controller
from cogames.policy.scripted_agent.state import AgentState

__all__ = [
    "ScriptedAgentPolicy",
    "ScriptedAgentPolicyImpl",
    "Hyperparameters",
    "HYPERPARAMETER_PRESETS",
    "Navigator",
    "GamePhase",
    "create_controller",
    "AgentState",
    "C",
    "FeatureNames",
    "StationMaps",
]
