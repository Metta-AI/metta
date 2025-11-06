"""Scripted agent policies for CoGames.

Contains two baseline policies:
- BaselinePolicy: Single/multi-agent resource gathering and heart assembly
- UnclippingPolicy: Extends baseline with extractor unclipping capability

And their hyperparameter configurations:
- BaselineHyperparameters: Configuration for baseline agent behavior
- UnclippingHyperparameters: Configuration for unclipping agent behavior
"""

import cogames.policy.scripted_agent.baseline_agent as baseline_agent
import cogames.policy.scripted_agent.unclipping_agent as unclipping_agent

BaselinePolicy = baseline_agent.BaselinePolicy
BaselineHyperparameters = baseline_agent.BaselineHyperparameters
BASELINE_HYPERPARAMETER_PRESETS = baseline_agent.BASELINE_HYPERPARAMETER_PRESETS
UnclippingPolicy = unclipping_agent.UnclippingPolicy
UnclippingHyperparameters = unclipping_agent.UnclippingHyperparameters

__all__ = [
    "BaselinePolicy",
    "BaselineHyperparameters",
    "BASELINE_HYPERPARAMETER_PRESETS",
    "UnclippingPolicy",
    "UnclippingHyperparameters",
]
