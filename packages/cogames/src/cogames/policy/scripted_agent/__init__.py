"""Scripted agent policies for CoGames.

Contains three baseline policies:
- SimpleBaselinePolicy: Single-agent resource gathering and heart assembly
- UnclippingPolicy: Extends baseline with extractor unclipping capability
- CoordinatingPolicy: Multi-agent coordination around stations
"""

from cogames.policy.scripted_agent.coordinating_agent import CoordinatingPolicy
from cogames.policy.scripted_agent.simple_baseline_agent import SimpleBaselinePolicy
from cogames.policy.scripted_agent.unclipping_agent import UnclippingPolicy

__all__ = [
    "SimpleBaselinePolicy",
    "UnclippingPolicy",
    "CoordinatingPolicy",
]
