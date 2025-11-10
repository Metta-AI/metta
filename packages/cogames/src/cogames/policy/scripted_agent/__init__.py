"""Scripted agent policies for CoGames.

Contains two baseline policies:
- BaselinePolicy: Single/multi-agent resource gathering and heart assembly
- UnclippingPolicy: Extends baseline with extractor unclipping capability
"""

from cogames.policy.scripted_agent.baseline_agent import BaselinePolicy, NoopBaselinePolicy
from cogames.policy.scripted_agent.unclipping_agent import UnclippingPolicy

__all__ = [
    "BaselinePolicy",
    "NoopBaselinePolicy",
    "UnclippingPolicy",
]
