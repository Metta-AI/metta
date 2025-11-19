"""
StarterAgent - a teaching-friendly scripted agent with a tiny decision tree.

Flow (single pass, easy to read):
1. If energy is low -> walk to the charger and recharge.
2. Else if carrying a heart -> deliver it to the chest.
3. Else if we hold every ingredient -> assemble a heart.
4. Else -> gather missing resources in a fixed order (carbon, oxygen, germanium, silicon).

Built to be copy/pastable into README snippets and runnable via:
`uv run cogames play --mission evals.extractor_hub_30 -p scripted_starter --steps 600`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

from .baseline_agent import BaselineAgentPolicyImpl
from .types import BaselineHyperparameters, ExtractorInfo, Phase, SimpleAgentState


@dataclass
class StarterHyperparameters(BaselineHyperparameters):
    """
    Keep hyperparameters intentionally sparse so the core flow stays visible.

    gather_order: fixed order we try to satisfy the heart recipe with.
    """

    gather_order: tuple[str, ...] = ("carbon", "oxygen", "germanium", "silicon")


class StarterAgentPolicyImpl(BaselineAgentPolicyImpl):
    """
    Minimal if/else agent built on top of the BaselineAgent parsing/navigation helpers.

    The only logic we override is:
    - phase selection (_update_phase)
    - resource preference (_find_any_needed_extractor)
    - deficit calculation fallback (_calculate_deficits)
    """

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        hyperparams: Optional[StarterHyperparameters] = None,
    ):
        self._hyperparams: StarterHyperparameters = hyperparams or StarterHyperparameters()
        super().__init__(policy_env_info, agent_id, self._hyperparams)

    def _calculate_deficits(self, s: SimpleAgentState) -> dict[str, int]:
        """
        BaselineAgent requires the recipe to be discovered. The starter agent falls back to a
        1-each recipe so it can keep moving even before the assembler is observed.
        """
        if s.heart_recipe is None:
            return {
                "carbon": max(0, 1 - s.carbon),
                "oxygen": max(0, 1 - s.oxygen),
                "germanium": max(0, 1 - s.germanium),
                "silicon": max(0, 1 - s.silicon),
            }

        return super()._calculate_deficits(s)

    def _update_phase(self, s: SimpleAgentState) -> None:
        """Small decision tree readable enough for README snippets."""
        old_phase = s.phase

        if s.energy < self._hyperparams.recharge_threshold_low:
            s.phase = Phase.RECHARGE
        elif s.hearts > 0:
            s.phase = Phase.DELIVER
        else:
            heart_recipe = s.heart_recipe or {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}
            has_all_inputs = all(getattr(s, res, 0) >= amount for res, amount in heart_recipe.items())
            s.phase = Phase.ASSEMBLE if has_all_inputs else Phase.GATHER

        if s.phase != old_phase:
            # Reset cached paths when we change goals so navigation recalculates cleanly.
            s.cached_path = None
            s.cached_path_target = None

    def _find_any_needed_extractor(self, s: SimpleAgentState) -> Optional[tuple[ExtractorInfo, str]]:
        """
        Prefer a fixed gather order instead of "largest deficit first" used by BaselineAgent.
        Keeps the behavior predictable for walkthroughs.
        """
        deficits = self._calculate_deficits(s)

        for resource in self._hyperparams.gather_order:
            if deficits.get(resource, 0) > 0:
                extractor = self._find_nearest_extractor(s, resource)
                if extractor is not None:
                    return (extractor, resource)

        # Nothing needed or nothing found: fall back to the parent search (which can still explore).
        return super()._find_any_needed_extractor(s)


class StarterPolicy(MultiAgentPolicy):
    """
    Multi-agent wrapper so the starter agent can be launched with `-p scripted_starter`.
    """

    short_names = ["scripted_starter", "starter", "starter_agent"]

    def __init__(self, policy_env_info: PolicyEnvInterface, hyperparams: Optional[StarterHyperparameters] = None):
        super().__init__(policy_env_info)
        self._hyperparams: StarterHyperparameters = hyperparams or StarterHyperparameters()
        self._agent_policies: dict[int, StatefulAgentPolicy[SimpleAgentState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[SimpleAgentState]:
        if agent_id not in self._agent_policies:
            impl = StarterAgentPolicyImpl(self._policy_env_info, agent_id, self._hyperparams)
            self._agent_policies[agent_id] = StatefulAgentPolicy(impl, self._policy_env_info, agent_id=agent_id)
        return self._agent_policies[agent_id]
