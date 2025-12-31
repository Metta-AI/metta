"""
Sample Policy for the Cogames environment.

Helps a Cog move carbon from an extractor to a chest.

Note to users of this policy:
We don't intend for scripted policies to be the final word on how policies are generated (e.g., we expect the
environment to be complicated enough that trained agents will be necessary). So we expect that scripting policies
is a good way to start, but don't want you to get stuck here. Feel free to prove us wrong!

Note to cogames developers:
This policy should be kept relatively minimalist, without dependencies on intricate algorithms.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation


@dataclass
class StarterCogState:
    target_location: tuple[int, int] | None = None
    chest_location: tuple[int, int] | None = None
    resource_to_collect: str = "carbon"
    # Location of an extractor for the target resource
    extractor_location: tuple[int, int] | None = None
    # Current position relative to the starting position.
    # We expect some moves to fail, so all positions should be treated somewhat loosely.
    position: tuple[int, int] = (0, 0)
    have_inventory: bool = False


class StarterCogPolicyImpl(StatefulPolicyImpl[StarterCogState]):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
    ):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info

        # Action lookup
        self._actions = policy_env_info.actions

    def _parse_observation(self, obs: AgentObservation, state: StarterCogState) -> StarterCogState:
        """Parse the observation and update the state."""

        extractor_tag_value = self._policy_env_info.tags.index(f"{state.resource_to_collect}_extractor")
        chest_tag_value = self._policy_env_info.tags.index("chest")
        state.have_inventory = False
        for token in obs.tokens:
            if token.feature.name == "last_action":
                # Update our current (relative) position.
                # TODO: This is wrong if we moved to interact with a station.
                if token.value == self._policy_env_info.action_names.index("move_north"):
                    state.position = (state.position[0] - 1, state.position[1])
                elif token.value == self._policy_env_info.action_names.index("move_south"):
                    state.position = (state.position[0] + 1, state.position[1])
                elif token.value == self._policy_env_info.action_names.index("move_west"):
                    state.position = (state.position[0], state.position[1] - 1)
                elif token.value == self._policy_env_info.action_names.index("move_east"):
                    state.position = (state.position[0], state.position[1] + 1)
                break
        for token in obs.tokens:
            if token.location == (5, 5):
                if token.feature.name == f"inv:{state.resource_to_collect}":
                    state.have_inventory = True
                continue
            token_location = (state.position[0] + token.location[0] - 5, state.position[1] + token.location[1] - 5)
            if token.feature.name == "tag":
                if token.value == extractor_tag_value:
                    state.extractor_location = token_location
                elif token.value == chest_tag_value:
                    state.chest_location = token_location
            # It would probably be a good idea to keep track of obstacles
        return state

    def _go_to(self, state: StarterCogState, target: tuple[int, int]) -> Tuple[Action, StarterCogState]:
        """Go to the target location."""
        # Let's just go straight there! I hope we don't run into anything and get stuck.
        possible_actions = []
        if state.position[0] < target[0]:
            possible_actions.append("move_south")
        if state.position[0] > target[0]:
            possible_actions.append("move_north")
        if state.position[1] < target[1]:
            possible_actions.append("move_east")
        if state.position[1] > target[1]:
            possible_actions.append("move_west")
        action = Action(name=random.choice(possible_actions))
        return action, state

    def step_with_state(self, obs: AgentObservation, state: StarterCogState) -> Tuple[Action, StarterCogState]:
        """Compute the action for this Cog."""
        state = self._parse_observation(obs, state)
        if state.have_inventory:
            if state.chest_location is not None:
                return self._go_to(state, state.chest_location)
        else:
            if state.extractor_location is not None:
                return self._go_to(state, state.extractor_location)
        direction = random.choice(["north", "south", "east", "west"])
        return Action(name="move_" + direction), state

    def initial_agent_state(self) -> StarterCogState:
        """Get the initial state for a new agent."""
        return StarterCogState()


# ============================================================================
# Policy Wrapper Classes
# ============================================================================


class StarterPolicy(MultiAgentPolicy):
    # short_names = ["scripted_starter"]  # Uncomment to register a shorthand

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu"):
        super().__init__(policy_env_info, device=device)
        self._agent_policies: dict[int, StatefulAgentPolicy[StarterCogState]] = {}

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[StarterCogState]:
        if agent_id not in self._agent_policies:
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                StarterCogPolicyImpl(self._policy_env_info, agent_id),
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]
