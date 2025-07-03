import torch

from metta.agent.metta_agent import MettaAgent


class MockAgent(MettaAgent):
    """
    An agent that always does nothing. Used for tests and to run play without requiring a policy
    """

    def __init__(self):
        pass

    def activate_actions(self, *args):
        pass

    def __call__(self, obs, state):
        num_agents = obs.shape[0]
        return (torch.zeros((num_agents, 2)), None, None, None, None)

    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device,
        is_training: bool = True,
    ):
        """
        Dummy implementation to satisfy the simulation interface.
        MockAgent doesn't need to actually initialize anything.
        Note: is_training parameter is deprecated and ignored.
        """
        pass
