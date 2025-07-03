import torch

from metta.agent.metta_agent import MettaAgent


class MockAgent(MettaAgent):
    """
    A fake agent that does nothing, used to run play without requiring a policy to be trained
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
        FakeAgent doesn't need to actually initialize anything.
        Note: is_training parameter is deprecated and ignored.
        """
        pass
