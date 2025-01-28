import gymnasium as gym
from mettagrid.mettagrid_env import MettaGridEnv


class SingleAgentSpaceWrapper(gym.Wrapper):
    def __init__(self, env: MettaGridEnv):
        super().__init__(env)
    
    @property
    def observation_space(self):
        return self.single_observation_space
    
    @property
    def action_space(self):
        return self.single_action_space

