import numpy as np
import torch
from omegaconf import OmegaConf

from metta.agent.policy_store import PolicyRecord
from metta.rl.pufferlib.vecenv import make_vecenv

class Simulator:
    """Simulate a policy for playing or tracing the environment"""

    def __init__(self, cfg: OmegaConf, env_cfg: OmegaConf, policy_record: PolicyRecord, num_steps: int = 500):
        """Initialize the simulator"""
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.device = cfg.device
        self.vecenv = make_vecenv(env_cfg, cfg.vectorization, num_envs=1, render_mode="human")
        self.obs, _ = self.vecenv.reset()
        self.env = self.vecenv.envs[0]
        self.policy_record = policy_record
        self.policy = self.policy_record.policy()
        self.policy_rnn_state = None
        self.rewards = np.zeros(self.vecenv.num_agents)
        self.total_rewards = np.zeros(self.vecenv.num_agents)
        self.num_agents = self.vecenv.num_agents
        self.num_steps = 500
        self.dones = np.zeros(self.vecenv.num_agents)
        self.trunc = np.zeros(self.vecenv.num_agents)

    def actions(self):
        """Get the actions for the current timestep"""
        with torch.no_grad():
            obs = torch.as_tensor(self.obs).to(device=self.device)
            actions, _, _, _, self.policy_rnn_state, _, _, _ = self.policy(obs, self.policy_rnn_state)
        return actions

    def step(self, actions):
        """Step the simulator forward one timestep"""
        (self.obs, self.rewards, self.dones, self.trunc, self.infos) = self.vecenv.step(actions.cpu().numpy())
        self.total_rewards += self.rewards

    def done(self):
        """Check if the episode is done"""
        return any(self.dones) or any(self.trunc)

    def run(self):
        """Run the simulator until the episode is done"""
        while not self.done():
            actions = self.actions()
            self.step(actions)

    def grid_objects(self):
        """Get the grid objects in the environment"""
        return self.env.grid_objects.values()
