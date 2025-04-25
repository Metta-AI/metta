import numpy as np
import torch

from metta.agent.policy_store import PolicyRecord, PolicyStore
from metta.sim.simulation_config import SimulationConfig
from metta.sim.vecenv import make_vecenv
from metta.util.config import config_from_path
from mettagrid.renderer.raylib.raylib_renderer import MettaGridRaylibRenderer


# TODO: Merge with Simulation
class Simulator:
    """Simulate a policy for playing or tracing the environment"""

    def __init__(self, config: SimulationConfig, policy_record: PolicyRecord, num_steps: int = 500):
        """Initialize the simulator"""
        self.config = config
        self.device = config.device
        self.env_cfg = config_from_path(config.env, config.env_overrides)
        self.vecenv = make_vecenv(self.env_cfg, config.vectorization, num_envs=1, render_mode="human")
        self.obs, _ = self.vecenv.reset()
        self.env = self.vecenv.envs[0]
        self.policy_record = policy_record
        self.policy = self.policy_record.policy()
        self.policy_rnn_state = None
        self.rewards = np.zeros(self.vecenv.num_agents)
        self.total_rewards = np.zeros(self.vecenv.num_agents)
        self.num_agents = self.vecenv.num_agents
        self.num_steps = num_steps
        self.dones = np.zeros(self.vecenv.num_agents)
        self.trunc = np.zeros(self.vecenv.num_agents)

    def actions(self):
        """Get the actions for the current timestep"""
        with torch.no_grad():
            obs = torch.as_tensor(self.obs).to(device=self.device)
            # actions, _, _, _, self.policy_rnn_state, _, _, _ = self.policy(obs, self.policy_rnn_state)
            actions, self.policy_rnn_state = self.policy.inference(obs, self.policy_rnn_state)

            # TODO - fix inference and then remove this patch

            # Ensure actions have the correct shape: (num_agents, 2)
            expected_shape = (self.num_agents, 2)
            if actions.shape != expected_shape:
                # Debug message (can be removed in production)
                print(f"Reshaping actions from {actions.shape} to {expected_shape}")

                if actions.numel() == expected_shape[0] * expected_shape[1]:
                    print("Case 1: If we have a flat tensor with correct total elements")
                    actions = actions.reshape(expected_shape)
                elif actions.dim() == 1 and actions.size(0) == 2:
                    print("Case 2: If we have a 1D tensor of length 2 (single action)")
                    actions = actions.reshape(expected_shape)
                    # Expand single action to all agents if needed
                    actions = actions.unsqueeze(0).expand(self.num_agents, -1)
                else:
                    print("Case 3: Other unexpected shapes")
                    # Create a default action tensor if needed
                    actions = torch.zeros(expected_shape, dtype=actions.dtype, device=actions.device)
            else:
                print(f"actions {actions.shape} had the expected shape {expected_shape}")

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


# TODO: Merge with Simulation
def play(config: SimulationConfig, policy_store: PolicyStore, policy_uri: str):
    device = config.device
    env_cfg = config_from_path(config.env, config.env_overrides)
    vecenv = make_vecenv(env_cfg, config.vectorization, num_envs=1, render_mode="human")

    obs, _ = vecenv.reset()
    env = vecenv.envs[0]
    policy_record = policy_store.policy(policy_uri)

    assert policy_record.metadata["action_names"] == env._c_env.action_names(), (
        f"Action names do not match: {policy_record.metadata['action_names']} != {env._c_env.action_names()}"
    )
    policy = policy_record.policy()

    renderer = MettaGridRaylibRenderer(env._c_env, env._env_cfg.game)
    policy_rnn_state = None

    rewards = np.zeros(vecenv.num_agents)
    total_rewards = np.zeros(vecenv.num_agents)

    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device=device)

            # Parallelize across opponents
            actions, _, _, _, policy_rnn_state, _, _, _ = policy(obs, policy_rnn_state)
            if actions.dim() == 0:  # scalar tensor like tensor(2)
                actions = torch.tensor([actions.item()])

        renderer.update(
            actions.cpu().numpy(),
            obs,
            rewards,
            total_rewards,
            env._c_env.current_timestep(),
        )
        renderer.render_and_wait()
        actions = renderer.get_actions()

        obs, rewards, dones, truncated, infos = vecenv.step(actions)
        total_rewards += rewards
        if any(dones) or any(truncated):
            print(f"Total rewards: {total_rewards}")
            break

    import pandas as pd

    agent_df = pd.DataFrame([infos[0]["agent"]]).T.reset_index()
    agent_df.columns = ["stat", "value"]
    agent_df = agent_df.sort_values("stat")
    game_df = pd.DataFrame([infos[0]["game"]]).T.reset_index()
    game_df.columns = ["stat", "value"]
    game_df = game_df.sort_values("stat")
    print("\nAgent stats:")
    print(agent_df.to_string(index=False))
    print("\nGame stats:")
    print(game_df.to_string(index=False))
