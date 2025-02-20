import os
import hydra
import json
from omegaconf import OmegaConf
import torch
import numpy as np
from rl.pufferlib.vecenv import make_vecenv
from agent.policy_store import PolicyStore
from agent.policy_store import PolicyRecord


def nice_orientation(orientation):
    """ Convert an orientation into a human-readable string """
    return ["north", "south", "west", "east"][orientation % 4]


def nice_actions(env, action):
    """ Convert a un-flattened action into a human-readable string """
    name = env.action_names()[action[0]]
    if name == "move":
        return name + ("_back", "_forward")[action[1] % 2]
    elif name == "rotate":
        return "rotate_" + nice_orientation(action[1])
    elif name == "attack":
        return "attack_" + str(action[1] // 3) + "_" + str(action[1] % 3)
    else:
        return name


class Simulator:
    """ Simulate a policy for playing or tracing the environment """

    def __init__(self, cfg: OmegaConf, policy_record: PolicyRecord):
        self.cfg = cfg
        self.policy_record = policy_record
        self.device = cfg.device
        self.vecenv = make_vecenv(
          cfg.env,
          cfg.vectorization,
          num_envs=1,
          render_mode="human"
        )
        self.obs, _ = self.vecenv.reset()
        self.env = self.vecenv.envs[0]
        self.policy = self.policy_record.policy()
        self.policy_rnn_state = None
        self.rewards = np.zeros(self.vecenv.num_agents)
        self.total_rewards = np.zeros(self.vecenv.num_agents)
        self.num_agents = self.vecenv.num_agents
        self.num_steps = 500

    def step(self):
        """ Step the simulator forward one timestep """
        with torch.no_grad():
            obs = torch.as_tensor(self.obs).to(device=self.device)
            actions, _, _, _, self.policy_rnn_state, _, _ = self.policy(
              obs,
              self.policy_rnn_state
            )

        actions_array = actions.cpu().numpy()
        step_info = []
        for id, action in enumerate(actions_array):
            for grid_object in self.env.grid_objects.values():
                if "agent_id" in grid_object and grid_object["agent_id"] == id:
                    agent = grid_object
                    break

            step_info.append({
                "agent": id,
                "action": action.tolist(),
                "action_name": nice_actions(self.env, action),
                "reward": self.rewards[id].item(),
                "total_reward": self.total_rewards[id].item(),
                "position": [agent["c"], agent["r"]],
                # "energy": agent["agent:energy"],
                "hp": agent["agent:hp"],
                "frozen": agent["agent:frozen"],
                "orientation": nice_orientation(agent["agent:orientation"]),
                # "shield": agent["agent:shield"],
                # "inventory": agent["agent:inv:r1"]
            })

        (self.obs, self.rewards, self.dones, self.trunc, self.infos) = \
          self.vecenv.step(actions.cpu().numpy())
        self.total_rewards += self.rewards

        for i in range(len(self.env.action_success)):
            step_info[i]["action_success"] = self.env.action_success[i]

        return step_info

    def run(self):
        """ Run the simulator until the episode is done """
        steps = []

        while True:
            steps.append(self.step())
            if any(self.dones) or any(self.trunc):
                break

        return steps
