import importlib
import os
import random
import sys

import numpy as np

from cogames.cli.mission import get_mission
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.join(current_dir, "bindings/generated")
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

fa = importlib.import_module("fast_agents")

# create an env config first
_, env_cfg, _ = get_mission("evals.extractor_hub_30", variants_arg=["lonely_heart"], cogs=1)
policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)

# let's create an env
sim = Simulation(env_cfg, seed=random.randint(0, 2**31 - 1))
raw_observation = sim._c_sim.observations()  # returns some numpy array, shape (num_agents, num_tokens, num_features)

assert raw_observation.shape == (1, 200, 3)  # 1 agent, 200 tokens, 3

# initialize nim agent
agent = fa.ThinkyAgent(0, policy_env_info.to_json())
agent.reset()

# get the next action from the nim agent
raw_actions = np.zeros(1, dtype=np.int32)
agent_id = 0
for i in range(10):
    agent.step(
        num_agents=1,
        num_tokens=200,
        size_token=3,
        raw_observations=raw_observation.ctypes.data,
        num_actions=1,
        raw_actions=raw_actions.ctypes.data,
    )
    action = int(raw_actions[0])
    print(f"step {i}: action: {action} ({policy_env_info.actions.actions()[action].name})")

    # step environment
    sim._c_sim.actions()[agent_id] = action
    sim.step()

    # update observations for next iteration
    raw_observation = sim._c_sim.observations()
