import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))

import heuristic_agents as ha
import numpy as np


# Initialize the DLL (sets control-c handler).
ha.init()

# Test single agent policy
num_agents = 1
num_tokens = 200
size_token = 3
num_actions = 1
agent = ha.HeuristicAgent(0, "{}")
observations = np.zeros((num_agents, num_tokens, size_token), dtype=np.uint8)
actions = np.zeros((num_agents), dtype=np.uint8)
agent.step(
  num_agents,
  num_tokens,
  size_token,
  observations.ctypes.data,
  num_actions,
  actions.ctypes.data
)
print(actions)
agent.reset()
