from __future__ import annotations

import os
import sys

import numpy as np


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bindings_dir = os.path.join(current_dir, "bindings/generated")
    if bindings_dir not in sys.path:
        sys.path.append(bindings_dir)

    import heuristic_agents as ha

    ha.init()

    num_agents = 1
    num_tokens = 200
    size_token = 3
    num_actions = 1
    agent = ha.HeuristicAgent(0, "{}")
    observations = np.zeros((num_agents, num_tokens, size_token), dtype=np.uint8)
    actions = np.zeros((num_agents), dtype=np.uint8)
    agent.step(num_agents, num_tokens, size_token, observations.ctypes.data, num_actions, actions.ctypes.data)
    print(actions)
    agent.reset()


if __name__ == "__main__":
    main()
