import ctypes
import os
import sys

import numpy as np


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "bindings/generated"))

    import fast_agents as fa

    # Initialize the DLL (sets control-c handler).
    fa.init()

    # Test single agent policy
    num_agents = 1
    num_tokens = 200
    size_token = 3
    num_actions = 1
    agent = fa.RandomAgent(0, "{}")
    observations = np.zeros((num_agents, num_tokens, size_token), dtype=np.uint8)
    actions = np.zeros((num_agents), dtype=np.int32)
    agent.step_batch(
        num_agents,
        num_tokens,
        size_token,
        observations.ctypes.data_as(ctypes.c_void_p),
        num_actions,
        actions.ctypes.data_as(ctypes.c_void_p),
    )
    print(actions)
    single_action = agent.step(num_tokens, size_token, observations[0].ctypes.data_as(ctypes.c_void_p))
    print(single_action)
    agent.reset()


if __name__ == "__main__":
    main()
