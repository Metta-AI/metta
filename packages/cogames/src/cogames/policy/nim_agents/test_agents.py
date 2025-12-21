import ctypes
import os
import sys

import numpy as np


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "bindings/generated"))

    import nim_agents as fa  # type: ignore[import-not-found]

    fa.nim_agents_init_chook()

    num_agents = 2
    num_tokens = 50
    size_token = 3
    observations = np.zeros((num_agents, num_tokens, size_token), dtype=np.uint8)
    actions = np.zeros(num_agents, dtype=np.int32)

    policy = fa.RandomPolicy("{}")
    subset = np.array([0, 1], dtype=np.int32)
    subset_ptr = subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    policy.step_batch(
        subset_ptr,
        subset.size,
        num_agents,
        num_tokens,
        size_token,
        observations.ctypes.data,
        num_agents,
        actions.ctypes.data,
    )
    print(actions)

    single_subset = np.array([0], dtype=np.int32)
    single_ptr = single_subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    policy.step_batch(
        single_ptr,
        single_subset.size,
        num_agents,
        num_tokens,
        size_token,
        observations.ctypes.data,
        num_agents,
        actions.ctypes.data,
    )
    print(actions[0])


if __name__ == "__main__":
    main()
