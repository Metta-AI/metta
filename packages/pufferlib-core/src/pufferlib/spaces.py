import gymnasium
import numpy as np

try:
    import gym
except ImportError:
    # Alias gymnasium as gym when gym is not available
    gym = gymnasium

Box = (gym.spaces.Box, gymnasium.spaces.Box)
Dict = (gym.spaces.Dict, gymnasium.spaces.Dict)
Discrete = (gym.spaces.Discrete, gymnasium.spaces.Discrete)
MultiBinary = (gym.spaces.MultiBinary, gymnasium.spaces.MultiBinary)
Tuple = (gym.spaces.Tuple, gymnasium.spaces.Tuple)


def joint_space(space, n):
    if isinstance(space, Discrete):
        max_value = np.int32(space.n - 1)
        return gymnasium.spaces.Box(low=np.int32(0), high=max_value, shape=(n,), dtype=np.int32)
    elif isinstance(space, Box):
        low = np.repeat(space.low[None], n, axis=0)
        high = np.repeat(space.high[None], n, axis=0)
        return gymnasium.spaces.Box(low=low, high=high, shape=(n, *space.shape), dtype=space.dtype)
    else:
        raise ValueError(f"Unsupported space: {space}")
