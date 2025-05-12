import logging
from typing import Optional

import hydra
import pufferlib
import pufferlib.vector
from omegaconf import DictConfig, ListConfig

from mettagrid.mettagrid_env import MettaGridEnv
from mettagrid.replay_writer import ReplayWriter
from mettagrid.stats_writer import StatsWriter

logger = logging.getLogger("vecenv")


def make_env_func(
    cfg: DictConfig,
    buf=None,
    render_mode="rgb_array",
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayWriter] = None,
    **kwargs,
):
    # Create the environment instance
    env = hydra.utils.instantiate(
        cfg, cfg, render_mode=render_mode, buf=buf, stats_writer=stats_writer, replay_writer=replay_writer, **kwargs
    )

    # Ensure the environment is properly initialized
    if hasattr(env, "_c_env") and env._c_env is None:
        raise ValueError("MettaGridEnv._c_env is None after hydra instantiation")

    return env


def make_vecenv(
    env_cfg: DictConfig | ListConfig,
    vectorization: str,
    num_envs=1,
    batch_size=None,
    num_workers=1,
    render_mode=None,
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayWriter] = None,
    **kwargs,
):
    # Determine the vectorizer class
    if vectorization == "serial" or num_workers == 1:
        vectorizer_cls = pufferlib.vector.Serial
    elif vectorization == "multiprocessing":
        vectorizer_cls = pufferlib.vector.Multiprocessing
    elif vectorization == "ray":
        vectorizer_cls = pufferlib.vector.Ray
    else:
        raise ValueError("Invalid --vector (serial/multiprocessing/ray).")

    # Check if num_envs is valid
    if num_envs < 1:
        raise ValueError(f"num_envs must be at least 1, got {num_envs}")

    env_kwargs = {
        "cfg": env_cfg,
        "render_mode": render_mode,
        "stats_writer": stats_writer,
        "replay_writer": replay_writer,
    }

    # Create lists of environment creators, args, and kwargs for each environment
    # Using list comprehension to create independent copies of arguments and keyword arguments
    env_creators = [make_env_func for _ in range(num_envs)]
    env_args_list = [[] for _ in range(num_envs)]  # Independent empty args lists for each environment
    env_kwargs_list = [{**env_kwargs} for _ in range(num_envs)]  # Independent kwargs dicts for each environment

    vecenv = vectorizer_cls(
        env_creators,  # First positional argument
        env_args_list,  # Second positional argument
        env_kwargs_list,  # Third positional argument
        num_envs,  # Fourth positional argument
        num_workers=num_workers,  # Keyword arguments from here
        batch_size=batch_size or num_envs,
        backend=MettaGridEnv,
        **kwargs,
    )

    return vecenv
