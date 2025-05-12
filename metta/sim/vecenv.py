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
        logger.warning("MettaGridEnv._c_env is None")
        # You might need to add code here to properly initialize _c_env
        # This depends on how MettaGridEnv is designed

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
        logger.error(f"num_envs is {num_envs}, which is less than 1!")

    env_kwargs = {
        "cfg": env_cfg,
        "render_mode": render_mode,
        "stats_writer": stats_writer,
        "replay_writer": replay_writer,
    }

    # Create lists of environment creators, args, and kwargs for each environment
    env_creators = [make_env_func] * num_envs
    env_args_list = [[]] * num_envs  # Empty args for each environment
    env_kwargs_list = [env_kwargs] * num_envs  # Same kwargs for each environment

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
