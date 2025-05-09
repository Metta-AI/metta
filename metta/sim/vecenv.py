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
    return hydra.utils.instantiate(
        cfg, cfg, render_mode=render_mode, buf=buf, stats_writer=stats_writer, replay_writer=replay_writer, **kwargs
    )


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

    vecenv = vectorizer_cls(
        env_fn=make_env_func,
        env_kwargs=env_kwargs,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,
        backend=MettaGridEnv,
        **kwargs,
    )
    return vecenv
