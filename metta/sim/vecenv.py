<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
import logging
from typing import Optional

import hydra
>>>>>>> ecf239561a2376997adee9a5cf9cb92fb998aaba
import pufferlib
import pufferlib.vector

<<<<<<< HEAD
<<<<<<< HEAD
from mettagrid.curriculum.curriculum import Curriculum
from mettagrid.mettagrid_env import MettaGridEnv


def make_env_func(curriculum: Curriculum, buf=None, render_mode="rgb_array"):
    return MettaGridEnv(curriculum=curriculum, buf=buf, render_mode=render_mode)
=======
=======
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
from metta.util.resolvers import register_resolvers
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
    # we are not calling into our configs hierarchy here so we need to manually register the custom resolvers
    register_resolvers()

    # Create the environment instance
    env = hydra.utils.instantiate(
        cfg, cfg, render_mode=render_mode, buf=buf, stats_writer=stats_writer, replay_writer=replay_writer, **kwargs
    )
    # Ensure the environment is properly initialized
    if hasattr(env, "_c_env") and env._c_env is None:
        raise ValueError("MettaGridEnv._c_env is None after hydra instantiation")
    return env
<<<<<<< HEAD
>>>>>>> ecf239561a2376997adee9a5cf9cb92fb998aaba
=======
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87


def make_vecenv(
    curriculum: Curriculum,
    vectorization: str,
    num_envs=1,
    batch_size=None,
    num_workers=1,
    render_mode=None,
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayWriter] = None,
    **kwargs,
):
    # Determine the vectorization class
    if vectorization == "serial" or num_workers == 1:
        vectorizer_cls = pufferlib.vector.Serial
    elif vectorization == "multiprocessing":
        vectorizer_cls = pufferlib.vector.Multiprocessing
    elif vectorization == "ray":
        vectorizer_cls = pufferlib.vector.Ray
    else:
        raise ValueError("Invalid --vector (serial/multiprocessing/ray).")

<<<<<<< HEAD
<<<<<<< HEAD
    vecenv_args = dict(
        env_kwargs=dict(curriculum=curriculum, render_mode=render_mode),
=======
=======
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
    # Check if num_envs is valid
    if num_envs < 1:
        raise ValueError(f"num_envs must be at least 1, got {num_envs}")

    env_kwargs = {
        "cfg": env_cfg,
        "render_mode": render_mode,
        "stats_writer": stats_writer,
        "replay_writer": replay_writer,
    }

    # Note: PufferLib's vector.make accepts Serial, Multiprocessing, and Ray as valid backends,
    # but the type annotations only allow PufferEnv.
    vecenv = pufferlib.vector.make(
        make_env_func,
        env_kwargs=env_kwargs,
        backend=vectorizer_cls,  # type: ignore - PufferEnv inferred type is incorrect
<<<<<<< HEAD
>>>>>>> ecf239561a2376997adee9a5cf9cb92fb998aaba
=======
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,
        **kwargs,
    )

    return vecenv
