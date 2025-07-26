import logging
from typing import Optional

import pufferlib
import pufferlib.vector
from pydantic import validate_call

from metta.common.util.logging_helpers import init_logging
from metta.common.util.resolvers import register_resolvers
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid import MettaGridEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

logger = logging.getLogger("vecenv")


@validate_call(config={"arbitrary_types_allowed": True})
def make_env_func(
    curriculum: Curriculum,
    buf=None,
    render_mode="rgb_array",
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayWriter] = None,
    is_training: bool = False,
    is_serial: bool = False,
    run_dir: str | None = None,
    **kwargs,
):
    if not is_serial:
        # Running in a new process, so we need to reinitialize logging and resolvers
        register_resolvers()
        init_logging(run_dir=run_dir)

    # Create the environment instance
    env = MettaGridEnv(
        curriculum,
        render_mode=render_mode,
        buf=buf,
        stats_writer=stats_writer,
        replay_writer=replay_writer,
        is_training=is_training,
        **kwargs,
    )
    # Ensure the environment is properly initialized
    if hasattr(env, "_c_env") and env._c_env is None:
        raise ValueError("MettaGridEnv._c_env is None after hydra instantiation")
    return env


@validate_call(config={"arbitrary_types_allowed": True})
def make_vecenv(
    curriculum: Curriculum,
    vectorization: str,
    num_envs=1,
    batch_size=None,
    num_workers=1,
    render_mode=None,
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayWriter] = None,
    is_training: bool = False,
    run_dir: str | None = None,
    **kwargs,
):
    # Determine the vectorization class
    is_serial = vectorization == "serial" or num_workers == 1

    if is_serial:
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
        "curriculum": curriculum,
        "render_mode": render_mode,
        "stats_writer": stats_writer,
        "replay_writer": replay_writer,
        "is_training": is_training,
        "is_serial": is_serial,
        "run_dir": run_dir,
    }

    # Note: PufferLib's vector.make accepts Serial, Multiprocessing, and Ray as valid backends,
    # but the type annotations only allow PufferEnv.
    vecenv = pufferlib.vector.make(
        make_env_func,
        env_kwargs=env_kwargs,
        backend=vectorizer_cls,  # type: ignore - PufferEnv inferred type is incorrect
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=batch_size or num_envs,
        **kwargs,
    )

    return vecenv
