import logging
from typing import Any, Optional

import pufferlib
import pufferlib.vector
from pufferlib.pufferlib import set_buffers
from pydantic import validate_call

from metta.cogworks.curriculum import Curriculum, CurriculumEnv
from metta.common.util.log_config import init_logging
from metta.mettagrid import MettaGridEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter
from metta.rl.periodic_reset_env import PeriodicResetConfig, PeriodicResetEnv

logger = logging.getLogger("vecenv")


def _extract_periodic_reset_config(
    curriculum: Curriculum, override_config: Optional[PeriodicResetConfig]
) -> Optional[PeriodicResetConfig]:
    """Extract periodic reset configuration from curriculum or override.

    Args:
        curriculum: The curriculum to check for embedded config
        override_config: Optional override configuration

    Returns:
        The periodic reset config to use, or None if not configured
    """
    if override_config is not None:
        return override_config

    # Check if the current task has embedded periodic reset config
    current_task = curriculum.get_task()
    env_cfg = current_task.get_env_cfg()

    # Look for the config in a standard way
    return getattr(env_cfg, "periodic_reset_config", None)


@validate_call(config={"arbitrary_types_allowed": True})
def make_env_func(
    curriculum: Curriculum,
    render_mode="rgb_array",
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayWriter] = None,
    is_training: bool = False,
    is_serial: bool = False,
    run_dir: str | None = None,
    buf: Optional[Any] = None,
    periodic_reset_config: Optional[PeriodicResetConfig] = None,
    **kwargs,
):
    if not is_serial:
        # Running in a new process, so we need to reinitialize logging
        init_logging(run_dir=run_dir)

    # Extract periodic reset configuration
    final_periodic_reset_config = _extract_periodic_reset_config(curriculum, periodic_reset_config)

    # Create base environment
    current_task = curriculum.get_task()
    env_cfg = current_task.get_env_cfg()

    env = MettaGridEnv(
        env_cfg,
        render_mode=render_mode,
        stats_writer=stats_writer,
        replay_writer=replay_writer,
        is_training=is_training,
    )
    set_buffers(env, buf)

    # Apply curriculum wrapper
    env = CurriculumEnv(env, curriculum)

    # Apply periodic reset wrapper if configured
    if final_periodic_reset_config is not None:
        env = PeriodicResetEnv(env, final_periodic_reset_config)

    return env


@validate_call(config={"arbitrary_types_allowed": True})
def make_vecenv(
    curriculum: Curriculum,
    vectorization: str,
    num_envs: int = 1,
    batch_size: int | None = None,
    num_workers: int = 1,
    render_mode: str | None = None,
    stats_writer: StatsWriter | None = None,
    replay_writer: ReplayWriter | None = None,
    is_training: bool = False,
    run_dir: str | None = None,
    periodic_reset_config: Optional[PeriodicResetConfig] = None,
    **kwargs,
) -> Any:  # Returns pufferlib VecEnv instance
    # Determine the vectorization class
    is_serial = vectorization == "serial" or num_workers == 1

    if is_serial:
        vectorizer_cls = pufferlib.vector.Serial
    elif vectorization == "multiprocessing":
        vectorizer_cls = pufferlib.vector.Multiprocessing
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
        "periodic_reset_config": periodic_reset_config,
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
