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
from metta.sim.tribal_genny import TribalGridEnv

logger = logging.getLogger("vecenv")


@validate_call(config={"arbitrary_types_allowed": True})
def make_env_func(
    curriculum: Curriculum,
    render_mode="rgb_array",
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayWriter] = None,
    is_training: bool = False,
    run_dir: str | None = None,
    buf: Optional[Any] = None,
    **kwargs,
):
    init_logging(run_dir=run_dir)

    env_cfg = curriculum.get_task().get_env_cfg()

    # Check if this is a tribal environment
    if hasattr(env_cfg, "environment_type") and env_cfg.environment_type == "tribal":
        # Convert TribalEnvConfig to dict for TribalGridEnv
        tribal_config = {
            "max_steps": env_cfg.game.max_steps,
            "ore_per_battery": env_cfg.game.ore_per_battery,
            "batteries_per_heart": env_cfg.game.batteries_per_heart,
            "enable_combat": env_cfg.game.enable_combat,
            "clippy_spawn_rate": env_cfg.game.clippy_spawn_rate,
            "clippy_damage": env_cfg.game.clippy_damage,
            "heart_reward": env_cfg.game.heart_reward,
            "ore_reward": env_cfg.game.ore_reward,
            "battery_reward": env_cfg.game.battery_reward,
            "survival_penalty": env_cfg.game.survival_penalty,
            "death_penalty": env_cfg.game.death_penalty,
        }

        env = TribalGridEnv(tribal_config, render_mode=render_mode, buf=buf)
        # TribalGridEnv handles PufferLib buffer setup internally
    else:
        # Standard MettaGrid environment
        env = MettaGridEnv(
            env_cfg,
            render_mode=render_mode,
            stats_writer=stats_writer,
            replay_writer=replay_writer,
            is_training=is_training,
        )
        # Only set buffers for non-tribal environments
        set_buffers(env, buf)
    env = CurriculumEnv(env, curriculum)

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
