import logging
from pathlib import Path
from typing import Any, Optional

from pydantic import validate_call

import pufferlib
import pufferlib.vector
from metta.cogworks.curriculum import Curriculum, CurriculumEnv
from metta.common.util.log_config import init_logging
from metta.sim.replay_log_writer import ReplayLogWriter
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.simulator import Simulator
from mettagrid.util.stats_writer import NoopStatsWriter, StatsWriter

logger = logging.getLogger("vecenv")


@validate_call(config={"arbitrary_types_allowed": True})
def make_env_func(
    curriculum: Curriculum,
    stats_writer: Optional[StatsWriter] = None,
    replay_writer: Optional[ReplayLogWriter] = None,
    run_dir: str | None = None,
    buf: Optional[Any] = None,
    **kwargs,
):
    if run_dir is not None:
        init_logging(run_dir=Path(run_dir))

    sim = Simulator()
    # Replay writer is added first so it can complete the replay_url for stats tracker
    if replay_writer is not None:
        sim.add_event_handler(replay_writer)
    stats_writer = stats_writer or NoopStatsWriter()
    sim.add_event_handler(StatsTracker(stats_writer))

    env = MettaGridPufferEnv(sim, curriculum.get_task().get_env_cfg(), buf)
    env = CurriculumEnv(env, curriculum)

    return env


@validate_call(config={"arbitrary_types_allowed": True})
def make_vecenv(
    curriculum: Curriculum,
    vectorization: str,
    num_envs: int = 1,
    batch_size: int | None = None,
    num_workers: int = 1,
    stats_writer: StatsWriter | None = None,
    replay_writer: ReplayLogWriter | None = None,
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
        "stats_writer": stats_writer,
        "replay_writer": replay_writer,
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
