import logging
import pathlib
import typing

import pydantic

import metta.cogworks.curriculum
import metta.common.util.log_config
import metta.sim.replay_log_writer
import mettagrid.envs.early_reset_handler
import mettagrid.envs.mettagrid_puffer_env
import mettagrid.envs.stats_tracker
import mettagrid.simulator
import mettagrid.util.stats_writer
import pufferlib
import pufferlib.vector

logger = logging.getLogger("vecenv")


@pydantic.validate_call(config={"arbitrary_types_allowed": True})
def make_env_func(
    curriculum: metta.cogworks.curriculum.Curriculum,
    stats_writer: typing.Optional[mettagrid.util.stats_writer.StatsWriter] = None,
    replay_writer: typing.Optional[metta.sim.replay_log_writer.ReplayLogWriter] = None,
    run_dir: str | None = None,
    buf: typing.Optional[typing.Any] = None,
    **kwargs,
):
    if run_dir is not None:
        metta.common.util.log_config.init_logging(run_dir=pathlib.Path(run_dir))

    sim = mettagrid.simulator.Simulator()
    # Replay writer is added first so it can complete the replay_url for stats tracker
    if replay_writer is not None:
        sim.add_event_handler(replay_writer)
    stats_writer = stats_writer or mettagrid.util.stats_writer.NoopStatsWriter()
    sim.add_event_handler(mettagrid.envs.stats_tracker.StatsTracker(stats_writer))
    sim.add_event_handler(mettagrid.envs.early_reset_handler.EarlyResetHandler())

    env = mettagrid.envs.mettagrid_puffer_env.MettaGridPufferEnv(sim, curriculum.get_task().get_env_cfg(), buf)
    env = metta.cogworks.curriculum.CurriculumEnv(env, curriculum)

    return env


@pydantic.validate_call(config={"arbitrary_types_allowed": True})
def make_vecenv(
    curriculum: metta.cogworks.curriculum.Curriculum,
    vectorization: str,
    num_envs: int = 1,
    batch_size: int | None = None,
    num_workers: int = 1,
    stats_writer: mettagrid.util.stats_writer.StatsWriter | None = None,
    replay_writer: metta.sim.replay_log_writer.ReplayLogWriter | None = None,
    run_dir: str | None = None,
    **kwargs,
) -> typing.Any:  # Returns pufferlib VecEnv instance
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
