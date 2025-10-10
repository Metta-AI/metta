"""Adaptive controller boilerplate for benchmark architectures.

This module provides:
- A minimal ExperimentScheduler implementation for the benchmark_architectures suite
- A helper to construct an AdaptiveController with store/dispatcher
- A convenience function to run the controller loop

Notes
-----
- Training commands reference the per-level recipe modules in this folder.
- Evaluation uses the latest available checkpoint via the ":latest" selector.

"""

from __future__ import annotations

from dataclasses import dataclass, field

from experiments.recipes.benchmark_architectures.level_1_basic import ARCHITECTURES
from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.adaptive.dispatcher.local import LocalDispatcher
from metta.adaptive.dispatcher.skypilot import SkypilotDispatcher
from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.protocols import Dispatcher, ExperimentScheduler
from metta.adaptive.stores.wandb import WandbStore
from metta.adaptive.utils import create_eval_job, create_training_job
from metta.common.util.constants import PROD_STATS_SERVER_URI


@dataclass
class BenchmarkArchSchedulerConfig:
    """Configuration for the benchmark architectures scheduler."""

    # Which recipe modules/levels to run
    levels: list[str] = field(
        default_factory=lambda: [
            "experiments.recipes.benchmark_architectures.level_1_basic",
            "experiments.recipes.benchmark_architectures.level_2_easy",
            "experiments.recipes.benchmark_architectures.level_3_medium",
            "experiments.recipes.benchmark_architectures.level_4_hard",
            "experiments.recipes.benchmark_architectures.level_5_expert",
        ],
    )

    # Architecture to iterate over:
    architecture_types: list[str] = field(
        default_factory=lambda: [str(k) for k in ARCHITECTURES.keys()]
    )

    seeds_per_level: int = 3

    # CLI entrypoints in each recipe module
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"


class BenchmarkArchScheduler(ExperimentScheduler):
    """Straightforward training → evaluation scheduler for benchmark levels."""

    def __init__(
        self,
        experiment_id: str,
        config: BenchmarkArchSchedulerConfig,
    ) -> None:
        """
        Init scheduler with experiment id and config.
        This is where you want to initialize experiment state in general.
        """
        self.experiment_id = experiment_id
        self.config = config
        self.modules: list[str] = list(config.levels)

    def schedule(
        self,
        runs: list[RunInfo],
        available_training_slots: int,
    ) -> list[JobDefinition]:
        """Create training jobs first (respecting slots), then eval jobs."""
        jobs: list[JobDefinition] = []
        slots = max(0, available_training_slots)
        run_map = {r.run_id: r for r in runs}

        for module in self.modules:
            level = module.rsplit(".", 1)[-1]
            for arch_type in self.config.architecture_types:
                for seed in range(self.config.seeds_per_level):
                    run_id = f"{self.experiment_id}.{arch_type}.{level}.s{seed:02d}"
                    info = run_map.get(run_id)

                    if info is None:
                        if slots <= 0:
                            continue
                        job = create_training_job(
                            run_id=run_id,
                            experiment_id=self.experiment_id,
                            recipe_module=module,
                            train_entrypoint=self.config.train_entrypoint,
                            stats_server_uri=PROD_STATS_SERVER_URI,
                            gpus=4,
                            nodes=4,
                            train_overrides={
                                "trainer.total_timesteps": self.config.total_timesteps,
                                "arch_type": arch_type,
                            },
                        )
                        job.metadata["benchmark/arch"] = arch_type
                        job.metadata["benchmark/seed"] = f"{seed:02d}"
                        job.metadata["benchmark/level"] = level
                        jobs.append(job)
                        slots -= 1
                    elif info.status == JobStatus.TRAINING_DONE_NO_EVAL:
                        jobs.append(
                            create_eval_job(
                                run_id=run_id,
                                experiment_id=self.experiment_id,
                                recipe_module=module,
                                stats_server_uri=PROD_STATS_SERVER_URI,
                                eval_entrypoint=self.config.eval_entrypoint,
                            ),
                        )

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        """Complete when all planned runs are COMPLETED or FAILED."""
        run_map = {r.run_id: r for r in runs}
        for module in self.modules:
            level = module.rsplit(".", 1)[-1]
            for arch_type in self.config.architecture_types:
                for seed in range(self.config.seeds_per_level):
                    run_id = f"{self.experiment_id}.{arch_type}.{level}.s{seed:02d}"
                    info = run_map.get(run_id)
                    if info is None:
                        return False
                    if info.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                        return False
        return True


def make_adaptive_controller(  # noqa: PLR0913
    *,
    experiment_id: str,
    wandb_entity: str = "metta-research",
    wandb_project: str = "metta",
    scheduler_config: BenchmarkArchSchedulerConfig | None = None,
    dispatcher: Dispatcher | None = None,
    use_skypilot: bool = False,
    max_parallel: int = 20,
    monitoring_interval: int = 60,
    resume: bool = False,
) -> AdaptiveController:
    sched = BenchmarkArchScheduler(
        experiment_id=experiment_id,
        config=scheduler_config or BenchmarkArchSchedulerConfig(),
    )

    disp: Dispatcher
    if dispatcher is not None:
        disp = dispatcher
    else:
        disp = (
            SkypilotDispatcher()
            if use_skypilot
            else LocalDispatcher(capture_output=True)
        )

    store = WandbStore(entity=wandb_entity, project=wandb_project)
    cfg = AdaptiveConfig(
        max_parallel=max_parallel,
        monitoring_interval=monitoring_interval,
        resume=resume,
        experiment_tags=["benchmark_architectures", experiment_id],
    )

    return AdaptiveController(
        experiment_id=experiment_id,
        scheduler=sched,
        dispatcher=disp,
        store=store,
        config=cfg,
    )


def run(
    experiment_id: str,
    local: bool = False,
    timesteps: int = 2000000000,
    max_parallel: int = 16,
    seeds_per_level: int = 3,
    gpus: int = 4,
    nodes: int = 4,
):
    make_adaptive_controller(
        experiment_id=experiment_id,
        scheduler_config=BenchmarkArchSchedulerConfig(
            total_timesteps=timesteps,
            seeds_per_level=seeds_per_level,
            gpus=gpus,
            nodes=nodes,
        ),
        use_skypilot=not local,
        max_parallel=max_parallel,
    ).run()


if __name__ == "__main__":
    run(
        experiment_id="benchmark_arch_sweep",
        local=False,
        timesteps=2_000_000_000,
        max_parallel=16,
        seeds_per_level=3,
        gpus=4,
        nodes=4,
    )
