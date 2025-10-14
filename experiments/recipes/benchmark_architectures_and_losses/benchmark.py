"""Benchmark controller for 2-axis architecture testing.

This module provides:
- A 2-axis grid scheduler (reward shaping × task complexity)
- Adaptive controller with store/dispatcher
- Convenience function to run the benchmark sweep
- Shared architecture configurations

Design
------
The benchmark tests architectures across two independent axes:

1. Reward Shaping (vertical axis):
   - dense: High intermediate rewards (0.5-0.9)
   - moderate: Medium intermediate rewards (0.2-0.5)
   - sparse: Minimal intermediate rewards (0.01-0.05)
   - terminal_only: Only heart reward (all intermediates=0.0)
   - adaptive: Learning progress-guided curriculum

2. Task Complexity (horizontal axis):
   - easy: 1:1 converter ratio, initial resources in buildings
   - medium: 2:1 converter ratio, no initial resources
   - hard: 3:1 converter ratio (default), no initial resources

Standardized across all recipes:
   - Map size: 20×20
   - Num agents: 20
   - Combat: Enabled

This design cleanly separates:
- Credit assignment & exploration (reward shaping axis)
- Resource chain complexity & planning (task complexity axis)
- Interaction effects between the two

Notes
-----
- Training commands reference the recipe modules in this folder.
- Evaluation uses the latest available checkpoint via the ":latest" selector.
- Grid cells can be selectively enabled/disabled for focused experiments.

"""

from dataclasses import dataclass, field
from typing import NamedTuple

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.adaptive.dispatcher.local import LocalDispatcher
from metta.adaptive.dispatcher.skypilot import SkypilotDispatcher
from metta.adaptive.models import JobDefinition, JobStatus, RunInfo
from metta.adaptive.protocols import Dispatcher, ExperimentScheduler
from metta.adaptive.stores.wandb import WandbStore
from metta.adaptive.utils import create_eval_job, create_training_job
from metta.agent.policies.agalite import AGaLiTeConfig
from metta.agent.policies.drama_policy import DramaPolicyConfig
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.fast_dynamics import FastDynamicsConfig
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from metta.agent.policies.gtrxl import gtrxl_policy_config
from metta.agent.policies.mamba_sliding import MambaSlidingConfig
from metta.agent.policies.memory_free import MemoryFreeConfig
from metta.agent.policies.puffer import PufferPolicyConfig
from metta.agent.policies.transformer import TransformerPolicyConfig
from metta.agent.policies.trxl import trxl_policy_config
from metta.agent.policies.trxl_nvidia import trxl_nvidia_policy_config
from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policies.vit_reset import ViTResetConfig
from metta.agent.policies.vit_sliding_trans import ViTSlidingTransConfig
from metta.common.util.constants import PROD_STATS_SERVER_URI

# Architecture configurations for benchmark testing
ARCHITECTURES = {
    "vit": ViTDefaultConfig(),
    "vit_sliding": ViTSlidingTransConfig(),
    "vit_reset": ViTResetConfig(),
    "transformer": TransformerPolicyConfig(),
    "fast": FastConfig(),
    "fast_lstm_reset": FastLSTMResetConfig(),
    "fast_dynamics": FastDynamicsConfig(),
    "memory_free": MemoryFreeConfig(),
    "agalite": AGaLiTeConfig(),
    "drama": DramaPolicyConfig(),
    "mamba_sliding": MambaSlidingConfig(),
    "gtrxl": gtrxl_policy_config(),
    "trxl": trxl_policy_config(),
    "trxl_nvidia": trxl_nvidia_policy_config(),
    "puffer": PufferPolicyConfig(),
}


class GridCell(NamedTuple):
    """Represents a single cell in the 2-axis benchmark grid."""

    reward_shaping: str  # dense, moderate, sparse, terminal_only
    task_complexity: str  # easy, medium, hard
    recipe_module: str  # Python module path
    enabled: bool = True  # Whether to run this cell


# Define the 2-axis grid mapping to recipe modules
# Format: GridCell(reward_shaping, task_complexity, recipe_module, enabled)
#
# Complete Grid Status (all cells now implemented):
#
#                    │  Easy Map  │ Medium Map │  Hard Map  │
#                    │ (15×15,12) │ (20×20,20) │ (25×25,24) │
# ───────────────────┼────────────┼────────────┼────────────┤
# Dense Rewards      │     ✓      │     ✓      │     ✓      │
# Moderate Rewards   │     ✓      │     ✓      │     ✓      │
# Sparse Rewards     │     ✓      │     ✓      │     ✓      │
# Terminal Only      │     ✓      │     ✓      │     ✓      │
# Adaptive Curriculum│     ✓      │     ✓      │     ✓      │
#
DEFAULT_GRID: list[GridCell] = [
    # Row 1: Dense rewards (high intermediate rewards 0.5-0.9)
    GridCell("dense", "easy", "experiments.recipes.benchmark_architectures.dense_easy"),
    GridCell(
        "dense", "medium", "experiments.recipes.benchmark_architectures.dense_medium"
    ),
    GridCell("dense", "hard", "experiments.recipes.benchmark_architectures.dense_hard"),
    # Row 2: Moderate rewards (medium intermediate rewards 0.2-0.7)
    GridCell(
        "moderate", "easy", "experiments.recipes.benchmark_architectures.moderate_easy"
    ),
    GridCell(
        "moderate",
        "medium",
        "experiments.recipes.benchmark_architectures.moderate_medium",
    ),
    GridCell(
        "moderate", "hard", "experiments.recipes.benchmark_architectures.moderate_hard"
    ),
    # Row 3: Sparse rewards (minimal intermediate rewards 0.01-0.05)
    GridCell(
        "sparse", "easy", "experiments.recipes.benchmark_architectures.sparse_easy"
    ),
    GridCell(
        "sparse", "medium", "experiments.recipes.benchmark_architectures.sparse_medium"
    ),
    GridCell(
        "sparse", "hard", "experiments.recipes.benchmark_architectures.sparse_hard"
    ),
    # Row 4: Terminal only (only heart reward, no intermediate rewards)
    GridCell(
        "terminal_only",
        "easy",
        "experiments.recipes.benchmark_architectures.terminal_easy",
    ),
    GridCell(
        "terminal_only",
        "medium",
        "experiments.recipes.benchmark_architectures.terminal_medium",
    ),
    GridCell(
        "terminal_only",
        "hard",
        "experiments.recipes.benchmark_architectures.terminal_hard",
    ),
    # Row 5: Adaptive curriculum (learning progress-guided task selection)
    GridCell(
        "adaptive",
        "easy",
        "experiments.recipes.benchmark_architectures.adaptive_easy",
    ),
    GridCell(
        "adaptive",
        "medium",
        "experiments.recipes.benchmark_architectures.adaptive_medium",
    ),
    GridCell(
        "adaptive",
        "hard",
        "experiments.recipes.benchmark_architectures.adaptive_hard",
    ),
]


@dataclass
class BenchmarkArchSchedulerConfig:
    """Configuration for the 2-axis benchmark architectures scheduler."""

    # Grid cells to test (reward_shaping × task_complexity)
    grid: list[GridCell] = field(default_factory=lambda: DEFAULT_GRID)

    # Architecture types to test
    architecture_types: list[str] = field(
        default_factory=lambda: [str(k) for k in ARCHITECTURES.keys()]
    )

    # Number of random seeds per (grid_cell, architecture) pair
    seeds_per_cell: int = 3

    # Total training timesteps per run
    total_timesteps: int = 2000000000

    # Hardware allocation per job
    gpus: int = 4
    nodes: int = 4

    # CLI entrypoints in each recipe module
    train_entrypoint: str = "train"
    eval_entrypoint: str = "evaluate"


class BenchmarkArchScheduler(ExperimentScheduler):
    """2-axis grid scheduler for benchmark architectures."""

    def __init__(
        self,
        experiment_id: str,
        config: BenchmarkArchSchedulerConfig,
    ) -> None:
        """
        Init scheduler with experiment id and config.
        Filters grid to only enabled cells.
        """
        self.experiment_id = experiment_id
        self.config = config
        self.grid_cells = [cell for cell in config.grid if cell.enabled]

    def schedule(
        self,
        runs: list[RunInfo],
        available_training_slots: int,
    ) -> list[JobDefinition]:
        """Create training jobs first (respecting slots), then eval jobs."""
        jobs: list[JobDefinition] = []
        slots = max(0, available_training_slots)
        run_map = {r.run_id: r for r in runs}

        for cell in self.grid_cells:
            for arch_type in self.config.architecture_types:
                for seed in range(self.config.seeds_per_cell):
                    # New run_id format: experiment.arch.reward.complexity.seed
                    run_id = f"{self.experiment_id}.{arch_type}.{cell.reward_shaping}.{cell.task_complexity}.s{seed:02d}"
                    info = run_map.get(run_id)

                    if info is None:
                        if slots <= 0:
                            continue
                        job = create_training_job(
                            run_id=run_id,
                            experiment_id=self.experiment_id,
                            recipe_module=cell.recipe_module,
                            train_entrypoint=self.config.train_entrypoint,
                            stats_server_uri=PROD_STATS_SERVER_URI,
                            gpus=self.config.gpus,
                            nodes=self.config.nodes,
                            train_overrides={
                                "trainer.total_timesteps": self.config.total_timesteps,
                                "arch_type": arch_type,
                            },
                        )
                        # Updated metadata for 2-axis grid
                        job.metadata["benchmark/arch"] = arch_type
                        job.metadata["benchmark/seed"] = f"{seed:02d}"
                        job.metadata["benchmark/reward_shaping"] = cell.reward_shaping
                        job.metadata["benchmark/task_complexity"] = cell.task_complexity
                        jobs.append(job)
                        slots -= 1
                    elif info.status == JobStatus.TRAINING_DONE_NO_EVAL:
                        jobs.append(
                            create_eval_job(
                                run_id=run_id,
                                experiment_id=self.experiment_id,
                                recipe_module=cell.recipe_module,
                                stats_server_uri=PROD_STATS_SERVER_URI,
                                eval_entrypoint=self.config.eval_entrypoint,
                            ),
                        )

        return jobs

    def is_experiment_complete(self, runs: list[RunInfo]) -> bool:
        """Complete when all planned runs are COMPLETED or FAILED."""
        run_map = {r.run_id: r for r in runs}
        for cell in self.grid_cells:
            for arch_type in self.config.architecture_types:
                for seed in range(self.config.seeds_per_cell):
                    run_id = f"{self.experiment_id}.{arch_type}.{cell.reward_shaping}.{cell.task_complexity}.s{seed:02d}"
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


def create_custom_grid(
    reward_levels: list[str] | None = None,
    complexity_levels: list[str] | None = None,
) -> list[GridCell]:
    """
    Create a custom grid with only specified reward/complexity combinations.

    Args:
        reward_levels: List of reward shaping levels to include
            (dense, moderate, sparse, terminal_only). If None, use all available.
        complexity_levels: List of task complexity levels to include
            (easy, medium, hard). If None, use all available.

    Returns:
        List of enabled GridCells matching the criteria
    """
    if reward_levels is None:
        reward_levels = ["dense", "moderate", "sparse", "terminal_only"]
    if complexity_levels is None:
        complexity_levels = ["easy", "medium", "hard"]

    return [
        cell
        for cell in DEFAULT_GRID
        if cell.reward_shaping in reward_levels
        and cell.task_complexity in complexity_levels
    ]


def run(
    experiment_id: str,
    local: bool = False,
    timesteps: int = 2_000_000_000,
    max_parallel: int = 16,
    seeds_per_cell: int = 3,
    gpus: int = 4,
    nodes: int = 4,
    grid: list[GridCell] | None = None,
    architecture_types: list[str] | None = None,
):
    """
    Run benchmark sweep across 2-axis grid.

    Args:
        experiment_id: Unique identifier for this experiment
        local: If True, run locally; if False, use Skypilot
        timesteps: Total training timesteps per run
        max_parallel: Maximum number of parallel jobs
        seeds_per_cell: Number of random seeds per (grid_cell, architecture) pair
        gpus: Number of GPUs per job
        nodes: Number of nodes per job
        grid: Custom grid cells to test. If None, uses DEFAULT_GRID
        architecture_types: List of architecture types to test. If None, uses all
    """
    make_adaptive_controller(
        experiment_id=experiment_id,
        scheduler_config=BenchmarkArchSchedulerConfig(
            total_timesteps=timesteps,
            seeds_per_cell=seeds_per_cell,
            gpus=gpus,
            nodes=nodes,
            grid=grid or DEFAULT_GRID,
            architecture_types=architecture_types
            or [str(k) for k in ARCHITECTURES.keys()],
        ),
        use_skypilot=not local,
        max_parallel=max_parallel,
    ).run()


if __name__ == "__main__":
    # Example: Full 2-axis grid sweep with all enabled cells
    run(
        experiment_id="benchmark_2axis_sweep",
        local=False,
        timesteps=2_000_000_000,
        max_parallel=16,
        seeds_per_cell=3,
        gpus=4,
        nodes=4,
    )

    # Example: Test only reward shaping axis (holding complexity constant at medium)
    # run(
    #     experiment_id="reward_shaping_sweep",
    #     grid=create_custom_grid(complexity_levels=["medium"]),
    #     timesteps=1_000_000,
    #     seeds_per_cell=3,
    # )

    # Example: Test only task complexity axis (holding rewards constant at moderate)
    # run(
    #     experiment_id="complexity_sweep",
    #     grid=create_custom_grid(reward_levels=["moderate"]),
    #     timesteps=1_000_000,
    #     seeds_per_cell=3,
    # )

    # Example: Test specific architectures only
    # run(
    #     experiment_id="vit_variants_sweep",
    #     architecture_types=["vit", "vit_sliding", "vit_reset"],
    #     timesteps=1_000_000,
    #     seeds_per_cell=3,
    # )
