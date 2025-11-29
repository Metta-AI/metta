from pathlib import Path
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.vit import ViTDefaultConfig
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.common.wandb.context import WandbConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import (
    InitialPolicyConfig,
    OptimizerConfig,
    TorchProfilerConfig,
    TrainerConfig,
)
from metta.rl.training import (
    CheckpointerConfig,
    EvaluatorConfig,
    GradientReporterConfig,
    HeartbeatConfig,
    StatsReporterConfig,
    TrainingEnvironmentConfig,
    WandbAborterConfig,
)
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config.mettagrid_config import EnvSupervisorConfig

# TODO(dehydration): make sure this trains as well as main on arena
# it's possible the maps are now different


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)
    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # arena_tasks.add_bucket("game.map_builder.instance.params.agents", [1, 2, 3, 4, 6])
    # arena_tasks.add_bucket("game.map_builder.width", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.height", [10, 20, 30, 40])
    # arena_tasks.add_bucket("game.map_builder.instance_border_width", [0, 6])

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])
    arena_tasks.add_bucket("game.agent.initial_inventory.ore_red", [0, 1, 3])
    arena_tasks.add_bucket("game.agent.initial_inventory.battery_red", [0, 3])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Default: bidirectional learning progress
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena", name="basic", env=basic_env),
        SimulationConfig(suite="arena", name="combat", env=combat_env),
    ]


def _make_baseline(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Create baseline configuration with all configs explicitly set."""
    if curriculum is None:
        curriculum = make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)
    eval_simulations = simulations()

    optimizer_config = OptimizerConfig(
        type="adamw_schedulefree",
        learning_rate=0.00092,
        beta1=0.9,
        beta2=0.999,
        eps=3.186531e-07,
        weight_decay=0.01,
        momentum=0.9,
        warmup_steps=1000,
    )

    trainer_config = TrainerConfig(
        total_timesteps=50_000_000_000,
        optimizer=optimizer_config,
        losses=LossesConfig(),
        require_contiguous_env_ids=False,
        verbose=True,
        batch_size=524288,
        minibatch_size=16384,
        bptt_horizon=64,
        update_epochs=1,
        scale_batches_by_world_size=False,
        compile=False,
        compile_mode="reduce-overhead",
        detect_anomaly=False,
        heartbeat=HeartbeatConfig(epoch_interval=1),
        initial_policy=InitialPolicyConfig(
            uri=None,
            type="top",
            range=1,
            metric="epoch",
            filters={},
        ),
        profiler=TorchProfilerConfig(
            interval_epochs=0,
            profile_dir=None,
        ),
    )

    training_env_config = TrainingEnvironmentConfig(
        curriculum=curriculum,
        num_workers=1,
        async_factor=2,
        auto_workers=True,
        forward_pass_minibatch_target_size=4096,
        zero_copy=True,
        vectorization="multiprocessing",
        seed=0,
        write_replays=False,
        replay_dir=Path("./train_dir/replays/training"),
        supervisor=EnvSupervisorConfig(),
        maps_cache_size=None,
    )

    evaluator_config = EvaluatorConfig(
        epoch_interval=100,
        evaluate_local=True,
        evaluate_remote=True,
        num_training_tasks=2,
        simulations=eval_simulations,
        training_replay_envs=[],
        replay_dir=None,
        skip_git_check=True,
        git_hash=None,
        verbose=False,
        allow_eval_without_stats=False,
    )

    return TrainTool(
        run=None,
        trainer=trainer_config,
        training_env=training_env_config,
        policy_architecture=ViTDefaultConfig(),
        initial_policy_uri=None,
        checkpointer=CheckpointerConfig(epoch_interval=30),
        gradient_reporter=GradientReporterConfig(epoch_interval=0),
        stats_server_uri=None,
        wandb=WandbConfig.Unconfigured(),
        group=None,
        evaluator=evaluator_config,
        torch_profiler=TorchProfilerConfig(interval_epochs=0, profile_dir=None),
        scheduler=None,
        context_checkpointer={},
        stats_reporter=StatsReporterConfig(),
        wandb_aborter=WandbAborterConfig(epoch_interval=5),
        map_preview_uri=None,
        disable_macbook_optimize=False,
        sandbox=False,
    )


BASELINE = _make_baseline()


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    baseline: Optional[TrainTool] = None,
) -> TrainTool:
    if baseline is None:
        baseline = _make_baseline(
            curriculum=curriculum,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )
    return baseline


def train_shaped(rewards: bool = True, baseline: Optional[TrainTool] = None) -> TrainTool:
    if baseline is None:
        baseline = BASELINE.model_copy(deep=True)
    else:
        baseline = baseline.model_copy(deep=True)

    env_cfg = mettagrid()
    env_cfg.game.agent.rewards.inventory["heart"] = 1
    env_cfg.game.agent.rewards.inventory_max["heart"] = 100

    if rewards:
        env_cfg.game.agent.rewards.inventory.update(
            {
                "ore_red": 0.1,
                "battery_red": 0.8,
                "laser": 0.5,
                "armor": 0.5,
                "blueprint": 0.5,
            }
        )
        env_cfg.game.agent.rewards.inventory_max.update(
            {
                "ore_red": 1,
                "battery_red": 1,
                "laser": 1,
                "armor": 1,
                "blueprint": 1,
            }
        )

    baseline.training_env.curriculum = cc.env_curriculum(env_cfg)
    baseline.evaluator.simulations = simulations()

    return baseline


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    return EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris,
    )


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def play(policy_uri: Optional[str] = None) -> PlayTool:
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)
