"""Arena Basic Easy Shaped (ABES) prod benchmark configuration."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.trxl import trxl_policy_config
from metta.agent.policies.vit_reset import ViTResetConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.losses import LossesConfig
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig

BENCHMARK_SEED = 63
BENCHMARK_TIMESTEPS = 5_000_000_000

_ARCHITECTURES: dict[str, Callable[[], PolicyArchitecture]] = {
    "vit_reset": ViTResetConfig,
    "trxl": trxl_policy_config,
}


def _resolve_seed(seed: int | None) -> int:
    return seed if seed is not None else BENCHMARK_SEED


def _resolve_policy_architecture(name: str) -> PolicyArchitecture:
    key = name.lower()
    if key not in _ARCHITECTURES:
        options = ", ".join(sorted(_ARCHITECTURES))
        raise ValueError(f"Unknown architecture '{name}'. Expected one of: {options}")
    return _ARCHITECTURES[key]()


def _build_trainer(total_timesteps: int = BENCHMARK_TIMESTEPS) -> TrainerConfig:
    return TrainerConfig(total_timesteps=total_timesteps, losses=LossesConfig())


def _build_system(seed: int) -> SystemConfig:
    return SystemConfig(seed=seed)


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    arena_env = eb.make_arena(num_agents=num_agents)
    arena_env.game.agent.rewards.inventory = {
        "heart": 1,
        "ore_red": 0.1,
        "battery_red": 0.8,
        "laser": 0.5,
        "armor": 0.5,
        "blueprint": 0.5,
    }
    arena_env.game.agent.rewards.inventory_max = {
        "heart": 100,
        "ore_red": 1,
        "battery_red": 1,
        "laser": 1,
        "armor": 1,
        "blueprint": 1,
    }
    arena_env.game.actions.attack.enabled = False
    arena_env.game.actions.attack.consumed_resources["laser"] = 100
    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or mettagrid()
    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0])
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    arena_tasks.add_bucket("game.map_builder.width", [64, 70, 80])
    arena_tasks.add_bucket("game.map_builder.height", [64, 70, 80])
    if "mine_red" in arena_env.game.objects:
        arena_tasks.add_bucket("game.objects.mine_red.initial_resource_count", [0, 1])
    if "generator_red" in arena_env.game.objects:
        arena_tasks.add_bucket("game.objects.generator_red.initial_resource_count", [0, 1])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    eval_env = env or mettagrid()
    eval_env.game.actions.attack.enabled = False
    eval_env.game.actions.attack.consumed_resources["laser"] = 100
    return [SimulationConfig(suite="arena", name="basic", env=eval_env)]


def train(
    *,
    architecture: str = "vit_reset",
    seed: int | None = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    effective_seed = _resolve_seed(seed)
    curriculum = make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging)
    return TrainTool(
        system=_build_system(effective_seed),
        trainer=_build_trainer(),
        training_env=TrainingEnvironmentConfig(curriculum=curriculum, seed=effective_seed),
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=_resolve_policy_architecture(architecture),
        torch_profiler=TorchProfilerConfig(),
    )


def evaluate(policy_uris: Optional[Sequence[str]] = None) -> EvaluateTool:
    return EvaluateTool(simulations=simulations(), policy_uris=policy_uris or [])


def play(policy_uri: Optional[str] = None) -> PlayTool:
    return PlayTool(sim=simulations()[0], policy_uri=policy_uri)


def replay(policy_uri: Optional[str] = None) -> ReplayTool:
    return ReplayTool(sim=simulations()[0], policy_uri=policy_uri)


def evaluate_in_sweep(policy_uri: str) -> EvaluateTool:
    sweep_env = mettagrid()
    sweep_env.game.actions.attack.enabled = False
    sweep_env.game.actions.attack.consumed_resources["laser"] = 100
    return EvaluateTool(
        simulations=[
            SimulationConfig(
                suite="sweep",
                name="basic",
                env=sweep_env,
                num_episodes=10,
                max_time_s=240,
            )
        ],
        policy_uris=[policy_uri],
    )


def sweep(sweep_name: str) -> SweepTool:
    parameters = [
        SP.LEARNING_RATE,
        SP.PPO_CLIP_COEF,
        SP.PPO_GAE_LAMBDA,
        SP.PPO_VF_COEF,
        SP.ADAM_EPS,
        SP.param(
            "trainer.total_timesteps",
            D.INT_UNIFORM,
            min=4.5e9,
            max=5.5e9,
            search_center=5e9,
        ),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="experiments.recipes.prod_benchmark.abes",
        train_entrypoint="train",
        eval_entrypoint="evaluate_in_sweep",
        objective="evaluator/eval_sweep/score",
        parameters=parameters,
        num_trials=80,
        num_parallel_trials=4,
    )
