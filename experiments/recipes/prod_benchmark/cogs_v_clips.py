"""Cogs-vs-Clips prod benchmark recipe."""

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
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.core import Distribution as D
from metta.sweep.core import SweepParameters as SP
from metta.sweep.core import make_sweep
from metta.tools.eval import EvaluateTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.builder import building
from mettagrid.config import AssemblerConfig, MettaGridConfig

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

    arena_env.game.objects.update(
        {
            "altar": building.assembler_altar,
            "mine_red": building.assembler_mine_red,
            "generator_red": building.assembler_generator_red,
            "lasery": building.assembler_lasery,
            "armory": building.assembler_armory,
        }
    )

    arena_env.game.actions.attack.enabled = False
    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

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
    stacks_env = basic_env.model_copy()

    for sim_env in (basic_env, stacks_env):
        sim_env.game.actions.attack.enabled = False
        sim_env.game.actions.attack.consumed_resources["laser"] = 100

    return [
        SimulationConfig(suite="cvc_arena", name="basic", env=basic_env),
        SimulationConfig(suite="cvc_arena", name="stacks", env=stacks_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    architecture: str = "vit_reset",
    seed: int | None = None,
) -> TrainTool:
    resolved_curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    effective_seed = _resolve_seed(seed)
    return TrainTool(
        system=_build_system(effective_seed),
        trainer=_build_trainer(),
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum, seed=effective_seed),
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=_resolve_policy_architecture(architecture),
    )


def train_shaped(
    rewards: bool = True,
    assemblers: bool = True,
    architecture: str = "vit_reset",
    seed: int | None = None,
) -> TrainTool:
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

    if assemblers:
        # Update altar recipe to require battery_red input
        altar_config = env_cfg.game.objects["altar"]
        assert isinstance(altar_config, AssemblerConfig)
        altar_config.protocols[0].input_resources["battery_red"] = 1

    curriculum = cc.env_curriculum(env_cfg)
    effective_seed = _resolve_seed(seed)

    return TrainTool(
        system=_build_system(effective_seed),
        trainer=_build_trainer(),
        training_env=TrainingEnvironmentConfig(curriculum=curriculum, seed=effective_seed),
        evaluator=EvaluatorConfig(simulations=simulations(env_cfg)),
        policy_architecture=_resolve_policy_architecture(architecture),
    )


def evaluate(
    policy_uris: str | Sequence[str] | None = None,
) -> EvaluateTool:
    return EvaluateTool(
        simulations=simulations(),
        policy_uris=policy_uris,
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
            min=5e9,
            max=6e9,
            search_center=5.5e9,
        ),
    ]

    return make_sweep(
        name=sweep_name,
        recipe="experiments.recipes.prod_benchmark.cogs_v_clips",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        objective="evaluator/eval_cvc_arena/score",
        parameters=parameters,
        num_trials=80,
        num_parallel_trials=4,
    )
