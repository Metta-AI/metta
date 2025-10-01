from typing import List, Optional, Sequence

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.transformer import (
    TransformerBackboneConfig,
    TransformerBackboneVariant,
    TransformerPolicyConfig,
)
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TorchProfilerConfig, TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig
from mettagrid.config import ConverterConfig


_DEFAULT_POLICY_ARCHITECTURE: TransformerPolicyConfig = TransformerPolicyConfig(
    variant=TransformerBackboneVariant.GTRXL,
    transformer=TransformerBackboneConfig(
        variant=TransformerBackboneVariant.GTRXL,
        latent_size=36,
        hidden_size=36,
        num_layers=3,
        n_heads=4,
        d_ff=128,
        max_seq_len=80,
        memory_len=20,
        dropout=0.0,
        attn_dropout=0.0,
        pre_lnorm=True,
        same_length=False,
        clamp_len=-1,
        positional_scale=0.1,
        use_gating=True,
        ext_len=0,
        activation_checkpoint=False,
        use_flash_checkpoint=False,
        allow_tf32=True,
        use_fused_layernorm=False,
    ),
    critic_hidden_dim=288,
    actor_hidden_dim=144,
    action_embedding_dim=13,
)


def default_policy_architecture() -> TransformerPolicyConfig:
    """Return a copy of the tuned GTrXL policy configuration for ABES runs."""

    return _DEFAULT_POLICY_ARCHITECTURE.model_copy(deep=True)


def make_mettagrid(num_agents: int = 24) -> MettaGridConfig:
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

    # Easy converter: 1 battery_red to 1 heart (instead of 3 to 1)
    altar = arena_env.game.objects.get("altar")
    if isinstance(altar, ConverterConfig) and hasattr(altar, "input_resources"):
        altar.input_resources["battery_red"] = 1

    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    arena_env = arena_env or make_mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # enable or disable attacks. we use cost instead of 'enabled'
    # to maintain action space consistency.
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # sometimes add initial_items to the buildings
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,  # More slices for arena complexity
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def make_evals(env: Optional[MettaGridConfig] = None) -> List[SimulationConfig]:
    basic_env = env or make_mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena", name="basic", env=basic_env),
        SimulationConfig(suite="arena", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | str | None = None,
) -> TrainTool:
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    eval_simulations = make_evals()
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )

    if policy_architecture is None or isinstance(policy_architecture, str):
        policy_architecture = default_policy_architecture()

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=eval_simulations),
        policy_architecture=policy_architecture,
        torch_profiler=TorchProfilerConfig(),
    )


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(sim=SimulationConfig(suite="arena", env=eval_env, name="eval"))


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(sim=SimulationConfig(suite="arena", env=eval_env, name="eval"))


def evaluate(
    policy_uri: str | None = None,
    simulations: Optional[Sequence[SimulationConfig]] = None,
) -> SimTool:
    simulations = simulations or make_evals()
    policy_uris = [policy_uri] if policy_uri is not None else None

    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
    )


def evaluate_in_sweep(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    """Evaluation function optimized for sweep runs.

    Uses 10 episodes per simulation with a 4-minute time limit to get
    reliable results quickly during hyperparameter sweeps.
    """
    if simulations is None:
        # Create sweep-optimized versions of the standard evaluations
        basic_env = make_mettagrid()
        basic_env.game.actions.attack.consumed_resources["laser"] = 100

        combat_env = basic_env.model_copy()
        combat_env.game.actions.attack.consumed_resources["laser"] = 1

        simulations = [
            SimulationConfig(
                suite="arena",
                name="basic",
                env=basic_env,
                num_episodes=10,  # 10 episodes for statistical reliability
                max_time_s=240,  # 4 minutes max per simulation
            ),
            SimulationConfig(
                suite="arena",
                name="combat",
                env=combat_env,
                num_episodes=10,
                max_time_s=240,
            ),
        ]

    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
    )
