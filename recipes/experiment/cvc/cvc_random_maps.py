"""Random map curriculum for CoGs vs Clips.

Uses procedural random map generation with mettagrid.mapgen.scenes.random.Random.Config,
with curriculum bucketing over map dimensions and object counts.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from typing import Optional, Sequence

import metta.cogworks.curriculum as cc
from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import Mission, NumCogsVariant
from cogames.cogs_vs_clips.sites import HELLO_WORLD
from cogames.cogs_vs_clips.variants import TrainingVariant
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import Span
from metta.rl.loss.losses import LossesConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool
from mettagrid.config import vibes as vibes_module
from mettagrid.config.mettagrid_config import ProtocolConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.random import Random
from recipes.experiment import cogs_v_clips


def _make_assembler_protocols(
    vibes_required: list[str],
    first_heart_cost: int = 10,
    additional_heart_cost: int = 5,
) -> list[ProtocolConfig]:
    """Create assembler protocols with configurable vibe requirements.

    Args:
        vibes_required: List of vibes that activate the heart-making protocol.
                       E.g., ["heart_a"] means only heart_a works,
                       ["default", "heart_a"] means either works.
        first_heart_cost: Base resource cost for 1 heart
        additional_heart_cost: Additional cost per extra heart
    """
    gear = [
        ("carbon", "decoder"),
        ("oxygen", "modulator"),
        ("germanium", "scrambler"),
        ("silicon", "resonator"),
    ]

    # Heart-making protocols (1-4 hearts)
    heart_protocols = [
        ProtocolConfig(
            vibes=vibes_required * (i + 1),  # Need more vibes for more hearts
            input_resources={
                "carbon": first_heart_cost + additional_heart_cost * i,
                "oxygen": first_heart_cost + additional_heart_cost * i,
                "germanium": max(1, (first_heart_cost + additional_heart_cost * i) // 5),
                "silicon": 3 * (first_heart_cost + additional_heart_cost * i),
            },
            output_resources={"heart": i + 1},
        )
        for i in range(4)
    ]

    # Gear-making protocols
    gear_protocols = [
        ProtocolConfig(
            vibes=["gear", f"{g[0]}_a"],
            input_resources={g[0]: 1},
            output_resources={g[1]: 1},
        )
        for g in gear
    ]

    return heart_protocols + gear_protocols


def make_random_maps_curriculum(
    num_cogs: int = 20,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
    variants: Optional[Sequence[str]] = None,
    heart_buckets: bool = False,
    resource_buckets: bool = False,
    initial_inventory_buckets: bool = False,
) -> CurriculumConfig:
    """Create a curriculum with randomly generated maps.

    Args:
        num_cogs: Number of agents per environment
        enable_detailed_slice_logging: Enable detailed logging for curriculum slices
        algorithm_config: Optional curriculum algorithm configuration
        variants: Optional mission variants to apply
        heart_buckets: Enable bucketing over heart inventory rewards
        resource_buckets: Enable bucketing over resource stat rewards
        initial_inventory_buckets: Enable bucketing over agent's initial inventory

    Returns:
        A CurriculumConfig with learning progress algorithm across map sizes and object counts
    """
    # Create base mission with random map generation
    mission = Mission(
        name="random_maps_training",
        description="Random procedural maps with varying dimensions and object counts",
        site=HELLO_WORLD,
        num_cogs=num_cogs,
        variants=[TrainingVariant()],  # Apply training variant for better learnability
    )

    # Create base environment
    base_env = mission.make_env()

    # Use full vibe set to maintain a single action space across train/eval.
    from mettagrid.config import vibes as vibes_module

    base_env.game.vibe_names = [v.name for v in vibes_module.VIBES]
    if base_env.game.actions.change_vibe:
        base_env.game.actions.change_vibe.vibes = list(vibes_module.VIBES)

    # Replace map builder with random map generator
    # Random.Config has too_many_is_ok=True, so it will cap objects to available space
    base_env.game.map_builder = MapGen.Config(
        width=80,  # Default, will be bucketed
        height=80,  # Default, will be bucketed
        border_width=5,
        instance=Random.Config(
            agents=num_cogs,
            objects={
                "assembler": 10,  # Default, will be bucketed
                "charger": 5,  # Default, will be bucketed
                "chest": 2,  # Default, will be bucketed
                "carbon_extractor": 5,  # Default, will be bucketed
                "oxygen_extractor": 5,  # Default, will be bucketed
                "germanium_extractor": 5,  # Default, will be bucketed
                "silicon_extractor": 5,  # Default, will be bucketed
            },
            too_many_is_ok=True,  # Automatically caps to available space
        ),
    )

    # Create bucketed tasks
    tasks = cc.bucketed(base_env)

    # Bucket over map dimensions (20x20 to 150x150)
    tasks.add_bucket("game.map_builder.width", [Span(30, 100)])
    tasks.add_bucket("game.map_builder.height", [Span(30, 100)])

    # Bucket over object counts (sparse to dense)
    # Using wide ranges that scale from small maps to large maps
    # too_many_is_ok=True ensures we don't error on small maps
    tasks.add_bucket("game.map_builder.instance.objects.assembler", [Span(10, 30)])
    tasks.add_bucket("game.map_builder.instance.objects.charger", [Span(10, 50)])
    tasks.add_bucket("game.map_builder.instance.objects.chest", [Span(10, 30)])
    tasks.add_bucket("game.map_builder.instance.objects.carbon_extractor", [Span(10, 50)])
    tasks.add_bucket("game.map_builder.instance.objects.oxygen_extractor", [Span(10, 50)])
    tasks.add_bucket("game.map_builder.instance.objects.germanium_extractor", [Span(10, 40)])
    tasks.add_bucket("game.map_builder.instance.objects.silicon_extractor", [Span(10, 50)])
    # Bucket over extractor max_uses (resource scarcity)
    # 0 = unlimited, higher = limited resource
    tasks.add_bucket("game.objects.carbon_extractor.max_uses", [1, 3, 8, 10, 20])
    tasks.add_bucket("game.objects.oxygen_extractor.max_uses", [1, 3, 8, 10, 20])
    tasks.add_bucket("game.objects.germanium_extractor.max_uses", [1, 3, 8, 10, 20])
    tasks.add_bucket("game.objects.silicon_extractor.max_uses", [1, 3, 8, 10, 20])

    # Bucket over entire assembler protocols list (different vibe requirements)
    tasks.add_bucket(
        "game.objects.assembler.protocols",
        [
            # Hard: Only heart_a vibe works
            _make_assembler_protocols(["heart_a"]),
            # Medium: heart_a or heart_b works
            _make_assembler_protocols(["heart_a", "heart_b"]),
            # Easy: default vibe also works
            _make_assembler_protocols(["default", "heart_a", "heart_b"]),
            # All vibes work
            _make_assembler_protocols(
                [
                    "default",
                    "heart_a",
                    "heart_b",
                    "carbon_a",
                    "carbon_b",
                    "oxygen_a",
                    "oxygen_b",
                    "germanium_a",
                    "germanium_b",
                    "silicon_a",
                    "silicon_b",
                ]
            ),
        ],
    )

    # Standard curriculum buckets
    tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500, 2000, 3000, 4000])

    tasks.add_bucket("game.agent.rewards.stats.chest.heart.deposited_by_agent", [3])

    if heart_buckets:
        tasks.add_bucket("game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0])
    if resource_buckets:
        tasks.add_bucket("game.agent.rewards.stats.carbon.gained", [0.0, 0.001])
        tasks.add_bucket("game.agent.rewards.stats.oxygen.gained", [0.0, 0.001])
        tasks.add_bucket("game.agent.rewards.stats.germanium.gained", [0.0, 0.001])
        tasks.add_bucket("game.agent.rewards.stats.silicon.gained", [0.0, 0.001])

        stats_max_cap = 0.5
        tasks.add_bucket("game.agent.rewards.stats_max.carbon.gained", [stats_max_cap])
        tasks.add_bucket("game.agent.rewards.stats_max.oxygen.gained", [stats_max_cap])
        tasks.add_bucket("game.agent.rewards.stats_max.germanium.gained", [stats_max_cap])
        tasks.add_bucket("game.agent.rewards.stats_max.silicon.gained", [stats_max_cap])

    if initial_inventory_buckets:
        # Bucket over agent's starting inventory
        tasks.add_bucket("game.agent.initial_inventory.carbon", [0, 20, 50, 75])
        tasks.add_bucket("game.agent.initial_inventory.oxygen", [0, 20, 50, 75])
        tasks.add_bucket("game.agent.initial_inventory.germanium", [0, 20, 50, 75])
        tasks.add_bucket("game.agent.initial_inventory.silicon", [0, 20, 50, 75])
        tasks.add_bucket("game.agent.initial_inventory.heart", [0, 1, 2])

    # Configure learning progress algorithm
    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=3000,
            max_slice_axes=4,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return tasks.to_curriculum(
        num_active_tasks=2000,
        algorithm_config=algorithm_config,
    )


def make_training_eval_suite(
    num_cogs: int = 20,
    max_evals: Optional[int] = 9,
) -> list[SimulationConfig]:
    """Create evaluation suite with TrainingVariant applied.

    This ensures eval environments have the same action space (18 actions)
    as training environments, allowing policy checkpoints to be loaded.

    Args:
        num_cogs: Number of agents per evaluation
        max_evals: Maximum number of eval simulations (None for all)

    Returns:
        List of SimulationConfig with TrainingVariant applied
    """
    simulations: list[SimulationConfig] = []

    for mission_template in EVAL_MISSIONS:
        # Skip multi-agent only missions for single agent
        if num_cogs == 1 and mission_template.name in {"go_together", "single_use_swarm"}:
            continue

        # Skip distant_resources - causes eval hangs
        if mission_template.name == "distant_resources":
            continue

        # Create mission with TrainingVariant for matching action space
        mission = mission_template.with_variants(
            [
                NumCogsVariant(num_cogs=num_cogs),
                TrainingVariant(),
            ]
        )

        env_cfg = mission.make_env()

        # Apply the same vibe restrictions as training
        env_cfg.game.vibe_names = [v.name for v in vibes_module.VIBES[:13]]
        if env_cfg.game.actions.change_vibe:
            env_cfg.game.actions.change_vibe.vibes = list(vibes_module.VIBES[:13])
        if env_cfg.game.actions.attack:
            env_cfg.game.actions.attack.enabled = False

        sim = SimulationConfig(
            suite="cogs_vs_clips",
            name=f"{mission_template.name}_{num_cogs}cogs",
            env=env_cfg,
        )
        simulations.append(sim)

        if max_evals is not None and len(simulations) >= max_evals:
            break

    return simulations


def train(
    num_cogs: int = 20,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    heart_buckets=False,
    resource_buckets=False,
    initial_inventory_buckets=False,
) -> TrainTool:
    """Create a training tool for random maps curriculum.

    Args:
        num_cogs: Number of agents per environment
        curriculum: Optional curriculum configuration
        enable_detailed_slice_logging: Enable detailed logging
        heart_buckets: Enable bucketing over heart inventory rewards
        resource_buckets: Enable bucketing over resource stat rewards
        initial_inventory_buckets: Enable bucketing over agent's initial inventory

    Returns:
        A TrainTool configured with the random maps curriculum

    Examples:
        Train with 4 agents:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.train num_cogs=4

        Train with 8 agents:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.train num_cogs=8

        Train with initial inventory bucketing:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.train initial_inventory_buckets=True
    """
    resolved_curriculum = curriculum or make_random_maps_curriculum(
        num_cogs=num_cogs,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        heart_buckets=heart_buckets,
        resource_buckets=resource_buckets,
        initial_inventory_buckets=initial_inventory_buckets,
    )

    trainer_cfg = TrainerConfig(
        losses=LossesConfig(),
    )

    # Use custom eval suite with TrainingVariant applied (matching 18-action space)
    eval_suite = make_training_eval_suite(num_cogs=num_cogs)

    evaluator_cfg = EvaluatorConfig(
        simulations=eval_suite,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=resolved_curriculum),
        evaluator=evaluator_cfg,
    )


def evaluate(
    policy_uris: list[str] | str,
    num_cogs: int = 20,
    difficulty: str | None = "standard",
    variants: Optional[Sequence[str]] = None,
) -> EvaluateTool:
    """Evaluate policies on standard CVC missions.

    Args:
        policy_uris: Policy URIs to evaluate
        num_cogs: Number of agents per environment
        difficulty: Difficulty variant
        variants: Optional mission variants

    Returns:
        An EvaluateTool configured for evaluation
    """
    return EvaluateTool(
        simulations=cogs_v_clips.make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            variants=variants,
        ),
        policy_uris=policy_uris,
    )


def play_sparse(
    policy_uri: Optional[str] = None,
    num_cogs: int = 20,
    room_size: int = 80,
) -> PlayTool:
    """Play on a sparse randomly generated map (minimum objects).

    Args:
        policy_uri: Optional policy to use
        num_cogs: Number of agents
        room_size: Map width and height (creates square map)

    Returns:
        A PlayTool configured for sparse map play

    Examples:
        Play small sparse map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_sparse room_size=30

        Play large sparse map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_sparse room_size=150
    """
    mission = Mission(
        name="random_maps_sparse",
        description=f"Sparse random map {room_size}x{room_size}",
        site=HELLO_WORLD,
        num_cogs=num_cogs,
        variants=[TrainingVariant()],
    )

    env = mission.make_env()

    # Use full vibes to keep action space consistent with eval.
    from mettagrid.config import vibes as vibes_module

    env.game.vibe_names = [v.name for v in vibes_module.VIBES]
    if env.game.actions.change_vibe:
        env.game.actions.change_vibe.vibes = list(vibes_module.VIBES)

    env.game.map_builder = MapGen.Config(
        width=room_size,
        height=room_size,
        border_width=5,
        instance=Random.Config(
            agents=num_cogs,
            objects={
                "assembler": 1,
                "charger": 1,
                "chest": 1,
                "carbon_extractor": 2,
                "oxygen_extractor": 2,
                "germanium_extractor": 2,
                "silicon_extractor": 2,
            },
            too_many_is_ok=True,
        ),
    )

    sim = SimulationConfig(
        suite="random_maps",
        name=f"sparse_{room_size}x{room_size}_{num_cogs}cogs",
        env=env,
    )
    return PlayTool(sim=sim, policy_uri=policy_uri)


def play_dense(
    policy_uri: Optional[str] = None,
    num_cogs: int = 20,
    room_size: int = 100,
) -> PlayTool:
    """Play on a dense randomly generated map (maximum objects).

    Args:
        policy_uri: Optional policy to use
        num_cogs: Number of agents
        room_size: Map width and height (creates square map)

    Returns:
        A PlayTool configured for dense map play

    Examples:
        Play small dense map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_dense room_size=30

        Play large dense map:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.play_dense room_size=150
    """
    mission = Mission(
        name="random_maps_dense",
        description=f"Dense random map {room_size}x{room_size}",
        site=HELLO_WORLD,
        num_cogs=num_cogs,
        variants=[TrainingVariant()],
    )

    env = mission.make_env()

    # Restrict vibes to only heart_b and default
    env.game.vibe_names = [v.name for v in vibes_module.VIBES[:13]]
    if env.game.actions.change_vibe:
        env.game.actions.change_vibe.vibes = list(vibes_module.VIBES[:13])

    env.game.map_builder = MapGen.Config(
        width=room_size,
        height=room_size,
        border_width=5,
        instance=Random.Config(
            agents=num_cogs,
            objects={
                "assembler": 50,
                "charger": 50,
                "chest": 50,
                "carbon_extractor": 50,
                "oxygen_extractor": 50,
                "germanium_extractor": 50,
                "silicon_extractor": 50,
            },
            too_many_is_ok=True,  # Will cap to available space on small maps
        ),
    )

    sim = SimulationConfig(
        suite="random_maps",
        name=f"dense_{room_size}x{room_size}_{num_cogs}cogs",
        env=env,
    )
    return PlayTool(sim=sim, policy_uri=policy_uri)


def replay_curriculum(
    policy_uri: str,
    num_cogs: int = 20,
    room_size: int = 80,
    num_assemblers: int = 10,
    num_chargers: int = 10,
    num_chests: int = 10,
    num_extractors: int = 10,
    max_steps: int = 2000,
) -> PlayTool:
    """Replay a trained policy on a random map from the curriculum.

    Args:
        policy_uri: URI to the policy (S3 or local file)
        num_cogs: Number of agents
        room_size: Map width and height (creates square map)
        num_assemblers: Number of assemblers
        num_chargers: Number of chargers
        num_chests: Number of chests
        num_extractors: Number of extractors for each resource type
        max_steps: Maximum episode steps

    Returns:
        A PlayTool configured for curriculum replay

    Examples:
        Replay from S3:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.replay_curriculum \\
                policy_uri=s3://softmax-public/policies/cvc_random_maps_20agent_heartbuckets_True_resourcebuckets_False

        Replay from local checkpoint:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.replay_curriculum \\
                policy_uri=file://./train_dir/my_run/checkpoints/my_run:v12.pt
    """
    mission = Mission(
        name="random_maps_replay",
        description=f"Replay on random map {room_size}x{room_size}",
        site=HELLO_WORLD,
        num_cogs=num_cogs,
        variants=[TrainingVariant()],
    )

    env = mission.make_env()

    # Use full vibes to stay aligned with training/eval action space.
    env.game.vibe_names = [v.name for v in vibes_module.VIBES]
    if env.game.actions.change_vibe:
        env.game.actions.change_vibe.vibes = list(vibes_module.VIBES)

    env.game.map_builder = MapGen.Config(
        width=room_size,
        height=room_size,
        border_width=5,
        instance=Random.Config(
            agents=num_cogs,
            objects={
                "assembler": num_assemblers,
                "charger": num_chargers,
                "chest": num_chests,
                "carbon_extractor": num_extractors,
                "oxygen_extractor": num_extractors,
                "germanium_extractor": num_extractors,
                "silicon_extractor": num_extractors,
            },
            too_many_is_ok=True,
        ),
    )

    env.game.max_steps = max_steps

    sim = SimulationConfig(
        suite="random_maps",
        name=f"replay_{room_size}x{room_size}_{num_cogs}cogs",
        env=env,
    )
    return PlayTool(sim=sim, policy_uri=policy_uri)


def experiment(
    run_name: Optional[str] = None,
    num_cogs: int = 20,
    heartbeat_timeout: int = 3600,
    skip_git_check: bool = True,
    additional_args: Optional[list[str]] = None,
    heart_buckets: bool = False,
    resource_buckets: bool = False,
    initial_inventory_buckets: bool = False,
) -> None:
    """Submit a training job on AWS with 4 GPUs.

    Args:
        run_name: Optional run name
        num_cogs: Number of agents per environment
        heartbeat_timeout: Heartbeat timeout in seconds
        skip_git_check: Whether to skip git check
        additional_args: Additional arguments to pass
        heart_buckets: Enable bucketing over heart inventory rewards
        resource_buckets: Enable bucketing over resource stat rewards
        initial_inventory_buckets: Enable bucketing over agent's initial inventory

    Examples:
        Submit training:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.experiment

        Submit with custom name:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.experiment \\
                run_name=random_maps_4agent

        Submit with initial inventory bucketing:
            uv run ./tools/run.py recipes.experiment.cvc.cvc_random_maps.experiment \\
                initial_inventory_buckets=True
    """
    if run_name is None:
        timestamp = time.strftime("%Y-%m-%d_%H%M%S")
        run_name = (
            f"cvc_random_maps_{num_cogs}agent_heartbuckets_{heart_buckets}_"
            f"resourcebuckets_{resource_buckets}_invbuckets_{initial_inventory_buckets}_{timestamp}"
        )

    cmd = [
        "./devops/skypilot/launch.py",
        "recipes.experiment.cvc.cvc_random_maps.train",
        f"heart_buckets={heart_buckets}",
        f"resource_buckets={resource_buckets}",
        f"initial_inventory_buckets={initial_inventory_buckets}",
        f"run={run_name}-{datetime.now().strftime('%Y-%m-%d_%H%M%S')}",
        f"num_cogs={num_cogs}",
        "--gpus=4",
        "--heartbeat-timeout-seconds=3600",
    ]

    if skip_git_check:
        cmd.append("--skip-git-check")

    if additional_args:
        cmd.extend(additional_args)

    print(f"Launching random maps training job: {run_name}")
    print(f"  Agents: {num_cogs}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    subprocess.run(cmd, check=True)
    print(f"✓ Successfully launched job: {run_name}")


__all__ = [
    "make_random_maps_curriculum",
    "train",
    "evaluate",
    "play_sparse",
    "play_dense",
    "replay_curriculum",
    "experiment",
]

if __name__ == "__main__":
    experiment(heart_buckets=False, resource_buckets=False, initial_inventory_buckets=True)
    experiment(heart_buckets=True, resource_buckets=False, initial_inventory_buckets=True)
    experiment(heart_buckets=True, resource_buckets=True, initial_inventory_buckets=True)
    experiment(heart_buckets=False, resource_buckets=True, initial_inventory_buckets=True)
    experiment(heart_buckets=False, resource_buckets=False)
    experiment(heart_buckets=True, resource_buckets=False)
    experiment(heart_buckets=True, resource_buckets=True)
    experiment(heart_buckets=False, resource_buckets=True)
