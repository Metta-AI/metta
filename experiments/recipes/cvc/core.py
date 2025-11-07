"""Training recipe for Cogs vs Clips eval missions.

This recipe provides training and evaluation on the 13 canonical CoGs vs Clips
evaluation missions, supporting curriculum learning with difficulty variants.
"""

import typing

import cogames
import metta.cogworks.curriculum as cc
import cogames.cogs_vs_clips.evals.eval_missions
import cogames.cogs_vs_clips.mission
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.rl.training
import metta.sim.simulation_config
import metta.tools.eval
import metta.tools.play
import metta.tools.train
import mettagrid.config.mettagrid_config


def make_eval_suite(
    num_cogs: int = 4,
    difficulty: str = "standard",
    subset: typing.Optional[list[str]] = None,
) -> list[metta.sim.simulation_config.SimulationConfig]:
    """Create a suite of eval simulations from CoGames missions.

    Args:
        num_cogs: Number of agents per mission (1, 2, 4, or 8)
        difficulty: Difficulty variant to apply (e.g., "standard", "hard", "story_mode")
        subset: Optional list of mission names to include (defaults to all)

    Returns:
        List of SimulationConfig objects
    """
    # Filter missions if subset specified
    if subset:
        missions = [
            m
            for m in cogames.cogs_vs_clips.evals.eval_missions.EVAL_MISSIONS
            if m.name in subset
        ]
    else:
        missions = cogames.cogs_vs_clips.evals.eval_missions.EVAL_MISSIONS

    simulations = []
    for mission_template in missions:
        # Skip missions that don't make sense for single agent
        if num_cogs == 1 and mission_template.name in [
            "go_together",
            "single_use_swarm",
        ]:
            continue

        # Apply number of agents variant
        mission_with_cogs = mission_template.with_variants(
            [cogames.cogs_vs_clips.mission.NumCogsVariant(num_cogs=num_cogs)]
        )

        # Create env config
        env_cfg = mission_with_cogs.make_env()

        # Create simulation
        sim = metta.sim.simulation_config.SimulationConfig(
            suite="cogs_vs_clips",
            name=f"{mission_template.name}_{num_cogs}cogs",
            env=env_cfg,
        )
        simulations.append(sim)

    return simulations


def make_training_env(
    num_cogs: int = 4,
    mission_name: str = "extractor_hub_30",
) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    """Create a single training environment from a mission.

    Args:
        num_cogs: Number of agents
        mission_name: Name of the mission to use for training

    Returns:
        MettaGridConfig ready for training
    """
    # Find the mission instance
    mission_template = None
    for mission in cogames.cogs_vs_clips.evals.eval_missions.EVAL_MISSIONS:
        if mission.name == mission_name:
            mission_template = mission
            break

    if mission_template is None:
        raise ValueError(f"Mission '{mission_name}' not found in EVAL_MISSIONS")

    mission_with_cogs = mission_template.with_variants(
        [cogames.cogs_vs_clips.mission.NumCogsVariant(num_cogs=num_cogs)]
    )
    return mission_with_cogs.make_env()


def make_curriculum(
    num_cogs: int = 4,
    base_missions: typing.Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumAlgorithmConfig
    ] = None,
) -> metta.cogworks.curriculum.curriculum.CurriculumConfig:
    """Create a curriculum for CoGs vs Clips training.

    This creates a curriculum that varies:
    - Mission type (different maps/layouts)
    - Agent count (1-8 agents)
    - Difficulty parameters (efficiency, energy regen, etc.)

    Args:
        num_cogs: Default number of agents
        base_missions: List of mission names to include in curriculum
                      (defaults to small/medium missions)
        enable_detailed_slice_logging: Enable detailed curriculum logging
        algorithm_config: Curriculum algorithm config (defaults to Learning Progress)

    Returns:
        CurriculumConfig for training
    """
    # Default to missions that are good for training
    if base_missions is None:
        base_missions = [
            "extractor_hub_30",
            "extractor_hub_50",
            "collect_resources_classic",
            "collect_resources_spread",
            "oxygen_bottleneck",
            "energy_starved",
        ]

    # Create separate task sets for each mission type
    # Note: Each mission already has its own difficulty tuning (efficiency, max_uses, etc.)
    # so we don't need to vary those - just vary episode length and reward weight
    all_mission_tasks = []
    for mission_name in base_missions:
        mission_env = make_training_env(num_cogs=num_cogs, mission_name=mission_name)
        mission_tasks = cc.bucketed(mission_env)

        # Vary episode length (missions timeout at different rates)
        mission_tasks.add_bucket("game.max_steps", [750, 1000, 1250, 1500])

        # Vary reward weight (affects learning signal)
        mission_tasks.add_bucket(
            "game.agent.rewards.inventory.heart", [0.1, 0.333, 0.5, 1.0]
        )

        all_mission_tasks.append(mission_tasks)

    # Merge all mission task sets
    merged_tasks = cc.merge(all_mission_tasks)

    # Configure learning progress algorithm
    if algorithm_config is None:
        algorithm_config = metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=2000,  # More tasks due to mission variety
            max_slice_axes=4,  # Multiple dimensions of variation
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return merged_tasks.to_curriculum(
        num_active_tasks=1500,
        algorithm_config=algorithm_config,
    )


def train(
    num_cogs: int = 4,
    curriculum: typing.Optional[
        metta.cogworks.curriculum.curriculum.CurriculumConfig
    ] = None,
    base_missions: typing.Optional[list[str]] = None,
    enable_detailed_slice_logging: bool = False,
) -> metta.tools.train.TrainTool:
    """Create a training tool for CoGs vs Clips.

    Args:
        num_cogs: Number of agents to train with
        curriculum: Optional custom curriculum (defaults to make_curriculum)
        base_missions: Missions to include in training curriculum
        enable_detailed_slice_logging: Enable detailed logging

    Returns:
        TrainTool configured for CoGs vs Clips training
    """
    # Create or use provided curriculum
    resolved_curriculum = curriculum or make_curriculum(
        num_cogs=num_cogs,
        base_missions=base_missions,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    # Create evaluation suite (test on all missions with standard difficulty)
    eval_suite = make_eval_suite(num_cogs=num_cogs, difficulty="standard")

    evaluator_cfg = metta.rl.training.EvaluatorConfig(
        simulations=eval_suite,
    )

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(
            curriculum=resolved_curriculum
        ),
        evaluator=evaluator_cfg,
    )


def train_single_mission(
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
) -> metta.tools.train.TrainTool:
    """Train on a single mission without curriculum.

    Useful for debugging or quick experiments.

    Args:
        mission_name: Name of mission to train on
        num_cogs: Number of agents

    Returns:
        TrainTool for single-mission training
    """
    env = make_training_env(num_cogs=num_cogs, mission_name=mission_name)

    # Create single-env curriculum
    curriculum = cc.env_curriculum(env)

    # Still evaluate on multiple missions
    eval_suite = make_eval_suite(num_cogs=num_cogs, difficulty="standard")

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=metta.rl.training.EvaluatorConfig(simulations=eval_suite),
    )


def evaluate(
    policy_uris: str | typing.Sequence[str] | None = None,
    num_cogs: int = 4,
    difficulty: str = "standard",
    subset: typing.Optional[list[str]] = None,
) -> metta.tools.eval.EvaluateTool:
    """Evaluate policies on CoGs vs Clips missions.

    Args:
        policy_uris: Policy URIs to evaluate
        num_cogs: Number of agents per mission
        difficulty: Difficulty variant to test
        subset: Optional subset of mission names

    Returns:
        EvaluateTool configured for evaluation
    """
    return metta.tools.eval.EvaluateTool(
        simulations=make_eval_suite(
            num_cogs=num_cogs,
            difficulty=difficulty,
            subset=subset,
        ),
        policy_uris=policy_uris,
    )


def play(
    policy_uri: typing.Optional[str] = None,
    mission_name: str = "extractor_hub_30",
    num_cogs: int = 4,
) -> metta.tools.play.PlayTool:
    """Play a single mission with a policy.

    Args:
        policy_uri: URI of policy to load (None for random policy)
        mission_name: Name of mission to play
        num_cogs: Number of agents

    Returns:
        PlayTool configured for interactive play
    """
    env = make_training_env(num_cogs=num_cogs, mission_name=mission_name)
    sim = metta.sim.simulation_config.SimulationConfig(
        suite="cogs_vs_clips",
        name=f"{mission_name}_{num_cogs}cogs",
        env=env,
    )
    return metta.tools.play.PlayTool(sim=sim, policy_uri=policy_uri)


def play_training_env(
    policy_uri: typing.Optional[str] = None,
    num_cogs: int = 4,
) -> metta.tools.play.PlayTool:
    """Play the default training environment.

    Args:
        policy_uri: URI of policy to load
        num_cogs: Number of agents

    Returns:
        PlayTool for the training environment
    """
    return play(
        policy_uri=policy_uri,
        mission_name="extractor_hub_30",
        num_cogs=num_cogs,
    )


# Convenience: Pre-configured training recipes for different scenarios


def train_small_maps(num_cogs: int = 4) -> metta.tools.train.TrainTool:
    """Train on small maps (30x30, classic layouts)."""
    return train(
        num_cogs=num_cogs,
        base_missions=[
            "extractor_hub_30",
            "collect_resources_classic",
            "oxygen_bottleneck",
        ],
    )


def train_medium_maps(num_cogs: int = 4) -> metta.tools.train.TrainTool:
    """Train on medium maps (50x50, spread layouts)."""
    return train(
        num_cogs=num_cogs,
        base_missions=[
            "extractor_hub_50",
            "collect_resources_spread",
            "energy_starved",
        ],
    )


def train_large_maps(num_cogs: int = 8) -> metta.tools.train.TrainTool:
    """Train on large maps with more agents."""
    return train(
        num_cogs=num_cogs,
        base_missions=["extractor_hub_70", "collect_far", "divide_and_conquer"],
    )


def train_coordination(num_cogs: int = 4) -> metta.tools.train.TrainTool:
    """Train on missions emphasizing multi-agent coordination."""
    return train(
        num_cogs=num_cogs,
        base_missions=["go_together", "divide_and_conquer", "collect_resources_spread"],
    )
