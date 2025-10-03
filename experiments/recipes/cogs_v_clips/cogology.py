"""
Cogology: Progressive Curriculum Training Recipe

A backchaining curriculum that trains agents through 9 progressive stages:
1. Goal Delivery - Learn to deposit hearts
2. Simple Assembly - Craft hearts from resources
3. Single Resource Foraging - Forage one resource
4. Multi-Resource Foraging - Forage all resources
5. Resource Depletion - Handle depleting resources
6-9. Premade Maps - Apply skills to real terrain

Features:
- Automatic stage progression based on success criteria
- 12 variants per stage (A-L): resource-based and energy-based coordination
- "My chest" concept: per-agent chest ownership and rewards
- Learning progress sampling within stages
- 50% task pool eviction on stage transitions
"""

import random
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from cogames.cogs_vs_clips.scenarios import make_game, make_game_from_map, games
from mettagrid.config.mettagrid_config import (
    AgentRewards,
    MettaGridConfig,
    Field as ConfigField,
)
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.train import TrainTool


# =============================================================================
# Stage Configuration
# =============================================================================


@dataclass
class CogologyStageConfig:
    """Configuration for a single curriculum stage."""

    # Identification
    stage_id: int
    name: str
    description: str

    # Map configuration
    map_type: Literal["generated", "premade"]
    map_names: list[str] | None = None  # For premade maps
    map_size: tuple[int, int] = (20, 20)  # For generated maps
    num_agents: int = 24

    # Objects on map
    num_assemblers: int = 2
    num_chests: int = 4
    num_chargers: int = 0
    num_carbon_extractors: int = 0
    num_oxygen_extractors: int = 0
    num_germanium_extractors: int = 0
    num_silicon_extractors: int = 0
    extractor_max_uses: int | None = None  # None = infinite

    # Initial inventory distribution (agents randomly select from options)
    initial_inventory_options: list[dict[str, int]] = field(default_factory=list)

    # Variants (A-L): A-F resource-based, G-L energy-based
    variants: list[str] = field(default_factory=list)

    # Recipe configuration
    recipe_type: Literal["resources_only", "energy_only", "resources_and_energy"] = (
        "resources_only"
    )
    energy_cost_per_heart: int = 10  # 1/10th of initial 100 energy

    # Success criteria
    success_rate_threshold: float = 0.90
    max_energy_death_rate: float = 0.10
    returns_plateau_threshold: float = 0.05
    returns_plateau_window: int = 100
    min_episodes_before_progression: int = 1000

    # Learning progress config for within-stage sampling
    use_learning_progress: bool = True
    learning_progress_config: dict = field(
        default_factory=lambda: {
            "ema_timescale": 0.1,
            "exploration_bonus": 0.15,
            "max_memory_tasks": 500,
            "progress_smoothing": 0.15,
            "num_active_tasks": 200,
            "rand_task_rate": 0.25,
        }
    )


# =============================================================================
# Task Generator
# =============================================================================


class CogologyTaskGenerator(TaskGenerator):
    """
    Task generator for Cogology curriculum.

    Generates tasks based on stage configuration, sampling from variants
    and applying appropriate reward structures.
    """

    class Config(TaskGeneratorConfig["CogologyTaskGenerator"]):
        stage_config: CogologyStageConfig = ConfigField(...)
        speed_reward_coef: float = ConfigField(default=0.01)
        stochastic_shaping: bool = ConfigField(default=False)

    def __init__(self, config: "CogologyTaskGenerator.Config"):
        super().__init__(config)
        self.stage = config.stage_config
        self.speed_reward_coef = config.speed_reward_coef
        self.stochastic_shaping = config.stochastic_shaping

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        """Generate a task for the current stage."""

        # Select variant (if stage has variants)
        variant = rng.choice(self.stage.variants) if self.stage.variants else "A"

        # Generate map based on type
        if self.stage.map_type == "premade":
            env = self._generate_premade_map(rng)
        else:
            env = self._generate_procedural_map(rng, variant)

        # Set initial inventory
        if self.stage.initial_inventory_options:
            initial_inv = rng.choice(self.stage.initial_inventory_options)
            env.game.agent.initial_inventory.update(initial_inv)

        # Configure rewards
        env.game.agent.rewards = self._build_reward_config(rng)

        # Set task label
        env.label = f"stage_{self.stage.stage_id}_{self.stage.name}_variant_{variant}_task_{task_id}"

        return env

    def _generate_premade_map(self, rng: random.Random) -> MettaGridConfig:
        """Generate task from premade map."""
        map_name = rng.choice(self.stage.map_names)
        return make_game_from_map(map_name, num_agents=self.stage.num_agents)

    def _generate_procedural_map(
        self, rng: random.Random, variant: str
    ) -> MettaGridConfig:
        """Generate procedurally generated map."""
        # TODO: Implement multi-room generation for variants
        # For now, use simple single-room generation
        env = make_game(
            num_cogs=self.stage.num_agents,
            width=self.stage.map_size[0],
            height=self.stage.map_size[1],
            num_assemblers=self.stage.num_assemblers,
            num_chests=self.stage.num_chests,
            num_chargers=self.stage.num_chargers,
            num_carbon_extractors=self.stage.num_carbon_extractors,
            num_oxygen_extractors=self.stage.num_oxygen_extractors,
            num_germanium_extractors=self.stage.num_germanium_extractors,
            num_silicon_extractors=self.stage.num_silicon_extractors,
        )

        # Set extractor depletion if configured
        if self.stage.extractor_max_uses is not None:
            self._set_extractor_depletion(env)

        # Configure recipe based on variant
        self._configure_recipe(env, variant)

        return env

    def _set_extractor_depletion(self, env: MettaGridConfig):
        """Set max_uses on all extractors for depletion stages."""
        extractor_types = [
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        ]
        for extractor_name in extractor_types:
            if extractor_name in env.game.objects:
                env.game.objects[
                    extractor_name
                ].max_uses = self.stage.extractor_max_uses

    def _configure_recipe(self, env: MettaGridConfig, variant: str):
        """Configure recipe based on variant (A-L) and stage."""
        # TODO: Implement variant-specific recipe configuration
        # Variants A-F: resource-based coordination
        # Variants G-L: energy-based coordination
        # Different position requirements per variant
        pass

    def _build_reward_config(self, rng: random.Random) -> AgentRewards:
        """Build reward configuration for current stage."""

        # Base rewards: reward when "my chest" has 3+ hearts
        # TODO: Implement "my chest" stat tracking
        stats_rewards = {
            # "my_chest.has_three_hearts": self.speed_reward_coef,
            "chest.heart.amount": 1.0,  # Temporary fallback
        }

        # Optional stochastic reward shaping (Stages 4-5 only)
        inventory_rewards = {}
        if self.stochastic_shaping and self.stage.stage_id >= 4:
            inventory_rewards = {
                "carbon": rng.uniform(0.0, 0.3 if self.stage.stage_id == 4 else 0.5),
                "oxygen": rng.uniform(0.0, 0.3 if self.stage.stage_id == 4 else 0.5),
                "germanium": rng.uniform(0.0, 0.3 if self.stage.stage_id == 4 else 0.5),
                "silicon": rng.uniform(0.0, 0.3 if self.stage.stage_id == 4 else 0.5),
                "energy": rng.uniform(0.0, 0.1 if self.stage.stage_id == 4 else 0.2),
            }

        return AgentRewards(
            stats=stats_rewards,
            inventory=inventory_rewards,
        )


# =============================================================================
# Success Tracking
# =============================================================================


class CogologySuccessTracker:
    """
    Tracks success metrics and determines stage progression.

    Monitors per-chest success rates, energy deaths, and returns plateau
    to decide when agents are ready to advance to the next stage.
    """

    def __init__(self, stage_config: CogologyStageConfig):
        self.config = stage_config
        self.episode_count = 0
        self.chests_with_success_count = 0
        self.total_chests_count = 0
        self.energy_death_count = 0

        # Returns tracking for plateau detection
        self.returns_history = deque(maxlen=stage_config.returns_plateau_window)
        self.recent_mean_return = 0.0
        self.previous_mean_return = 0.0

    def record_episode(
        self,
        chest_success_counts: dict[str, int],
        died_from_energy: bool,
        episode_return: float,
    ):
        """Record episode results."""
        self.episode_count += 1

        # Track per-chest success (each chest with 3+ hearts counts as success)
        for chest_pos, hearts in chest_success_counts.items():
            self.total_chests_count += 1
            if hearts >= 3:
                self.chests_with_success_count += 1

        if died_from_energy:
            self.energy_death_count += 1

        self.returns_history.append(episode_return)

        # Update return means periodically
        if self.episode_count % 100 == 0:
            self.previous_mean_return = self.recent_mean_return
            self.recent_mean_return = (
                sum(self.returns_history) / len(self.returns_history)
                if self.returns_history
                else 0.0
            )

    @property
    def success_rate(self) -> float:
        """Fraction of 'my chests' that achieved 3+ hearts."""
        if self.total_chests_count == 0:
            return 0.0
        return self.chests_with_success_count / self.total_chests_count

    @property
    def energy_death_rate(self) -> float:
        """Fraction of episodes with energy deaths."""
        if self.episode_count == 0:
            return 0.0
        return self.energy_death_count / self.episode_count

    @property
    def returns_plateaued(self) -> bool:
        """Check if returns have plateaued."""
        if len(self.returns_history) < self.config.returns_plateau_window:
            return False

        if self.previous_mean_return == 0:
            return False

        improvement = (self.recent_mean_return - self.previous_mean_return) / abs(
            self.previous_mean_return
        )
        return improvement < self.config.returns_plateau_threshold

    def should_progress(self) -> bool:
        """Check if all criteria met for stage progression."""
        if self.episode_count < self.config.min_episodes_before_progression:
            return False

        criteria = [
            self.success_rate >= self.config.success_rate_threshold,
            self.energy_death_rate <= self.config.max_energy_death_rate,
            self.returns_plateaued,
        ]

        return all(criteria)

    def get_metrics(self) -> dict:
        """Get current metrics for logging."""
        return {
            "curriculum/stage_id": self.config.stage_id,
            "curriculum/stage_name": self.config.name,
            "curriculum/episode_count": self.episode_count,
            "curriculum/success_rate": self.success_rate,
            "curriculum/energy_death_rate": self.energy_death_rate,
            "curriculum/recent_mean_return": self.recent_mean_return,
            "curriculum/returns_improvement": (
                (self.recent_mean_return - self.previous_mean_return)
                / abs(self.previous_mean_return)
                if self.previous_mean_return != 0
                else 0.0
            ),
            "curriculum/should_progress": self.should_progress(),
        }


# =============================================================================
# Automatic Progression Callback
# =============================================================================


class CogologyProgressionCallback:
    """
    Callback for automatic stage progression.

    Monitors success tracker and advances to next stage when criteria are met.
    Handles checkpoint saving, task pool eviction, and stage transitions.
    """

    def __init__(
        self,
        stages: list[CogologyStageConfig],
        current_stage_idx: int,
        task_generator: CogologyTaskGenerator,
        success_tracker: CogologySuccessTracker,
    ):
        self.stages = stages
        self.current_stage_idx = current_stage_idx
        self.task_generator = task_generator
        self.success_tracker = success_tracker

    def on_epoch_end(self, trainer, epoch: int):
        """Check progression criteria after each epoch."""
        # TODO: Implement epoch end callback
        # Check if progression criteria met
        # If yes, call _advance_stage
        pass

    def _advance_stage(self, trainer):
        """Advance to next stage."""
        # TODO: Implement stage advancement
        # 1. Save checkpoint with stage marker
        # 2. Log stage transition to WandB/console
        # 3. Evict 50% of task pool
        # 4. Update task generator to new stage config
        # 5. Reset success tracker for new stage
        # 6. Increment current_stage_idx
        pass


# =============================================================================
# Stage Definitions
# =============================================================================


def _create_stage_configs() -> list[CogologyStageConfig]:
    """Create all 9 stage configurations."""

    stages = [
        # Stage 1: Goal Delivery
        CogologyStageConfig(
            stage_id=1,
            name="goal_delivery",
            description="Learn to deposit 3+ hearts from inventory into your chest",
            map_type="generated",
            map_size=(10, 10),
            num_agents=24,
            num_assemblers=0,
            num_chests=4,
            initial_inventory_options=[
                {"heart": 3},
                {"heart": 4},
                {"heart": 5},
            ],
            variants=[],  # No variants for Stage 1
            success_rate_threshold=0.95,
        ),
        # Stage 2: Simple Assembly
        CogologyStageConfig(
            stage_id=2,
            name="simple_assembly",
            description="Craft hearts from resources and deposit",
            map_type="generated",
            map_size=(15, 15),
            num_agents=24,
            num_assemblers=2,
            num_chests=4,
            initial_inventory_options=[
                {"carbon": 3, "oxygen": 3, "germanium": 3, "silicon": 3},
                {"carbon": 4, "oxygen": 4, "germanium": 4, "silicon": 4},
                {"carbon": 5, "oxygen": 5, "germanium": 5, "silicon": 5},
            ],
            variants=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
            success_rate_threshold=0.90,
        ),
        # Stage 3: Single Resource Foraging
        CogologyStageConfig(
            stage_id=3,
            name="single_resource_foraging",
            description="Forage one abundant resource, craft, deposit",
            map_type="generated",
            map_size=(20, 20),
            num_agents=24,
            num_assemblers=2,
            num_chests=4,
            num_carbon_extractors=8,
            extractor_max_uses=None,  # Infinite
            initial_inventory_options=[
                {"oxygen": 3, "germanium": 3, "silicon": 3},
                {"oxygen": 4, "germanium": 4, "silicon": 4},
                {"oxygen": 5, "germanium": 5, "silicon": 5},
            ],
            variants=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
            success_rate_threshold=0.85,
        ),
        # Stage 4: Multi-Resource Foraging (Abundant)
        CogologyStageConfig(
            stage_id=4,
            name="multi_resource_abundant",
            description="Forage all resources from abundant sources",
            map_type="generated",
            map_size=(25, 25),
            num_agents=24,
            num_assemblers=2,
            num_chests=4,
            num_chargers=4,
            num_carbon_extractors=6,
            num_oxygen_extractors=6,
            num_germanium_extractors=6,
            num_silicon_extractors=6,
            extractor_max_uses=None,  # Infinite
            initial_inventory_options=[{}],  # Empty
            variants=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
            success_rate_threshold=0.85,
            max_energy_death_rate=0.10,
        ),
        # Stage 5: Resource Depletion
        CogologyStageConfig(
            stage_id=5,
            name="resource_depletion",
            description="Handle depleting resources",
            map_type="generated",
            map_size=(30, 30),
            num_agents=24,
            num_assemblers=3,
            num_chests=4,
            num_chargers=6,
            num_carbon_extractors=8,
            num_oxygen_extractors=8,
            num_germanium_extractors=8,
            num_silicon_extractors=8,
            extractor_max_uses=15,  # Depletable
            initial_inventory_options=[{}],
            variants=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
            success_rate_threshold=0.80,
        ),
        # Stage 6: Small Premade Maps
        CogologyStageConfig(
            stage_id=6,
            name="small_premade",
            description="Small premade maps with terrain",
            map_type="premade",
            map_names=["training_facility_1", "training_facility_2"],
            num_agents=4,
            success_rate_threshold=0.75,
        ),
        # Stage 7: Medium Premade Maps
        CogologyStageConfig(
            stage_id=7,
            name="medium_premade",
            description="Medium complexity premade maps",
            map_type="premade",
            map_names=["machina_1", "machina_2", "machina_3"],
            num_agents=4,
            success_rate_threshold=0.70,
        ),
        # Stage 8: Large Premade Maps
        CogologyStageConfig(
            stage_id=8,
            name="large_premade",
            description="Large scale coordination",
            map_type="premade",
            map_names=[
                "machina_1_big",
                "machina_2_bigger",
                "machina_3_big",
                "machina_4_bigger",
                "machina_5_big",
                "machina_6_bigger",
                "machina_7_big",
            ],
            num_agents=4,
            success_rate_threshold=0.65,
        ),
        # Stage 9: Advanced Clipped Maps
        CogologyStageConfig(
            stage_id=9,
            name="clipped_advanced",
            description="Advanced mechanics with clipping",
            map_type="premade",
            map_names=["training_facility_6"],
            num_agents=4,
            success_rate_threshold=0.60,
        ),
    ]

    return stages


# =============================================================================
# Training Functions
# =============================================================================


def train(
    stage: str = "all",
    speed_reward_coef: float = 0.01,
    entropy_coef: float = 0.01,
    learning_rate: float = 3e-4,
    stochastic_shaping: bool = False,
    run_name: str | None = None,
) -> TrainTool:
    """
    Train agents with automatic curriculum progression.
    
    Args:
        stage: "all" for automatic progression, or "stage_1" through "stage_9"
        speed_reward_coef: Coefficient for speed-to-3-hearts reward
        entropy_coef: Entropy coefficient for exploration
        learning_rate: Learning rate for optimizer
        stochastic_shaping: Enable stochastic reward shaping (Stages 4-5)
        run_name: Optional run name for tracking
    
    Example:
        # Automatic progression (recommended)
        uv run ./tools/run.py experiments.recipes.cogs_v_clips.cogology.train \\
            run=cogology_full_auto \\
            stage=all \\
            speed_reward_coef=0.01
        
        # Single stage testing
        uv run ./tools/run.py experiments.recipes.cogs_v_clips.cogology.train \\
            run=cogology_stage3 \\
            stage=stage_3
    """

    # Load stage configs
    stages = _create_stage_configs()

    # Determine starting stage
    if stage == "all":
        current_stage = stages[0]
        # TODO: Enable automatic progression with callbacks
        # enable_automatic_progression = True
    else:
        stage_id = int(stage.replace("stage_", "")) - 1
        current_stage = stages[stage_id]
        # enable_automatic_progression = False

    # Create task generator
    task_generator_config = CogologyTaskGenerator.Config(
        stage_config=current_stage,
        speed_reward_coef=speed_reward_coef,
        stochastic_shaping=stochastic_shaping,
    )
    # task_generator = task_generator_config.create()  # TODO: Use for callbacks

    # Set up curriculum with learning progress
    algorithm_config = LearningProgressConfig(**current_stage.learning_progress_config)
    curriculum = CurriculumConfig(
        task_generator=task_generator_config,
        algorithm_config=algorithm_config,
    )

    # Configure trainer
    trainer_cfg = TrainerConfig(
        losses=LossConfig(entropy_coef=entropy_coef),
        learning_rate=learning_rate,
        total_timesteps=10_000_000,
    )

    # Use LSTM policy
    policy_config = FastLSTMResetConfig()

    # Set up evaluator
    evaluator = EvaluatorConfig(
        simulations=make_eval_suite(),
        evaluate_remote=True,
        evaluate_local=True,
        eval_frequency=50,
    )

    # TODO: Add callbacks
    # callbacks = []
    # if enable_automatic_progression:
    #     callbacks.append(CogologyProgressionCallback(...))
    # callbacks.append(CogologySuccessTracker(...))

    run_name = (
        run_name or f"cogology_{stage}_speed{speed_reward_coef}_ent{entropy_coef}"
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=evaluator,
        stats_server_uri="https://api.observatory.softmax-research.net",
        run_name=run_name,
    )


def make_eval_suite() -> list[SimulationConfig]:
    """Create evaluation suite from all premade maps."""
    game_configs = games()
    eval_sims = []

    eval_map_keys = [
        "training_facility_1",
        "training_facility_2",
        "training_facility_3",
        "training_facility_4",
        "training_facility_5",
        "training_facility_6",
        "machina_1",
        "machina_2",
        "machina_3",
        "machina_1_big",
        "machina_2_bigger",
        "machina_3_big",
        "machina_4_bigger",
        "machina_5_big",
        "machina_6_bigger",
        "machina_7_big",
    ]

    for map_key in eval_map_keys:
        if map_key in game_configs:
            eval_sims.append(
                SimulationConfig(
                    env=game_configs[map_key],
                    suite="cogology",
                    name=f"eval_{map_key}",
                )
            )

    return eval_sims


def play(
    stage: str = "stage_1",
    checkpoint_uri: str | None = None,
    speed_reward_coef: float = 0.01,
) -> PlayTool:
    """
    Interactive play for a specific stage.

    Args:
        stage: Stage to play ("stage_1" through "stage_9")
        checkpoint_uri: Optional checkpoint to load policy from
        speed_reward_coef: Speed reward coefficient
    """
    stages = _create_stage_configs()
    stage_id = int(stage.replace("stage_", "")) - 1
    current_stage = stages[stage_id]

    task_generator_config = CogologyTaskGenerator.Config(
        stage_config=current_stage,
        speed_reward_coef=speed_reward_coef,
    )
    task_generator = task_generator_config.create()

    # Generate a sample task
    sample_env = task_generator.get_task(0)

    return PlayTool(
        sim=SimulationConfig(
            env=sample_env,
            suite="cogology",
            name=f"play_{stage}",
        ),
        policy_uri=checkpoint_uri,
    )


def replay(
    checkpoint_uri: str,
    eval_map: str = "training_facility_1",
    num_episodes: int = 10,
):
    """
    Generate replays on evaluation maps.

    Args:
        checkpoint_uri: Path to checkpoint (file:// or wandb://)
        eval_map: Which evaluation map to replay
        num_episodes: Number of episodes to replay
    """
    # TODO: Implement replay function
    # Need to import ReplayTool
    game_configs = games()
    if eval_map not in game_configs:
        raise ValueError(
            f"Unknown map: {eval_map}. Available: {list(game_configs.keys())}"
        )

    # Return replay tool configuration
    pass


def experiment(
    use_automatic_progression: bool = True,
    stages: list[str] = ["stage_1", "stage_2", "stage_3"],
    speed_reward_coef_default: float = 0.01,
    entropy_coef_default: float = 0.01,
):
    """
    Launch parameter sweep with automatic curriculum progression.

    Sweeps factors of 2 around default values.

    Args:
        use_automatic_progression: If True, train with stage="all"
        stages: Which stages to train (only used if not automatic)
        speed_reward_coef_default: Default speed reward coefficient
        entropy_coef_default: Default entropy coefficient
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d")

    # Factor of 2 sweep
    speed_reward_coefs = [
        speed_reward_coef_default * 0.5,
        speed_reward_coef_default,
        speed_reward_coef_default * 2.0,
    ]

    entropy_coefs = [
        entropy_coef_default * 0.5,
        entropy_coef_default,
        entropy_coef_default * 2.0,
    ]

    stage_to_run = "all" if use_automatic_progression else stages[0]

    for speed_coef in speed_reward_coefs:
        for ent_coef in entropy_coefs:
            run_name = f"cogology_{stage_to_run}_speed{speed_coef:.4f}_ent{ent_coef:.4f}_{timestamp}"

            cmd = [
                "./devops/skypilot/launch.py",
                "experiments.recipes.cogs_v_clips.cogology.train",
                f"run={run_name}",
                f"stage={stage_to_run}",
                f"speed_reward_coef={speed_coef}",
                f"entropy_coef={ent_coef}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
            ]

            print(f"Launching: {' '.join(cmd)}")
            subprocess.run(cmd)
            time.sleep(2)


if __name__ == "__main__":
    experiment()
