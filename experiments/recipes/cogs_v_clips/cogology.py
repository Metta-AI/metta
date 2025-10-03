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

from cogames.cogs_vs_clips.scenarios import (
    _base_game_config,
    games,
    make_game_from_map,
)
from mettagrid.config.mettagrid_config import (
    AgentConfig,
    AgentRewards,
    AssemblerConfig,
    ChestConfig,
    MettaGridConfig,
    RecipeConfig,
    Field as ConfigField,
)
from metta.agent.policies.vit_reset import ViTResetConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.rl.loss import LossConfig
from metta.rl.training.component import TrainerComponent
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
            "progress_smoothing": 0.01,
            "num_active_tasks": 200,
            "rand_task_rate": 0.25,
            "memory": 25,  # Number of samples to remember per task
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

        # Configure default rewards (will be overridden per-agent)
        env.game.agent.rewards = self._build_reward_config(rng)

        # Configure per-agent rewards (map each agent to their room's chest)
        agents_per_room = self._get_agents_per_room(variant)
        self._configure_per_agent_rewards(env, variant, agents_per_room, rng)

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
        """Generate procedurally generated map with multi-room architecture.

        Uses MapGen + RoomGrid to create isolated rooms with agents.
        Each room gets its own chest with unique chest_id for per-agent rewards.
        """
        # Determine room layout based on variant
        agents_per_room = self._get_agents_per_room(variant)
        num_rooms = self.stage.num_agents // agents_per_room

        # Create room template with agents
        room_template = self._create_room_template(agents_per_room)

        # Use MapGen to create the multi-room grid
        from mettagrid.mapgen.mapgen import MapGen
        from mettagrid.mapgen.scenes.inline_ascii import InlineAscii

        mapgen = MapGen.Config(
            instance=InlineAscii.Config(data=room_template),
            num_agents=self.stage.num_agents,  # Total agents across all rooms
            border_width=5,  # Outer border
            instance_border_width=5,  # Wide borders between rooms for isolation
        ).create()
        level = mapgen.build()

        # Convert to MettaGridConfig
        env = self._create_env_from_map(level, num_rooms)

        # Set unique chest_id for each chest (room-based)
        self._configure_chest_ids(env, num_rooms)

        # Configure resource limits per stage
        self._configure_resource_limits(env)

        # Set extractor depletion if configured
        if self.stage.extractor_max_uses is not None:
            self._set_extractor_depletion(env)

        # Configure recipe based on variant
        self._configure_recipe(env, variant)

        return env

    def _create_room_template(self, agents_per_room: int) -> str:
        """Create ASCII template for a single room with agents.

        Args:
            agents_per_room: Number of agents to place in the room

        Returns:
            ASCII string with agents marked as '@'
        """
        import random as stdlib_random

        # 50% chance of small room (5x5) vs medium room (7x7)
        use_small = stdlib_random.random() < 0.5

        # Create a simple room with agents spread out
        if agents_per_room == 1:
            # Small: just agent, Medium: agent with padding
            return "@" if use_small else ".@."
        elif agents_per_room == 2:
            # Small: tight spacing, Medium: more space
            return "@\n@" if use_small else ".@.\n.@."
        elif agents_per_room == 3:
            # Small: 2x2 grid, Medium: spread out
            return "@.@\n@" if use_small else "@.@\n.@."
        elif agents_per_room == 4:
            # Small: 2x2 tight, Medium: 2x2 with space
            return "@.@\n@.@" if use_small else "@...@\n.....\n@...@"
        else:
            # Fallback: single agent
            return "@"

    def _create_env_from_map(self, level, num_rooms: int) -> MettaGridConfig:
        """Create MettaGridConfig from MapGen level with random object placement.

        Uses RandomMapBuilder-style placement: shuffles all objects and places them
        randomly in empty spaces within each room.

        Args:
            level: GameMap from MapGen.build()
            num_rooms: Number of rooms in the map

        Returns:
            MettaGridConfig with map_builder and randomized object placement
        """
        from mettagrid.map_builder.ascii import AsciiMapBuilder

        # Create base config
        cfg = _base_game_config(self.stage.num_agents)

        # Convert object names back to ASCII characters
        name_to_char = {
            "wall": "#",
            "empty": ".",
            "agent.agent": "@",
            "agent.prey": "p",
            "agent.predator": "P",
            "chest": "C",
            "assembler": "Z",
            "converter": "c",
            "altar": "_",
        }

        # Convert grid to ASCII characters
        char_grid = []
        for row in level.grid:
            char_row = [name_to_char.get(cell, ".") for cell in row]
            char_grid.append(char_row)

        # Randomize object placement within each room
        self._randomize_object_placement(char_grid, num_rooms)

        cfg.game.map_builder = AsciiMapBuilder.Config(map_data=char_grid)

        return cfg

    def _randomize_object_placement(self, char_grid: list[list[str]], num_rooms: int):
        """Randomly place objects in empty spaces of the map.

        Mimics RandomMapBuilder's approach: creates list of all objects,
        shuffles them, and places randomly in available empty spaces.

        Args:
            char_grid: The character grid to modify
            num_rooms: Number of rooms (for chest placement)
        """
        import random as stdlib_random

        # Use task-specific random seed for varied placements
        rng = stdlib_random.Random()

        # Collect all empty positions
        empty_positions = []
        for y in range(len(char_grid)):
            for x in range(len(char_grid[y])):
                if char_grid[y][x] == ".":
                    empty_positions.append((y, x))

        # Calculate total objects to place
        total_objects = (
            num_rooms  # chests (1 per room)
            + self.stage.num_assemblers
            + self.stage.num_chargers
            + self.stage.num_carbon_extractors
            + self.stage.num_oxygen_extractors
            + self.stage.num_germanium_extractors
            + self.stage.num_silicon_extractors
        )

        # Ensure we don't try to place more objects than we have space
        if total_objects > len(empty_positions):
            # Scale down proportionally
            scale = len(empty_positions) / total_objects
            num_rooms = max(1, int(num_rooms * scale))
            total_objects = min(total_objects, len(empty_positions))

        # Create object list (similar to RandomMapBuilder)
        objects = []
        objects.extend(["C"] * num_rooms)  # Chests
        objects.extend(["Z"] * self.stage.num_assemblers)  # Assemblers

        # Add chargers if stage has them
        if self.stage.num_chargers > 0:
            objects.extend(
                ["H"]
                * min(self.stage.num_chargers, len(empty_positions) - len(objects))
            )

        # Add extractors (using official map_char from stations.py)
        if self.stage.num_carbon_extractors > 0:
            objects.extend(
                ["N"]  # carbon_extractor map_char
                * min(
                    self.stage.num_carbon_extractors,
                    len(empty_positions) - len(objects),
                )
            )
        if self.stage.num_oxygen_extractors > 0:
            objects.extend(
                ["O"]  # oxygen_extractor map_char
                * min(
                    self.stage.num_oxygen_extractors,
                    len(empty_positions) - len(objects),
                )
            )
        if self.stage.num_germanium_extractors > 0:
            objects.extend(
                ["E"]  # germanium_extractor map_char
                * min(
                    self.stage.num_germanium_extractors,
                    len(empty_positions) - len(objects),
                )
            )
        if self.stage.num_silicon_extractors > 0:
            objects.extend(
                ["I"]  # silicon_extractor map_char
                * min(
                    self.stage.num_silicon_extractors,
                    len(empty_positions) - len(objects),
                )
            )

        # Shuffle positions and place objects
        rng.shuffle(empty_positions)

        for i, obj_char in enumerate(objects):
            if i < len(empty_positions):
                y, x = empty_positions[i]
                char_grid[y][x] = obj_char

    def _get_agents_per_room(self, variant: str) -> int:
        """Get number of agents per room based on variant."""
        # Variants A-F (resource-based) and G-L (energy-based):
        # A, G: 1 agent per room
        # B, H: 2 agents per room
        # C, I: 3 agents per room
        # D, J: 4 agents per room
        # E, K: 2 agents per room (position-dependent)
        # F, L: 4 agents per room (position-dependent)
        variant_map = {
            "A": 1,
            "G": 1,
            "B": 2,
            "H": 2,
            "E": 2,
            "K": 2,
            "C": 3,
            "I": 3,
            "D": 4,
            "J": 4,
            "F": 4,
            "L": 4,
        }
        return variant_map.get(variant, 1)

    def _configure_chest_ids(self, env: MettaGridConfig, num_rooms: int):
        """Configure unique chest_id for each chest (enables per-agent rewards)."""
        # Get all chest objects
        chest_objects = [
            (key, obj)
            for key, obj in env.game.objects.items()
            if isinstance(obj, ChestConfig)
        ]

        # Assign unique chest_id to each chest
        for i, (chest_key, chest_config) in enumerate(chest_objects[:num_rooms]):
            chest_config.chest_id = f"room_{i}"

    def _configure_resource_limits(self, env: MettaGridConfig):
        """Configure resource limits based on stage requirements.

        Stage 1 needs higher heart limits (agents start with 3-5 hearts).
        Later stages may need higher resource limits for complex crafting.
        """
        # Stage-specific resource limits
        if self.stage.stage_id == 1:
            # Stage 1: Agents start with 3-5 hearts, need to hold them
            env.game.agent.resource_limits["heart"] = 5
        else:
            env.game.agent.resource_limits["heart"] = 1

        # Ensure energy is sufficient (especially for stages without chargers)
        if self.stage.num_chargers == 0:
            # No chargers: increase energy capacity
            env.game.agent.resource_limits["energy"] = 200

        # Copy to all per-agent configs if they exist
        if env.game.agents:
            for agent_config in env.game.agents:
                agent_config.resource_limits = env.game.agent.resource_limits.copy()

    def _configure_per_agent_rewards(
        self,
        env: MettaGridConfig,
        variant: str,
        agents_per_room: int,
        rng: random.Random,
    ):
        """Configure per-agent rewards so each agent tracks their room's chest.

        Args:
            env: The environment config
            variant: Stage variant (A-L)
            agents_per_room: Number of agents per room
            rng: Random number generator for stochastic rewards
        """
        # Create per-agent configs
        env.game.agents = []
        for agent_id in range(self.stage.num_agents):
            # Determine which room this agent belongs to
            room_id = agent_id // agents_per_room

            # Build reward config for this agent
            stats_rewards = {
                f"chest_room_{room_id}.heart.amount": 1.0,  # Track their room's chest
            }

            # Optional: Speed reward (when their chest has 3+ hearts)
            # TODO: Implement "has_three_hearts" stat in C++
            # stats_rewards[f"chest_room_{room_id}.has_three_hearts"] = self.speed_reward_coef

            # Optional stochastic reward shaping (Stages 4-5 only)
            inventory_rewards = {}
            if self.stochastic_shaping and self.stage.stage_id >= 4:
                inventory_rewards = {
                    "carbon": rng.uniform(
                        0.0, 0.3 if self.stage.stage_id == 4 else 0.5
                    ),
                    "oxygen": rng.uniform(
                        0.0, 0.3 if self.stage.stage_id == 4 else 0.5
                    ),
                    "germanium": rng.uniform(
                        0.0, 0.3 if self.stage.stage_id == 4 else 0.5
                    ),
                    "silicon": rng.uniform(
                        0.0, 0.3 if self.stage.stage_id == 4 else 0.5
                    ),
                    "energy": rng.uniform(
                        0.0, 0.1 if self.stage.stage_id == 4 else 0.2
                    ),
                }

            # Create agent config
            agent_config = AgentConfig(
                rewards=AgentRewards(
                    stats=stats_rewards,
                    inventory=inventory_rewards,
                )
            )

            # Copy other settings from default agent config
            if self.stage.initial_inventory_options:
                agent_config.initial_inventory = rng.choice(
                    self.stage.initial_inventory_options
                )

            env.game.agents.append(agent_config)

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
        """Configure recipe based on variant (A-L) and stage.

        Variants A-F: Resource-based recipes
        Variants G-L: Energy-based recipes
        """
        # Get all assemblers
        assemblers = [
            (key, obj)
            for key, obj in env.game.objects.items()
            if isinstance(obj, AssemblerConfig)
        ]

        if not assemblers:
            return  # No assemblers to configure

        # Determine recipe configuration based on variant
        recipe_positions, recipe_config = self._get_recipe_for_variant(variant)

        # Apply recipe to all assemblers
        for assembler_key, assembler in assemblers:
            assembler.recipes = [(recipe_positions, recipe_config)]

    def _get_recipe_for_variant(self, variant: str) -> tuple[list[str], RecipeConfig]:
        """Get recipe configuration for a specific variant.

        Returns:
            Tuple of (position_requirements, recipe_config)
        """
        # Variants A-F: Resource-based
        if variant in ["A", "B", "C", "D", "E", "F"]:
            recipe = RecipeConfig(
                input_resources={
                    "carbon": 1,
                    "oxygen": 1,
                    "germanium": 1,
                    "silicon": 1,
                },
                output_resources={"heart": 1},
                cooldown=1,
            )
        # Variants G-L: Energy-based
        else:  # variant in ["G", "H", "I", "J", "K", "L"]
            if self.stage.stage_id == 2:
                # Stage 2 energy-only: no physical resources
                recipe = RecipeConfig(
                    input_resources={"energy": 10},
                    output_resources={"heart": 1},
                    cooldown=1,
                )
            else:
                # Stages 3-5: energy + physical resources
                recipe = RecipeConfig(
                    input_resources={
                        "carbon": 1,
                        "oxygen": 1,
                        "germanium": 1,
                        "silicon": 1,
                        "energy": 10,
                    },
                    output_resources={"heart": 1},
                    cooldown=1,
                )

        # Position requirements based on variant
        positions = self._get_position_requirements(variant)

        return (positions, recipe)

    def _get_position_requirements(self, variant: str) -> list[str]:
        """Get position requirements for a variant.

        Returns:
            List of positions like ["Any"], ["Any", "Any"], ["N", "S"], ["N", "S", "E", "W"]
        """
        position_map = {
            # Single agent
            "A": ["Any"],
            "G": ["Any"],
            # Multi-agent any position
            "B": ["Any", "Any"],
            "H": ["Any", "Any"],
            "C": ["Any", "Any", "Any"],
            "I": ["Any", "Any", "Any"],
            "D": ["Any", "Any", "Any", "Any"],
            "J": ["Any", "Any", "Any", "Any"],
            # Position-dependent
            "E": ["N", "S"],
            "K": ["N", "S"],
            "F": ["N", "S", "E", "W"],
            "L": ["N", "S", "E", "W"],
        }
        return position_map.get(variant, ["Any"])

    def _build_reward_config(self, rng: random.Random) -> AgentRewards:
        """Build reward configuration for current stage.

        Note: Per-agent reward configuration is handled separately in the
        task generator. This builds the base reward structure that will be
        customized per-agent to track their specific chest.
        """

        # Base rewards: Per-chest tracking (agent-specific chest_id set elsewhere)
        # Each agent will track "chest_room_{their_room}.heart.amount"
        stats_rewards = {
            "chest.heart.amount": 1.0,  # Will be overridden per-agent
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


class CogologyProgressionCallback(TrainerComponent):
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
        epoch_interval: int = 10,
    ):
        super().__init__(epoch_interval=epoch_interval)
        self.stages = stages
        self.current_stage_idx = current_stage_idx
        self.task_generator = task_generator
        self.success_tracker = success_tracker
        self.progression_checked = False

    def on_step(self, infos: list[dict]):
        """Extract episode-level information for success tracking."""
        # Process info from each agent
        for info in infos:
            # Check for episode completion
            if "episode" in info:
                episode_info = info["episode"]
                episode_return = episode_info.get("r", 0.0)

                # Extract chest success counts from environment
                # For MVP, we'll use a simplified approach
                chest_success_counts = self._extract_chest_counts(info)
                died_from_energy = info.get("died_from_energy", False)

                # Record episode in success tracker
                self.success_tracker.record_episode(
                    chest_success_counts=chest_success_counts,
                    died_from_energy=died_from_energy,
                    episode_return=episode_return,
                )

    def _extract_chest_counts(self, info: dict) -> dict[str, int]:
        """
        Extract chest heart counts from episode info.

        For MVP, we'll use a simplified approach:
        - Check if the episode was successful (return > threshold)
        - Assume all chests for that agent had 3+ hearts

        In future, this should extract actual per-chest counts from env stats.
        """
        # Simple heuristic: if episode return > 2.0, assume 3+ hearts were deposited
        episode_return = info.get("episode", {}).get("r", 0.0)
        agent_id = info.get("agent_id", 0)

        # Map agent to their chest
        chest_id = f"chest_{agent_id}"
        hearts_count = 3 if episode_return > 2.0 else 0

        return {chest_id: hearts_count}

    def on_epoch_end(self, epoch: int):
        """Check progression criteria after each epoch."""
        if self.current_stage_idx >= len(self.stages) - 1:
            # Already at final stage
            return

        # Log current metrics
        metrics = self.success_tracker.get_metrics()

        # Log to console
        if epoch % 50 == 0:
            print(f"\n[Epoch {epoch}] Cogology Progress:")
            print(f"  Stage: {self.success_tracker.config.name}")
            print(f"  Success Rate: {metrics['curriculum/success_rate']:.2%}")
            print(f"  Episodes: {metrics['curriculum/episode_count']}")
            print(f"  Mean Return: {metrics['curriculum/recent_mean_return']:.2f}")
            print(f"  Should Progress: {metrics['curriculum/should_progress']}")

        # Check if progression criteria met
        if self.success_tracker.should_progress() and not self.progression_checked:
            print(f"\n{'=' * 60}")
            print("STAGE PROGRESSION TRIGGERED")
            print(f"{'=' * 60}")
            self._advance_stage()
            self.progression_checked = True

    def _advance_stage(self):
        """Advance to next stage."""
        current_stage = self.stages[self.current_stage_idx]
        self.current_stage_idx += 1
        next_stage = self.stages[self.current_stage_idx]

        print(
            f"\nAdvancing from Stage {current_stage.stage_id} → Stage {next_stage.stage_id}"
        )
        print(f"  From: {current_stage.name}")
        print(f"  To: {next_stage.name}")

        # 1. Save checkpoint with stage marker
        print("  → Saving checkpoint...")
        # TODO: Implement checkpoint saving via trainer context
        # self.context.save_checkpoint(f"stage_{current_stage.stage_id}_complete")

        # 2. Log stage transition
        print("  → Logging stage transition...")
        if hasattr(self.context, "logger"):
            self.context.logger.log(
                {
                    "curriculum/stage_transition": True,
                    "curriculum/from_stage": current_stage.stage_id,
                    "curriculum/to_stage": next_stage.stage_id,
                    "curriculum/from_stage_name": current_stage.name,
                    "curriculum/to_stage_name": next_stage.name,
                }
            )

        # 3. Evict 50% of task pool
        print("  → Evicting 50% of task pool...")
        if hasattr(self.context, "curriculum"):
            num_evicted = self.context.curriculum.evict_proportion(0.5)
            print(f"     Evicted {num_evicted} tasks from pool")
        else:
            print("     Warning: No curriculum context available for eviction")

        # 4. Update task generator to new stage config
        print("  → Updating task generator...")
        self.task_generator.stage = next_stage
        self.task_generator.config.stage_config = next_stage

        # 5. Reset success tracker for new stage
        print("  → Resetting success tracker...")
        self.success_tracker = CogologySuccessTracker(next_stage)
        self.progression_checked = False

        print(f"\n{'=' * 60}")
        print(f"Stage {next_stage.stage_id} ({next_stage.name}) started!")
        print(f"  Target Success Rate: {next_stage.success_rate_threshold:.1%}")
        print(f"  Min Episodes: {next_stage.min_episodes_before_progression}")
        print(f"{'=' * 60}\n")


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
    entropy_coef: float = 0.01,  # PPO default
    learning_rate: float = 3e-4,
    stochastic_shaping: bool = False,
    run_name: str | None = None,
) -> TrainTool:
    """
    Train agents with automatic curriculum progression using ViT + LSTM reset policy.
    
    Args:
        stage: "all" for automatic progression, or "stage_1" through "stage_9"
        speed_reward_coef: Coefficient for speed-to-3-hearts reward
        entropy_coef: Entropy coefficient for exploration (default: 0.01, PPO default)
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
        stage_id = 0
        enable_automatic_progression = True
    else:
        stage_id = int(stage.replace("stage_", "")) - 1
        current_stage = stages[stage_id]
        enable_automatic_progression = False

    # Create task generator
    task_generator_config = CogologyTaskGenerator.Config(
        stage_config=current_stage,
        speed_reward_coef=speed_reward_coef,
        stochastic_shaping=stochastic_shaping,
    )
    task_generator = task_generator_config.create()

    # Set up curriculum with learning progress
    algorithm_config = LearningProgressConfig(**current_stage.learning_progress_config)
    curriculum = CurriculumConfig(
        task_generator=task_generator_config,
        algorithm_config=algorithm_config,
    )

    # Configure trainer
    from metta.rl.loss.ppo import PPOConfig
    from metta.rl.trainer_config import OptimizerConfig

    trainer_cfg = TrainerConfig(
        losses=LossConfig(loss_configs={"ppo": PPOConfig(ent_coef=entropy_coef)}),
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        total_timesteps=10_000_000,
    )

    # Use ViT with LSTM reset policy
    policy_config = ViTResetConfig()

    # Set up evaluator
    evaluator = EvaluatorConfig(
        simulations=make_eval_suite(),
        evaluate_remote=True,
        evaluate_local=True,
        epoch_interval=50,  # Evaluate every 50 epochs
    )

    # Set up callbacks for success tracking and automatic progression
    progression_callbacks = []
    if enable_automatic_progression:
        success_tracker = CogologySuccessTracker(current_stage)
        progression_callback = CogologyProgressionCallback(
            stages=stages,
            current_stage_idx=stage_id,
            task_generator=task_generator,
            success_tracker=success_tracker,
            epoch_interval=10,  # Check for progression every 10 epochs
        )
        progression_callbacks.append(progression_callback)

    final_run_name = (
        run_name or f"cogology_{stage}_speed{speed_reward_coef}_ent{entropy_coef}"
    )

    return TrainTool(
        run=final_run_name,
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=evaluator,
        stats_server_uri="https://api.observatory.softmax-research.net",
        training_components=progression_callbacks,
        group="cogology",
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
            run_name = f"msb_cogology_{stage_to_run}_speed{speed_coef:.4f}_ent{ent_coef:.4f}_{timestamp}"

            cmd = [
                "./devops/skypilot/launch.py",
                "experiments.recipes.cogs_v_clips.cogology.train",
                f"run={run_name}",
                f"stage={stage_to_run}",
                f"speed_reward_coef={speed_coef}",
                f"entropy_coef={ent_coef}",
                "--gpus=1",
                "--heartbeat-timeout=3600",
            ]

            print(f"Launching: {' '.join(cmd)}")
            subprocess.run(cmd)
            time.sleep(2)


if __name__ == "__main__":
    experiment()
