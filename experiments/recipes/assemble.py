"""
Assembler coordination environments for multi-agent reinforcement learning.

This module implements 8 different assembler-based coordination tasks where agents must
work together to operate an assembler that requires specific formations and resources.

Environment Types:
1. Resource collection with position-independent assembler (exact agent count)
2. Mining + delivery with position-independent assembler (exact agent count)
3. Formation required, all orientations work (exact agent count)
4. Formation required, specific orientation only (exact agent count)
5. Resource collection with position-independent assembler (extra agents)
6. Mining + delivery with position-independent assembler (extra agents)
7. Formation required, all orientations work (extra agents)
8. Formation required, specific orientation only (extra agents)

Formation Patterns Include:
- Lines: horizontal, vertical, diagonal (NW-SE, NE-SW)
- T-shapes: cardinal and diagonal orientations
- Crosses: orthogonal and diagonal
- L-shapes: various orientations with diagonal corners
- Partial and full rings around the assembler

Reward Sharing: Each agent receives individual rewards for hearts. The 0.5 reward sharing
for credit assignment should be implemented at the training/wrapper level, not in the
environment configuration itself.
"""

import random
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.builder import building
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AssemblerConfig,
    GameConfig,
    MettaGridConfig,
    Position,
    RecipeConfig,
)
from mettagrid.map_builder.perimeter_incontext import PerimeterInContextMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from pydantic import Field

# Resource types available for assembler tasks
RESOURCE_TYPES = [
    "ore_red",
    "ore_blue",
    "ore_green",
    "battery_red",
    "battery_blue",
    "battery_green",
    "laser",
    "blueprint",
    "armor",
]

# Mine types for resource generation
MINE_TYPES = {
    "ore_red": "mine_red",
    "ore_blue": "mine_blue",
    "ore_green": "mine_green",
}


class LPParams:
    """Learning Progress parameters for curriculum learning."""

    def __init__(
        self,
        ema_timescale: float = 0.001,
        exploration_bonus: float = 0.1,
        max_memory_tasks: int = 1000,
        max_slice_axes: int = 3,
        progress_smoothing: float = 0.1,
        enable_detailed_slice_logging: bool = False,
        num_active_tasks: int = 1000,
        rand_task_rate: float = 0.25,
    ):
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus
        self.max_memory_tasks = max_memory_tasks
        self.max_slice_axes = max_slice_axes
        self.progress_smoothing = progress_smoothing
        self.enable_detailed_slice_logging = enable_detailed_slice_logging
        self.num_active_tasks = num_active_tasks
        self.rand_task_rate = rand_task_rate


@dataclass
class FormationPattern:
    """Defines a formation pattern for agents around the assembler."""

    name: str
    positions: List[Position]
    num_agents: int
    description: str


# Formation patterns for different numbers of agents
FORMATION_PATTERNS = {
    2: [
        FormationPattern("line_horizontal", ["W", "E"], 2, "Horizontal line"),
        FormationPattern("line_vertical", ["N", "S"], 2, "Vertical line"),
        FormationPattern("line_diagonal_nw_se", ["NW", "SE"], 2, "Diagonal line NW-SE"),
        FormationPattern("line_diagonal_ne_sw", ["NE", "SW"], 2, "Diagonal line NE-SW"),
    ],
    3: [
        FormationPattern("T_up", ["N", "W", "E"], 3, "T-shape pointing up"),
        FormationPattern("T_down", ["S", "W", "E"], 3, "T-shape pointing down"),
        FormationPattern("T_left", ["W", "N", "S"], 3, "T-shape pointing left"),
        FormationPattern("T_right", ["E", "N", "S"], 3, "T-shape pointing right"),
        FormationPattern("T_diagonal_nw", ["NW", "N", "W"], 3, "T-shape diagonal NW"),
        FormationPattern("T_diagonal_ne", ["NE", "N", "E"], 3, "T-shape diagonal NE"),
        FormationPattern("T_diagonal_sw", ["SW", "S", "W"], 3, "T-shape diagonal SW"),
        FormationPattern("T_diagonal_se", ["SE", "S", "E"], 3, "T-shape diagonal SE"),
    ],
    4: [
        FormationPattern("cross", ["N", "S", "W", "E"], 4, "Cross/plus formation"),
        FormationPattern(
            "cross_diagonal", ["NW", "NE", "SW", "SE"], 4, "Diagonal cross formation"
        ),
        FormationPattern("L_shape_nw", ["N", "W", "NW", "S"], 4, "L-shape pointing NW"),
        FormationPattern("L_shape_ne", ["N", "E", "NE", "S"], 4, "L-shape pointing NE"),
        FormationPattern("L_shape_sw", ["S", "W", "SW", "N"], 4, "L-shape pointing SW"),
        FormationPattern("L_shape_se", ["S", "E", "SE", "N"], 4, "L-shape pointing SE"),
    ],
    5: [
        FormationPattern(
            "plus_corner_nw", ["N", "S", "W", "E", "NW"], 5, "Cross with NW corner"
        ),
        FormationPattern(
            "plus_corner_ne", ["N", "S", "W", "E", "NE"], 5, "Cross with NE corner"
        ),
        FormationPattern(
            "plus_corner_sw", ["N", "S", "W", "E", "SW"], 5, "Cross with SW corner"
        ),
        FormationPattern(
            "plus_corner_se", ["N", "S", "W", "E", "SE"], 5, "Cross with SE corner"
        ),
        FormationPattern(
            "diagonal_plus_center",
            ["NW", "NE", "SW", "SE", "N"],
            5,
            "Diagonal cross + center",
        ),
        FormationPattern(
            "L_extended_nw", ["N", "W", "S", "SW", "NW"], 5, "Extended L-shape NW"
        ),
        FormationPattern(
            "L_extended_ne", ["N", "E", "S", "SE", "NE"], 5, "Extended L-shape NE"
        ),
    ],
    6: [
        FormationPattern(
            "full_sides",
            ["N", "S", "W", "E", "NW", "NE"],
            6,
            "Six positions - top half",
        ),
        FormationPattern(
            "full_sides_alt",
            ["N", "S", "W", "E", "SW", "SE"],
            6,
            "Six positions - bottom half",
        ),
        FormationPattern(
            "hexagon_partial", ["NW", "N", "NE", "SW", "S", "SE"], 6, "Partial hexagon"
        ),
        FormationPattern(
            "double_L", ["N", "W", "NW", "S", "E", "SE"], 6, "Double L formation"
        ),
    ],
    7: [
        FormationPattern(
            "almost_full",
            ["N", "S", "W", "E", "NW", "NE", "SW"],
            7,
            "Almost full - missing SE",
        ),
        FormationPattern(
            "almost_full_alt",
            ["N", "S", "W", "E", "NW", "NE", "SE"],
            7,
            "Almost full - missing SW",
        ),
        FormationPattern(
            "ring_plus_center",
            ["NW", "N", "NE", "W", "E", "SW", "S"],
            7,
            "Ring formation with gaps",
        ),
    ],
    8: [
        FormationPattern(
            "full_ring",
            ["NW", "N", "NE", "W", "E", "SW", "S", "SE"],
            8,
            "Complete ring around assembler",
        ),
    ],
}


class AssemblerTaskGenerator(TaskGenerator):
    """Task generator for assembler coordination environments."""

    class Config(TaskGeneratorConfig["AssemblerTaskGenerator"]):
        """Configuration for AssemblerTaskGenerator."""

        # Environment types: 1-4 are exact agent count, 5-8 have extra agents
        environment_types: list[int] = Field(
            default_factory=lambda: list(range(1, 9)),
            description="Environment types to sample from (1-8)",
        )

        # Agent counts for exact match environments (1-4)
        num_agents_exact: list[int] = Field(
            default_factory=lambda: [2, 3, 4, 5, 6, 7, 8],
            description="Number of agents for exact match environments",
        )

        # Agent counts for extra agent environments (5-8)
        num_agents_extra: list[int] = Field(
            default_factory=lambda: [4, 5, 6, 7, 8, 9, 10],
            description="Number of agents for extra agent environments",
        )

        # Total resources needed
        total_resources: list[int] = Field(
            default_factory=lambda: [3, 4, 6, 8, 10, 12],
            description="Total resources needed for completion",
        )

        # Number of mines for environments 2 and 6
        num_mines: list[int] = Field(
            default_factory=lambda: [2, 3, 4, 5, 6],
            description="Number of mines to place",
        )

        # Initial resource ratio for environments 1 and 5
        initial_resource_ratios: list[float] = Field(
            default_factory=lambda: [0.0, 0.3, 0.5, 0.7],
            description="Fraction of resources agents start with",
        )

        # Room sizes
        room_sizes: list[str] = Field(
            default_factory=lambda: ["small", "medium"],
            description="Room sizes to sample from",
        )

        # Note: Reward sharing will be implemented at the environment wrapper level
        # Individual agents get full reward, shared component handled by training

        max_steps: int = Field(default=512, description="Episode length")

    def __init__(self, config: "AssemblerTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.resource_types = RESOURCE_TYPES.copy()

    def _get_formation_patterns(self, num_agents: int) -> List[FormationPattern]:
        """Get available formation patterns for a given number of agents."""
        if num_agents in FORMATION_PATTERNS:
            return FORMATION_PATTERNS[num_agents]
        else:
            # For larger numbers, use the largest available pattern
            max_agents = max(FORMATION_PATTERNS.keys())
            return FORMATION_PATTERNS[max_agents]

    def _create_recipes_for_environment_type(
        self,
        env_type: int,
        num_agents: int,
        resources_needed: Dict[str, int],
        rng: random.Random,
    ) -> List[tuple[List[Position], RecipeConfig]]:
        """Create recipes based on environment type."""
        recipes = []

        if env_type in [1, 2, 5, 6]:  # Position-independent environments
            recipe = RecipeConfig(
                input_resources=resources_needed,
                output_resources={"heart": 1},
                cooldown=30,
            )
            recipes.append((["Any"], recipe))

        elif env_type in [3, 7]:  # Formation required (all orientations work)
            formation_patterns = self._get_formation_patterns(num_agents)
            for pattern in formation_patterns:
                recipe = RecipeConfig(
                    input_resources=resources_needed,
                    output_resources={"heart": 1},
                    cooldown=30,
                )
                recipes.append((pattern.positions, recipe))

        elif env_type in [4, 8]:  # Formation required (specific orientation)
            formation_patterns = self._get_formation_patterns(num_agents)
            if formation_patterns:
                chosen_pattern = rng.choice(formation_patterns)
                recipe = RecipeConfig(
                    input_resources=resources_needed,
                    output_resources={"heart": 1},
                    cooldown=30,
                )
                recipes.append((chosen_pattern.positions, recipe))

        return recipes

    def _place_mines_for_resources(
        self, resources_needed: Dict[str, int], num_mines: int, rng: random.Random
    ) -> Dict[str, int]:
        """Determine mine placement for needed resources."""
        map_builder_objects = {}

        # Only place mines for ore resources
        ore_resources = {
            k: v for k, v in resources_needed.items() if k.startswith("ore_")
        }

        if not ore_resources:
            return map_builder_objects

        # Distribute mines across needed ore types
        mine_types_needed = list(set(MINE_TYPES[res] for res in ore_resources.keys()))
        mines_per_type = max(1, num_mines // len(mine_types_needed))

        for mine_type in mine_types_needed:
            map_builder_objects[mine_type] = mines_per_type

        # Add any remaining mines randomly
        remaining_mines = num_mines - sum(map_builder_objects.values())
        for _ in range(remaining_mines):
            mine_type = rng.choice(mine_types_needed)
            map_builder_objects[mine_type] += 1

        return map_builder_objects

    def _generate_initial_inventory(
        self,
        resources_needed: Dict[str, int],
        initial_ratio: float,
        num_agents: int,
        rng: random.Random,
    ) -> Dict[str, int]:
        """Generate initial inventory distribution for agents."""
        initial_inventory = {}

        for resource, needed in resources_needed.items():
            initial_amount = int(needed * initial_ratio)
            if initial_amount > 0:
                initial_inventory[resource] = initial_amount

        return initial_inventory

    def _make_env_cfg(
        self,
        env_type: int,
        num_agents: int,
        recipe_agent_count: int,
        resources_needed: Dict[str, int],
        num_mines: int,
        initial_ratio: float,
        room_size: str,
        rng: random.Random,
        max_steps: int = 512,
    ) -> MettaGridConfig:
        """Create environment configuration."""

        # Create assembler recipes using recipe_agent_count for formation requirements
        recipes = self._create_recipes_for_environment_type(
            env_type, recipe_agent_count, resources_needed, rng
        )

        # Create assembler configuration
        assembler = AssemblerConfig(
            type_id=20,  # Use unique ID for assembler
            recipes=recipes,
        )

        # Base game objects
        game_objects = {
            "assembler": assembler,
            "wall": building.wall,
        }

        # Map builder objects (what gets placed on the map)
        map_builder_objects = {"assembler": 1}  # Always place one assembler

        # Add mines for environments that need them
        if env_type in [2, 6]:  # Mining environments
            mine_objects = self._place_mines_for_resources(
                resources_needed, num_mines, rng
            )
            map_builder_objects.update(mine_objects)

            # Add mine configurations to game objects
            for mine_type, count in mine_objects.items():
                if mine_type == "mine_red":
                    game_objects["mine_red"] = building.mine_red
                elif mine_type == "mine_blue":
                    game_objects["mine_blue"] = building.mine_blue
                elif mine_type == "mine_green":
                    game_objects["mine_green"] = building.mine_green

        # Determine room dimensions
        size_range = (
            (8, 12)
            if room_size == "medium"
            else (12, 18)
            if room_size == "large"
            else (6, 10)  # small
        )

        width, height = (
            rng.randint(size_range[0], size_range[1]),
            rng.randint(size_range[0], size_range[1]),
        )

        # Generate initial inventory for agents
        initial_inventory = {}
        if env_type in [1, 5]:  # Environments where agents start with resources
            initial_inventory = self._generate_initial_inventory(
                resources_needed, initial_ratio, num_agents, rng
            )

        # Set up agent configuration
        # Note: For credit assignment, each agent gets individual reward for hearts
        # Reward sharing of 0.5 will be implemented at training level
        agent_rewards = AgentRewards(
            inventory={"heart": 1.0}  # Individual reward for getting heart
        )

        agent_config = AgentConfig(
            default_resource_limit=10,
            resource_limits={"heart": 100},
            rewards=agent_rewards,
            initial_inventory=initial_inventory,
        )

        # Configure actions
        actions = ActionsConfig(
            noop=ActionConfig(),
            move=ActionConfig(),
            rotate=ActionConfig(enabled=False),
            put_items=ActionConfig(),
            get_items=ActionConfig(),
        )

        # Create environment
        return MettaGridConfig(
            game=GameConfig(
                max_steps=max_steps,
                num_agents=num_agents,
                objects=game_objects,
                actions=actions,
                agent=agent_config,
                map_builder=MapGen.Config(
                    instances=num_agents,
                    instance_map=PerimeterInContextMapBuilder.Config(
                        agents=1,
                        width=width,
                        height=height,
                        objects=map_builder_objects,
                    ),
                ),
            )
        )

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        """Generate a single task configuration."""

        # Sample environment type
        env_type = rng.choice(self.config.environment_types)

        # Determine number of agents based on environment type
        if env_type <= 4:  # Exact match environments
            num_agents = rng.choice(self.config.num_agents_exact)
        else:  # Extra agent environments (5-8)
            num_agents = rng.choice(self.config.num_agents_extra)

        # For extra agent environments (5-8), the recipe should require fewer agents than total
        if env_type > 4:
            # Recipe requires 2-4 agents, but environment has more agents
            recipe_agent_count = rng.randint(2, min(4, num_agents - 1))
        else:
            # For exact environments (1-4), recipe requires all agents
            recipe_agent_count = num_agents

        # Sample resources needed
        total_resource_count = rng.choice(self.config.total_resources)

        # Choose resource types
        num_resource_types = rng.randint(1, min(3, len(self.resource_types)))
        chosen_resources = rng.sample(self.resource_types, num_resource_types)

        # Distribute total resources across chosen types
        resources_needed = {}
        remaining = total_resource_count
        for i, resource in enumerate(chosen_resources):
            if i == len(chosen_resources) - 1:  # Last resource gets remainder
                resources_needed[resource] = remaining
            else:
                amount = rng.randint(
                    1, max(1, remaining - len(chosen_resources) + i + 1)
                )
                resources_needed[resource] = amount
                remaining -= amount

        # Sample other parameters
        num_mines = rng.choice(self.config.num_mines)
        initial_ratio = rng.choice(self.config.initial_resource_ratios)
        room_size = rng.choice(self.config.room_sizes)

        # Generate environment configuration
        env_cfg = self._make_env_cfg(
            env_type=env_type,
            num_agents=num_agents,
            recipe_agent_count=recipe_agent_count,
            resources_needed=resources_needed,
            num_mines=num_mines,
            initial_ratio=initial_ratio,
            room_size=room_size,
            rng=rng,
            max_steps=self.config.max_steps,
        )

        # Add metadata for evaluation
        env_cfg.label = f"assemble_env{env_type}_agents{num_agents}_res{total_resource_count}_{room_size}"

        # Estimate rewards for curriculum learning
        best_case, worst_case = self._estimate_max_rewards(
            env_type, num_agents, total_resource_count, self.config.max_steps
        )

        env_cfg.game.reward_estimates = {
            "best_case_optimal_reward": best_case,
            "worst_case_optimal_reward": worst_case,
        }

        return env_cfg

    def _estimate_max_rewards(
        self, env_type: int, num_agents: int, total_resources: int, max_steps: int
    ) -> tuple[float, float]:
        """Estimate maximum possible rewards for the task."""

        # Base time estimates
        movement_cost = 5  # Steps to move to assembler
        resource_gathering_cost = 8  # Steps to gather one resource
        formation_setup_cost = 10  # Additional steps to form patterns
        assembler_cooldown = 30  # Cooldown between uses

        # Environment-specific costs
        if env_type in [1, 5]:  # Start with some resources
            gather_cost = (
                total_resources * resource_gathering_cost * 0.3
            )  # Only need to gather some
        elif env_type in [2, 6]:  # Must mine all resources
            gather_cost = total_resources * resource_gathering_cost
        else:  # Formation environments with resources
            gather_cost = 0  # Assume resources are provided

        if env_type in [3, 4, 7, 8]:  # Formation required
            formation_cost = formation_setup_cost
        else:
            formation_cost = 0

        # Best case: Perfect coordination
        best_case_time = (
            movement_cost + gather_cost + formation_cost + assembler_cooldown
        )
        best_case_hearts = max(
            1, (max_steps - best_case_time) // assembler_cooldown + 1
        )

        # Worst case: Poor coordination, trial and error
        exploration_factor = 3 if env_type in [3, 4, 7, 8] else 1.5
        worst_case_time = (
            movement_cost + gather_cost + formation_cost
        ) * exploration_factor
        worst_case_hearts = max(0, (max_steps - worst_case_time) // assembler_cooldown)

        return float(best_case_hearts), float(worst_case_hearts)


def make_mettagrid() -> MettaGridConfig:
    """Create a basic assembler environment for testing."""
    task_generator_cfg = AssemblerTaskGenerator.Config(
        environment_types=[1],
        num_agents_exact=[3],
        total_resources=[4],
        room_sizes=["small"],
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum(
    environment_types: list[int] = None,
    num_agents_exact: list[int] = None,
    num_agents_extra: list[int] = None,
    total_resources: list[int] = None,
    num_mines: list[int] = None,
    initial_resource_ratios: list[float] = None,
    room_sizes: list[str] = None,
    lp_params: LPParams = LPParams(),
) -> CurriculumConfig:
    """Create curriculum configuration for assembler tasks."""

    # Use defaults if not specified
    environment_types = environment_types or list(range(1, 9))
    num_agents_exact = num_agents_exact or [2, 3, 4, 5, 6]
    num_agents_extra = num_agents_extra or [4, 5, 6, 7, 8]
    total_resources = total_resources or [3, 4, 6, 8, 10]
    num_mines = num_mines or [2, 3, 4, 5]
    initial_resource_ratios = initial_resource_ratios or [0.0, 0.3, 0.5, 0.7]
    room_sizes = room_sizes or ["small", "medium"]

    task_generator_cfg = AssemblerTaskGenerator.Config(
        environment_types=environment_types,
        num_agents_exact=num_agents_exact,
        num_agents_extra=num_agents_extra,
        total_resources=total_resources,
        num_mines=num_mines,
        initial_resource_ratios=initial_resource_ratios,
        room_sizes=room_sizes,
    )

    algorithm_config = LearningProgressConfig(
        ema_timescale=lp_params.ema_timescale,
        exploration_bonus=lp_params.exploration_bonus,
        max_memory_tasks=lp_params.max_memory_tasks,
        max_slice_axes=lp_params.max_slice_axes,
        progress_smoothing=lp_params.progress_smoothing,
        enable_detailed_slice_logging=lp_params.enable_detailed_slice_logging,
        num_active_tasks=lp_params.num_active_tasks,
        rand_task_rate=lp_params.rand_task_rate,
    )

    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum_style: str = "basic", lp_params: Union[LPParams, Dict] = LPParams()
) -> TrainTool:
    """Create training configuration for assembler tasks."""

    # Handle case where run tool passes dict instead of LPParams object
    if isinstance(lp_params, dict):
        lp_params = LPParams(**lp_params)

    curriculum_args = {
        "basic": {
            "environment_types": [1, 2, 3, 4],  # Only exact agent environments
            "num_agents_exact": [2, 3, 4],
            "total_resources": [3, 4, 6],
            "room_sizes": ["small"],
            "lp_params": lp_params,
        },
        "with_extra_agents": {
            "environment_types": list(range(1, 9)),  # All environments
            "num_agents_exact": [2, 3, 4, 5],
            "num_agents_extra": [4, 5, 6, 7],
            "total_resources": [3, 4, 6, 8],
            "room_sizes": ["small", "medium"],
            "lp_params": lp_params,
        },
        "complex": {
            "environment_types": list(range(1, 9)),
            "num_agents_exact": [2, 3, 4, 5, 6, 7, 8],  # Now includes larger formations
            "num_agents_extra": [4, 5, 6, 7, 8, 9, 10],  # Supports larger groups
            "total_resources": [3, 4, 6, 8, 10, 12],
            "room_sizes": ["small", "medium", "large"],
            "lp_params": lp_params,
        },
    }

    curriculum = make_curriculum(**curriculum_args[curriculum_style])

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum,
        evaluation=EvaluationConfig(simulations=make_assembler_eval_suite()),
    )

    # Assembler tasks may need longer episodes for coordination
    # batch_size must be divisible by minibatch_size (default 16384)
    trainer_cfg.batch_size = 2064384  # 126 * 16384 = 2064384
    trainer_cfg.bptt_horizon = 512

    return TrainTool(trainer=trainer_cfg)


def make_assembler_eval_suite() -> List[SimulationConfig]:
    """Create evaluation suite for assembler environments."""
    simulations = []

    # Test each environment type
    for env_type in range(1, 9):
        # Small environment
        task_gen_cfg = AssemblerTaskGenerator.Config(
            environment_types=[env_type],
            num_agents_exact=[3] if env_type <= 4 else [],
            num_agents_extra=[5] if env_type > 4 else [],
            total_resources=[6],
            room_sizes=["small"],
        )
        task_gen = AssemblerTaskGenerator(task_gen_cfg)
        env = task_gen.get_task(0)

        simulations.append(
            SimulationConfig(name=f"assembler/env_{env_type}_simple", env=env)
        )

        # Complex environment
        task_gen_cfg_complex = AssemblerTaskGenerator.Config(
            environment_types=[env_type],
            num_agents_exact=[4] if env_type <= 4 else [],
            num_agents_extra=[6] if env_type > 4 else [],
            total_resources=[10],
            room_sizes=["medium"],
        )
        task_gen_complex = AssemblerTaskGenerator(task_gen_cfg_complex)
        env_complex = task_gen_complex.get_task(1)

        simulations.append(
            SimulationConfig(name=f"assembler/env_{env_type}_complex", env=env_complex)
        )

    return simulations


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    """Create play tool for assembler environments."""
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="assembler_playground",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    """Create replay tool for assembler environments."""
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="assembler_replay",
        ),
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    """Create evaluation tool for assembler environments."""
    simulations = simulations or make_assembler_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    """Run hyperparameter sweep experiment."""
    curriculum_styles = ["basic", "with_extra_agents", "complex"]
    progress_smoothings = list(np.linspace(0.05, 0.15, 2))
    exploration_bonuses = list(np.linspace(0.05, 0.15, 2))

    total_experiments = (
        len(curriculum_styles) * len(progress_smoothings) * len(exploration_bonuses)
    )
    print(f"Total experiments to run: {total_experiments}")

    for curriculum_style in curriculum_styles:
        for progress_smoothing in progress_smoothings:
            for exploration_bonus in exploration_bonuses:
                LPParams(
                    progress_smoothing=progress_smoothing,
                    exploration_bonus=exploration_bonus,
                )

                subprocess.run(
                    [
                        "./devops/skypilot/launch.py",
                        "experiments.recipes.assemble.train",
                        f"run=assemble_{curriculum_style}_PS{progress_smoothing:.2f}_EB{exploration_bonus:.2f}",
                        f"curriculum_style={curriculum_style}",
                        f"lp_params.progress_smoothing={progress_smoothing:.2f}",
                        f"lp_params.exploration_bonus={exploration_bonus:.2f}",
                        "--gpus=4",
                        "--heartbeat-timeout=3600",
                        "--skip-git-check",
                    ]
                )
                time.sleep(1)


if __name__ == "__main__":
    experiment()
