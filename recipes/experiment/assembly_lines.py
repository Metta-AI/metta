"""Assembly Lines recipe - STABLE
This recipe is automatically validated in CI and release processes.
"""

import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from metta.agent.policies.vit import ViTDefaultConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.builder import empty_assemblers
from mettagrid.builder.envs import make_assembly_lines
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    ProtocolConfig,
)

curriculum_args = {
    "level_0": {
        "chain_lengths": [1],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny"],
        "terrains": ["no-terrain"],
    },
    "level_1": {
        "chain_lengths": [1, 2],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny"],
        "terrains": ["no-terrain"],
    },
    "level_2": {
        "chain_lengths": [1, 2, 3],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny"],
        "terrains": ["no-terrain"],
    },
    "tiny": {
        "chain_lengths": [1, 2, 3, 4],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny"],
        "terrains": ["no-terrain"],
    },
    "tiny_small": {
        "chain_lengths": [1, 2, 3, 4],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny", "small"],
        "terrains": ["no-terrain"],
    },
    "all_room_sizes": {
        "chain_lengths": [1, 2, 3, 4],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
        "terrains": ["no-terrain"],
    },
    "longer_chains": {
        "chain_lengths": [1, 2, 3, 4, 5, 6],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
        "terrains": ["no-terrain"],
    },
    "terrain_1": {
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny", "small"],
        "terrains": ["no-terrain", "sparse", "balanced", "dense"],
    },
    "terrain_2": {
        "chain_lengths": [2, 3, 4],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small"],
        "terrains": ["no-terrain", "sparse", "balanced", "dense"],
    },
    "terrain_3": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
        "terrains": ["no-terrain", "sparse", "balanced", "dense"],
    },
    "terrain_4": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
        "terrains": ["no-terrain", "sparse", "balanced", "dense"],
    },
    "full": {
        "chain_lengths": [2, 3, 4, 5, 6],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium", "large"],
        "terrains": ["no-terrain", "sparse", "balanced", "dense"],
    },
    "terrain_2_progressive": {
        # Progressive curriculum: starts with easiest tasks, automatically progresses to harder ones
        # The learning progress algorithm will sample easier tasks more initially, then shift to harder
        # tasks as the agent learns. This follows curriculum learning principles (Bengio et al., 2009).
        "chain_lengths": [1, 2, 3],  # Start with chain_length=1 for basic pattern learning
        "num_sinks": [0, 1],  # Start without 2 sinks to reduce initial complexity
        "room_sizes": ["tiny", "small"],  # Small rooms for easier navigation
        "terrains": ["no-terrain", "sparse", "balanced"],  # Exclude dense terrain initially
    },
}

ASSEMBLER_TYPES = {
    "generator_red": empty_assemblers.generator_red,
    "generator_blue": empty_assemblers.generator_blue,
    "generator_green": empty_assemblers.generator_green,
    "mine_red": empty_assemblers.mine_red,
    "mine_blue": empty_assemblers.mine_blue,
    "mine_green": empty_assemblers.mine_green,
    "altar": empty_assemblers.altar,
    "factory": empty_assemblers.factory,
    "temple": empty_assemblers.temple,
    "armory": empty_assemblers.armory,
    "lab": empty_assemblers.lab,
    "lasery": empty_assemblers.lasery,
}

size_ranges = {
    "tiny": (5, 10),  # 2 objects 2 agents max for assemblers
    "small": (10, 20),  # 9 objects, 5 agents max
    "medium": (20, 30),
    "large": (30, 40),
    "xlarge": (40, 50),
}

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


def make_assembly_line_eval_env(
    chain_length: int,
    num_sinks: int,
    room_size: str,
    terrain: str,
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        chain_lengths=[chain_length],
        num_sinks=[num_sinks],
        room_sizes=[room_size],
        terrains=[terrain],
    )
    task_generator = AssemblyLinesTaskGenerator(task_generator_cfg)
    # Use deterministic seed based on task parameters for reproducible evaluation
    seed = hash((chain_length, num_sinks, room_size, terrain)) % (2**31)
    random.seed(seed)
    env = task_generator.get_task(seed)
    # Normalize heart reward to 0.333 for consistent reward scaling
    env.game.agent.rewards.inventory["heart"] = 0.333
    return env


def make_assembly_line_eval_suite() -> list[SimulationConfig]:
    return [
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium",
            env=make_assembly_line_eval_env(2, 2, "medium", "no-terrain"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="2c_2s_medium_balanced",
            env=make_assembly_line_eval_env(2, 2, "medium", "balanced"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_1s_medium",
            env=make_assembly_line_eval_env(3, 1, "medium", "no-terrain"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="3c_2s_medium",
            env=make_assembly_line_eval_env(3, 2, "medium", "no-terrain"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_terrain_dense",
            env=make_assembly_line_eval_env(4, 1, "medium", "dense"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_1s_medium_balanced",
            env=make_assembly_line_eval_env(4, 1, "medium", "balanced"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_medium_balanced",
            env=make_assembly_line_eval_env(4, 2, "medium", "balanced"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="4c_2s_large_balanced",
            env=make_assembly_line_eval_env(4, 2, "large", "balanced"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium",
            env=make_assembly_line_eval_env(5, 1, "medium", "no-terrain"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_1s_medium_balanced",
            env=make_assembly_line_eval_env(5, 1, "medium", "balanced"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_medium_balanced",
            env=make_assembly_line_eval_env(5, 2, "medium", "balanced"),
        ),
        SimulationConfig(
            suite="in_context_ordered_chains",
            name="5c_2s_large_dense",
            env=make_assembly_line_eval_env(5, 2, "large", "dense"),
        ),
    ]


@dataclass
class _BuildCfg:
    used_objects: list[str] = field(default_factory=list)
    game_objects: dict[str, Any] = field(default_factory=dict)
    map_builder_objects: dict[str, int] = field(default_factory=dict)
    input_resources: set[str] = field(default_factory=set)


class AssemblyLinesTaskGenerator(TaskGenerator):
    def __init__(self, config: "AssemblyLinesTaskGenerator.Config"):
        super().__init__(config)
        self.assembler_types = ASSEMBLER_TYPES.copy()
        self.resource_types = RESOURCE_TYPES.copy()
        self.config = config

    class Config(TaskGeneratorConfig["AssemblyLinesTaskGenerator"]):
        chain_lengths: list[int]
        num_sinks: list[int]
        room_sizes: list[str]
        terrains: list[str]

    def _choose_assembler_name(self, pool: dict[str, Any], used: set[str], rng: random.Random) -> str:
        """Pick an unused assembler prefab name from the pool."""
        choices = [name for name in pool.keys() if name not in used]
        if not choices:
            raise ValueError("No available assembler names left to choose from.")
        return str(rng.choice(choices))

    def _add_assembler(
        self,
        input_resources: dict[str, int],
        output_resources: dict[str, int],
        cfg: _BuildCfg,
        rng: random.Random,
        cooldown: int = 10,
    ):
        assembler_name = self._choose_assembler_name(self.assembler_types, set(cfg.used_objects), rng)
        assembler = self.assembler_types[assembler_name].copy()
        cfg.used_objects.append(assembler_name)

        protocol = ProtocolConfig(
            input_resources=input_resources,
            output_resources=output_resources,
            cooldown=cooldown,
        )
        assembler.protocols = [protocol]
        cfg.game_objects[assembler_name] = assembler
        cfg.map_builder_objects[assembler_name] = 1

    def _make_resource_chain(
        self,
        chain_length: int,
        avg_hop: float,
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        resources = rng.sample(self.resource_types, chain_length)
        cooldown = avg_hop * chain_length
        resource_chain = ["nothing"] + list(resources) + ["heart"]
        for i in range(len(resource_chain) - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            input_resources = {} if input_resource == "nothing" else {input_resource: 1}
            if not input_resource == "nothing":
                cfg.input_resources.add(input_resource)
            self._add_assembler(
                input_resources=input_resources,
                output_resources={output_resource: 1},
                cfg=cfg,
                cooldown=int(cooldown),
                rng=rng,
            )

    def _make_sinks(
        self,
        num_sinks: int,
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        for _ in range(num_sinks):
            self._add_assembler(
                input_resources={resource: 1 for resource in list(cfg.input_resources)},
                output_resources={},
                cfg=cfg,
                rng=rng,
            )

    def _make_env_cfg(
        self,
        chain_length,
        num_sinks,
        width,
        height,
        max_steps,
        terrain,
        rng,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        self._make_resource_chain(chain_length, width + height / 2, cfg, rng)
        self._make_sinks(num_sinks, cfg, rng)

        # CRITICAL FIX: Add ALL assembler types to game_objects to maintain config invariants
        # The simulator requires all simulations to have the same object_type_names.
        # By including all assembler types (even unused ones), we ensure consistency.
        for assembler_name, assembler_config in self.assembler_types.items():
            if assembler_name not in cfg.game_objects:
                # Add unused assembler types with empty protocols so they exist in config
                # but won't be placed in the map (map_builder_objects doesn't include them)
                unused_assembler = assembler_config.copy()
                unused_assembler.protocols = []  # No protocols = not usable, but exists in config
                cfg.game_objects[assembler_name] = unused_assembler

        env_cfg = make_assembly_lines(
            num_agents=1,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            terrain=terrain,
            chain_length=chain_length,
            num_sinks=num_sinks,
        )

        # Normalize heart reward to 0.333 for consistent reward scaling in training
        # This matches the normalization used in evaluation and other recipes
        env_cfg.game.agent.rewards.inventory["heart"] = 0.333

        return env_cfg

    def calculate_max_steps(
        self, chain_length: int, num_sinks: int, width: int, height: int, terrain: str = "no-terrain"
    ) -> int:
        """Public API for calculating max_steps. Delegates to _calculate_max_steps."""
        return self._calculate_max_steps(chain_length, num_sinks, width, height, terrain)

    def _get_width_and_height(self, room_size: str, rng: random.Random):
        lo, hi = size_ranges[room_size]
        width = rng.randint(lo, hi)
        height = rng.randint(lo, hi)
        return width, height

    def _calculate_max_steps(self, chain_length: int, num_sinks: int, width: int, height: int, terrain: str) -> int:
        avg_hop = width + height / 2

        # Increase steps_per_attempt from 4 to 6 for better coverage of complex terrain
        steps_per_attempt = 6 * avg_hop
        sink_exploration_cost = steps_per_attempt * num_sinks
        chain_completion_cost = steps_per_attempt * chain_length
        target_completions = 10

        base_steps = int(sink_exploration_cost + target_completions * chain_completion_cost)

        # Add buffer for complex tasks (longer chains, multiple sinks, dense terrain)
        # This helps prevent timeouts on difficult tasks
        # Increased multipliers for dense terrain to prevent collapse (graphs show agent fails on dense terrain)
        # Dense terrain + long chains are the hardest tasks that cause performance collapse
        if terrain == "dense" and chain_length >= 4:
            complexity_multiplier = 3.0  # Increased from 2.5 to 3.0 - most time for hardest tasks
        elif terrain == "dense":
            complexity_multiplier = 2.2  # Increased from 1.8 to 2.2 - dense terrain needs more time
        elif terrain == "balanced" and chain_length >= 4:
            complexity_multiplier = 2.0  # New: balanced terrain + long chains need more time
        elif chain_length >= 4 or num_sinks >= 2:
            complexity_multiplier = 1.5  # More time for complex tasks
        else:
            complexity_multiplier = 1.0

        return int(base_steps * complexity_multiplier)

    def _setup_task(self, rng: random.Random):
        cfg = self.config
        chain_length = rng.choice(cfg.chain_lengths)
        num_sinks = rng.choice(cfg.num_sinks)
        room_size = rng.choice(cfg.room_sizes)
        terrain = rng.choice(cfg.terrains)
        width, height = self._get_width_and_height(room_size, rng)
        max_steps = self._calculate_max_steps(chain_length, num_sinks, width, height, terrain)
        return chain_length, num_sinks, room_size, width, height, max_steps, terrain

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
        estimate_max_rewards: bool = False,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        chain_length, num_sinks, room_size, width, height, max_steps, terrain = self._setup_task(rng)

        env_cfg = self._make_env_cfg(
            chain_length=chain_length,
            num_sinks=num_sinks,
            width=width,
            height=height,
            terrain=terrain,
            max_steps=max_steps,
            rng=rng,
        )

        env_cfg.label = f"{room_size}_{chain_length}chain_{num_sinks}sinks_{terrain}"
        return env_cfg


def make_task_generator_cfg(
    chain_lengths,
    num_sinks,
    room_sizes,
    terrains,
):
    return AssemblyLinesTaskGenerator.Config(
        chain_lengths=chain_lengths,
        num_sinks=num_sinks,
        room_sizes=room_sizes,
        terrains=terrains,
    )


def _compute_curriculum_difficulty(curriculum_style: str) -> dict[str, bool]:
    """
    Compute curriculum difficulty characteristics based on task parameters.

    Returns:
        Dictionary with difficulty flags:
        - has_dense_terrain: Whether curriculum includes dense terrain
        - has_long_chains: Whether curriculum includes chain_length >= 4
        - has_multiple_sinks: Whether curriculum includes num_sinks >= 2
        - starts_with_basics: Whether curriculum includes chain_length=1
    """
    cfg = curriculum_args.get(curriculum_style, curriculum_args["level_0"])
    return {
        "has_dense_terrain": "dense" in cfg.get("terrains", []),
        "has_long_chains": any(cl >= 4 for cl in cfg.get("chain_lengths", [])),
        "has_multiple_sinks": any(ns >= 2 for ns in cfg.get("num_sinks", [])),
        "starts_with_basics": 1 in cfg.get("chain_lengths", []),
    }


def _make_algorithm_config(curriculum_style: str) -> LearningProgressConfig:
    """
    Create learning progress algorithm configuration based on curriculum difficulty.

    Follows curriculum learning principles:
    - Epsilon-greedy exploration: rand_task_rate balances exploration vs exploitation
    - Learning progress: Algorithm identifies tasks at "zone of proximal development"
    - Slow adaptation: ema_timescale prevents premature curriculum shifts
    - Exploration bonus: Favors under-explored tasks to ensure coverage

    Args:
        curriculum_style: Name of curriculum style from curriculum_args

    Returns:
        LearningProgressConfig with settings appropriate for curriculum difficulty
    """
    difficulty = _compute_curriculum_difficulty(curriculum_style)

    # Base settings: Default LearningProgressConfig values are good starting points
    # Default: rand_task_rate=0.25, ema_timescale=0.001, exploration_bonus=0.1
    base_config = {
        "use_bidirectional": True,  # Use bidirectional learning progress (default)
    }

    # Adjust based on curriculum difficulty
    if difficulty["has_dense_terrain"] and difficulty["has_long_chains"]:
        # Hardest curricula: dense terrain + long chains
        # Need very conservative settings to prevent collapse
        # Theory: High exploration prevents premature focus on unsolvable tasks
        return LearningProgressConfig(
            rand_task_rate=0.65,  # 65% random exploration (epsilon-greedy)
            ema_timescale=0.0001,  # Slow adaptation: ~10k episodes to shift curriculum
            exploration_bonus=0.25,  # Moderate bonus for under-explored tasks
            **base_config,
        )
    elif difficulty["has_dense_terrain"] or difficulty["has_long_chains"]:
        # Medium-hard curricula: either dense terrain OR long chains
        # Moderate settings to balance exploration and learning progress
        return LearningProgressConfig(
            rand_task_rate=0.5,  # 50% random exploration
            ema_timescale=0.0003,  # Moderate adaptation: ~3k episodes to shift
            exploration_bonus=0.15,  # Standard bonus for exploration
            **base_config,
        )
    elif not difficulty["starts_with_basics"]:
        # Progressive curricula that don't start with chain_length=1
        # Need conservative settings to prevent premature hard task sampling
        return LearningProgressConfig(
            rand_task_rate=0.6,  # 60% random exploration
            ema_timescale=0.0002,  # Slow adaptation: ~5k episodes to shift
            exploration_bonus=0.2,  # Higher bonus to favor easier tasks
            **base_config,
        )
    else:
        # Progressive curricula starting with basics (chain_length=1)
        # Can use more standard settings as agent can learn from start
        # Theory: Agent can learn basics, so algorithm can be more responsive
        return LearningProgressConfig(
            rand_task_rate=0.4,  # 40% random exploration (more learning-progress driven)
            ema_timescale=0.0005,  # Moderate adaptation: ~2k episodes to shift
            exploration_bonus=0.15,  # Standard exploration bonus
            **base_config,
        )


def train(
    curriculum_style: str = "terrain_2_progressive",
) -> TrainTool:
    """
    Train ICL recipe with assembly line tasks using automatic curriculum progression.

    The curriculum learning algorithm automatically samples tasks based on learning progress:
    - Tasks with high learning progress (variance in performance) are sampled more
    - As agent masters easier tasks, harder tasks naturally get sampled more
    - This creates automatic progression from easy → hard without manual intervention

    Default curriculum_style is "terrain_2_progressive":
    - Starts with: chain_lengths=[1, 2, 3], num_sinks=[0, 1], terrains without "dense"
    - Agent learns chain_length=1 first, then progresses to 2, then 3
    - Algorithm settings are tuned for progressive curricula starting with basics

    Args:
        curriculum_style: Name of curriculum from curriculum_args dict

    Returns:
        TrainTool configured for ICL training with automatic curriculum progression
    """
    if curriculum_style not in curriculum_args:
        raise ValueError(
            f"Unknown curriculum_style: {curriculum_style}. "
            f"Available: {list(curriculum_args.keys())}"
        )

    task_generator_cfg = make_task_generator_cfg(**curriculum_args[curriculum_style])
    algorithm_config = _make_algorithm_config(curriculum_style)
    curriculum = CurriculumConfig(task_generator=task_generator_cfg, algorithm_config=algorithm_config)

    policy_config = ViTDefaultConfig()

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(simulations=make_assembly_line_eval_suite()),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_mettagrid(task_generator: AssemblyLinesTaskGenerator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def evaluate(policy_uris: str | list[str] | None = None) -> EvaluateTool:
    """Evaluate policies on assembly line tasks."""
    return EvaluateTool(
        simulations=make_assembly_line_eval_suite(),
        policy_uris=policy_uris or [],
    )


def play(curriculum_style: str = "level_0") -> PlayTool:
    task_generator = AssemblyLinesTaskGenerator(make_task_generator_cfg(**curriculum_args[curriculum_style]))
    return PlayTool(sim=SimulationConfig(env=make_mettagrid(task_generator), suite="assembly_lines", name="play"))


def replay(
    curriculum_style: str = "level_0",
) -> ReplayTool:
    task_generator = AssemblyLinesTaskGenerator(make_task_generator_cfg(**curriculum_args[curriculum_style]))
    # Default to the research policy if none specified
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_1.2.2025-09-24/icl_resource_chain_terrain_1.2.2025-09-24:v2070.pt"
    return ReplayTool(
        sim=SimulationConfig(env=make_mettagrid(task_generator), suite="assembly_lines", name="replay"),
        policy_uri=default_policy_uri,
    )


def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "recipes.experiment.assembly_lines.train",
                f"run=assembly_lines_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
