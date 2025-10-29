"""ICL Control Recipe - Architecture Benchmark Configuration.

This recipe implements a controlled benchmark for comparing agent architectures
(vit_reset vs trxl) on in-context learning tasks with assembly lines.

Key Configuration:
- Combat: OFF (assembly_lines environment has no attack/combat mechanics)
- Curriculum Learning: OFF (algorithm_config=None → uniform random task sampling)
- Seed Synchronization: ON (BENCHMARK_SEED=42 for reproducibility)
- Evaluations: ENABLED (every 30 epochs, local evaluation)

The controlled settings ensure fair comparison between architectures by:
1. Disabling adaptive curriculum learning (uniform random sampling instead)
2. Using fixed seed for deterministic weight initialization and task generation
3. Eliminating combat dynamics that could introduce confounding variables
"""

import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from metta.agent.policies.trxl import trxl_policy_config
from metta.agent.policies.vit_reset import ViTResetConfig
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.rl.loss import LossConfig
from metta.rl.system_config import SystemConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.builder import empty_assemblers
from mettagrid.builder.envs import make_assembly_lines
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    ProtocolConfig,
)

# Seed for reproducibility across environment, curriculum, and weight initialization
BENCHMARK_SEED = 63

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

    def _choose_assembler_name(
        self, pool: dict[str, Any], used: set[str], rng: random.Random
    ) -> str:
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
        assembler_name = self._choose_assembler_name(
            self.assembler_types, set(cfg.used_objects), rng
        )
        assembler = self.assembler_types[assembler_name].copy()
        cfg.used_objects.append(assembler_name)

        recipe = (
            [],
            ProtocolConfig(
                input_resources=input_resources,
                output_resources=output_resources,
                cooldown=cooldown,
            ),
        )
        assembler.recipes = [recipe]
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

        return make_assembly_lines(
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

    def calculate_max_steps(
        self, chain_length: int, num_sinks: int, width: int, height: int
    ) -> int:
        avg_hop = width + height / 2

        steps_per_attempt = 4 * avg_hop
        sink_exploration_cost = steps_per_attempt * num_sinks
        chain_completion_cost = steps_per_attempt * chain_length
        target_completions = 10

        return int(sink_exploration_cost + target_completions * chain_completion_cost)

    def _get_width_and_height(self, room_size: str, rng: random.Random):
        lo, hi = size_ranges[room_size]
        width = rng.randint(lo, hi)
        height = rng.randint(lo, hi)
        return width, height

    def _calculate_max_steps(
        self, chain_length: int, num_sinks: int, width: int, height: int
    ) -> int:
        avg_hop = width + height / 2

        steps_per_attempt = 4 * avg_hop
        sink_exploration_cost = steps_per_attempt * num_sinks
        chain_completion_cost = steps_per_attempt * chain_length
        target_completions = 10

        return int(sink_exploration_cost + target_completions * chain_completion_cost)

    def _setup_task(self, rng: random.Random):
        cfg = self.config
        chain_length = rng.choice(cfg.chain_lengths)
        num_sinks = rng.choice(cfg.num_sinks)
        room_size = rng.choice(cfg.room_sizes)
        terrain = rng.choice(cfg.terrains)
        width, height = self._get_width_and_height(room_size, rng)
        max_steps = self._calculate_max_steps(chain_length, num_sinks, width, height)
        return chain_length, num_sinks, room_size, width, height, max_steps, terrain

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
        estimate_max_rewards: bool = False,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        chain_length, num_sinks, room_size, width, height, max_steps, terrain = (
            self._setup_task(rng)
        )

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


def _get_policy_config(architecture: str) -> PolicyArchitecture:
    """Get policy configuration based on architecture name."""
    if architecture == "vit_reset":
        return ViTResetConfig()
    elif architecture == "trxl":
        return trxl_policy_config()
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. Must be 'vit_reset' or 'trxl'"
        )


def _validate_seed_synchronization(train_tool: TrainTool) -> None:
    """Validate seed synchronization across environment, curriculum, and weight initialization.

    This check ensures reproducibility before training begins by verifying:
    1. System seed (weight initialization via seed_everything)
    2. Training environment seed (environment episodes)
    3. Curriculum seed (task sampling - passed to Curriculum.__init__ via training_env.seed)

    All three must equal BENCHMARK_SEED for deterministic training.

    Raises:
        ValueError: If any seed doesn't match BENCHMARK_SEED or curriculum is missing
    """
    errors = []

    # Check 1: System seed for weight initialization
    if train_tool.system.seed != BENCHMARK_SEED:
        errors.append(
            f"System seed mismatch: expected {BENCHMARK_SEED}, got {train_tool.system.seed}"
        )

    # Check 2: Training environment seed (also used for curriculum)
    if train_tool.training_env.seed != BENCHMARK_SEED:
        errors.append(
            f"Training env seed mismatch: expected {BENCHMARK_SEED}, got {train_tool.training_env.seed}"
        )

    # Check 3: Curriculum configuration exists
    if train_tool.training_env.curriculum is None:
        errors.append("Curriculum config is None - benchmark requires curriculum")

    if errors:
        error_msg = "❌ Seed synchronization check FAILED:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)

    # All checks passed
    print("\n" + "=" * 70)
    print("✓ SEED VALIDATION PASSED - Benchmark is ready for reproducible training")
    print("=" * 70)
    print(f"  System seed:        {train_tool.system.seed} → weight initialization")
    print(
        f"  Environment seed:   {train_tool.training_env.seed} → environment episodes"
    )
    print(f"  Curriculum seed:    {train_tool.training_env.seed} → task sampling")
    print(f"  All synchronized to: BENCHMARK_SEED = {BENCHMARK_SEED}")
    print("=" * 70 + "\n")


def train(
    curriculum_style: str = "level_0",
    architecture: str = "vit_reset",
    validate_seeds: bool = True,
) -> TrainTool:
    """Train an agent with the specified curriculum and architecture.

    Args:
        curriculum_style: Curriculum difficulty level
        architecture: Policy architecture ('vit_reset' or 'trxl')
        validate_seeds: If True, validates seed synchronization before training (default: True)
    """
    from experiments.evals.assembly_lines import (
        make_assembly_line_eval_suite,
    )

    task_generator_cfg = make_task_generator_cfg(**curriculum_args[curriculum_style])
    # Curriculum learning disabled - tasks are uniformly randomly sampled
    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg, algorithm_config=None
    )

    policy_config = _get_policy_config(architecture)

    trainer_cfg = TrainerConfig(losses=LossConfig())

    train_tool = TrainTool(
        system=SystemConfig(seed=BENCHMARK_SEED),
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(
            curriculum=curriculum, seed=BENCHMARK_SEED
        ),
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(
            simulations=make_assembly_line_eval_suite(),
            epoch_interval=30,
            evaluate_local=True,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )

    # Validate seed synchronization before proceeding with training
    if validate_seeds:
        _validate_seed_synchronization(train_tool)

    return train_tool


def make_mettagrid(task_generator: AssemblyLinesTaskGenerator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "level_0") -> PlayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="assembly_lines", name="play"
        )
    )


def replay(
    curriculum_style: str = "level_0",
) -> ReplayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    # Default to the research policy if none specified
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_1.2.2025-09-24/icl_resource_chain_terrain_1.2.2025-09-24:v2070.pt"
    return ReplayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="assembly_lines", name="replay"
        ),
        policy_uri=default_policy_uri,
    )


def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.assembly_lines.train",
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
