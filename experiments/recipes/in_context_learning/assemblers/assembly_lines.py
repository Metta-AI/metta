import os
import random
import subprocess
import time
from typing import Optional

from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_icl_assembler
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Position,
)
from experiments.recipes.in_context_learning.in_context_learning import (
    ICLTaskGenerator,
    _BuildCfg,
    num_agents_to_positions,
    play_icl,
    replay_icl,
    train_icl,
)

curriculum_args = {
    "train": {
        "num_agents": [1, 2, 6, 12],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3],
        "chest_positions": [["N"], ["N", "S"], ["N", "S", "E"]],
        "num_chests": [2, 5, 8],
        # Special impossible task types (not combinatorial with above params)
        "impossible_tasks": ["deterministic", "noisy"],
        "impossible_task_probability": 0.01,  # 1% chance to sample an impossible task
    },
    "train_pairs": {
        "num_agents": [2, 6, 12],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[2],
        "chest_positions": [["N"]],
        "num_chests": [2, 5, 8],
    },
    "train_triplets": {
        "num_agents": [3, 6, 12],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[3],
        "chest_positions": [["N"]],
        "num_chests": [2, 5, 8],
    },
    "test": {
        "num_agents": [2],
        "chain_lengths": [2],
        "num_sinks": [0],
        "chest_positions": [["N"]],
        "num_chests": [1],
        "room_sizes": ["medium"],
        "positions": [["Any", "Any"]],
    },
}


def make_task_generator_cfg(
    num_agents: list[int],
    chain_lengths: list[int],
    num_sinks: list[int],
    room_sizes: list[str],
    positions: list[list[Position]],
    map_dir: Optional[str] = None,
    num_chests: list[int] = [0],
    chest_positions: list[list[Position]] = [["N"]],
    impossible_tasks: list[str] = [],
    impossible_task_probability: float = 0.0,
):
    return AssemblyLinesTaskGenerator.Config(
        num_agents=num_agents,
        num_resources=[c - 1 for c in chain_lengths],
        num_converters=num_sinks,
        room_sizes=room_sizes,
        positions=positions,
        map_dir=map_dir,
        num_chests=num_chests,
        chest_positions=chest_positions,
        impossible_tasks=impossible_tasks,
        impossible_task_probability=impossible_task_probability,
    )


class AssemblyLinesTaskGenerator(ICLTaskGenerator):
    """Task generator with support for impossible tasks for curriculum validation.

    Impossible tasks are sampled separately from normal tasks, not combinatorially.
    They serve as negative examples to validate curriculum learning behavior.
    """

    class Config(ICLTaskGenerator.Config):
        """Configuration for assembly line task generator with impossible tasks."""

        impossible_tasks: list[
            str
        ] = []  # List of impossible task types: "deterministic", "noisy"
        impossible_task_probability: float = (
            0.0  # Probability of sampling an impossible task
        )

    def __init__(self, config: "AssemblyLinesTaskGenerator.Config"):
        super().__init__(config)
        self.config: AssemblyLinesTaskGenerator.Config = config

    def _make_resource_chain(
        self,
        resources: list[str],
        avg_hop: float,
        position: list[Position],
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        cooldown = avg_hop * len(resources)
        resource_chain = ["nothing"] + list(resources) + ["heart"]
        for i in range(len(resource_chain) - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            input_resources = {} if input_resource == "nothing" else {input_resource: 1}
            self._add_assembler(
                input_resources=input_resources,
                output_resources={output_resource: 1},
                position=position,
                cfg=cfg,
                cooldown=int(cooldown),
                rng=rng,
            )

    def _make_sinks(
        self,
        num_sinks: int,
        position: list[Position],
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        for _ in range(num_sinks):
            self._add_assembler(
                input_resources={},
                output_resources={},
                position=position,
                cfg=cfg,
                rng=rng,
            )

    def _make_impossible_deterministic_chain(
        self,
        resources: list[str],
        avg_hop: float,
        position: list[Position],
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        """Create an impossible task that requires unobtainable resources.

        This creates a chain that requires 'unobtainium' which doesn't exist,
        ensuring the task always returns reward=0.
        """
        cooldown = avg_hop * (len(resources) + 2)
        # Create a chain that requires unobtainable resource
        resource_chain = ["unobtainium"] + list(resources) + ["heart"]
        for i in range(len(resource_chain) - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            input_resources = {input_resource: 1}
            self._add_assembler(
                input_resources=input_resources,
                output_resources={output_resource: 1},
                position=position,
                cfg=cfg,
                cooldown=int(cooldown),
                rng=rng,
            )

    def _make_impossible_noisy_chain(
        self,
        resources: list[str],
        avg_hop: float,
        position: list[Position],
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        """Create an impossible task with inconsistent/broken rules.

        This creates a chain with randomized cooldowns and broken recipes
        that sometimes work, sometimes don't - providing noisy rewards but
        no learnable pattern.
        """
        # Create chain with wildly varying cooldowns (unpredictable timing)
        resource_chain = ["nothing"] + list(resources) + ["heart"]
        for i in range(len(resource_chain) - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            input_resources = {} if input_resource == "nothing" else {input_resource: 1}

            # Random cooldown makes timing unpredictable
            random_cooldown = int(avg_hop * rng.uniform(0.5, 10.0))

            self._add_assembler(
                input_resources=input_resources,
                output_resources={output_resource: 1},
                position=position,
                cfg=cfg,
                cooldown=random_cooldown,
                rng=rng,
            )

        # Add extra broken assemblers that conflict with the chain
        for _ in range(rng.randint(2, 4)):
            # Random broken recipes that sometimes consume resources without benefit
            conflicting_resource = rng.choice(resources) if resources else "nothing"
            self._add_assembler(
                input_resources={conflicting_resource: 1}
                if conflicting_resource != "nothing"
                else {},
                output_resources={},  # Produces nothing (resource sink)
                position=position,
                cfg=cfg,
                cooldown=int(avg_hop * rng.uniform(1.0, 3.0)),
                rng=rng,
            )

    def _make_env_cfg(
        self,
        num_agents,
        resources,
        num_sinks,
        width,
        height,
        position,
        chest_position,
        num_chests,
        terrain,
        max_steps,
        num_instances,
        rng,
        dir=None,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        self._make_resource_chain(resources, width + height / 2, position, cfg, rng)
        self._make_sinks(num_sinks, position, cfg, rng)
        if num_chests > 0:
            self._make_chests(num_chests, cfg, chest_position)

        if dir is not None and os.path.exists(dir):
            return self.load_from_numpy(
                num_agents,
                max_steps,
                cfg.game_objects,
                cfg.map_builder_objects,
                dir,
                rng,
                num_instances,
            )

        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            terrain=terrain,
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

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        # First, check if we should generate an impossible task
        if (
            self.config.impossible_tasks
            and rng.random() < self.config.impossible_task_probability
        ):
            impossible_type = rng.choice(self.config.impossible_tasks)
            return self._generate_impossible_task(
                task_id, rng, num_instances, impossible_type
            )

        # Otherwise, generate a normal assembly line task
        return self._generate_normal_task(task_id, rng, num_instances)

    def _generate_normal_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        """Generate a normal (possible) assembly line task."""
        # Use parent class _setup_task to get all parameters
        (
            num_agents,
            resources,
            num_sinks,
            room_size,
            terrain,
            width,
            height,
            max_steps,
            position,
            chest_position,
            num_chests,
        ) = self._setup_task(rng)

        dir = (
            f"./train_dir/{self.config.map_dir}/{room_size}/{len(resources)}chain/{num_sinks}sinks/{terrain}"
            if self.config.map_dir is not None
            else None
        )

        icl_env = self._make_env_cfg(
            num_agents=num_agents,
            resources=resources,
            num_sinks=num_sinks,
            width=width,
            height=height,
            position=position,
            chest_position=chest_position,
            num_chests=num_chests,
            terrain=terrain,
            max_steps=max_steps,
            num_instances=num_instances or 24 // num_agents,
            rng=rng,
            dir=dir,
        )

        icl_env.label = f"{room_size}_{len(resources)}chain_{num_sinks}sinks_{terrain}"
        return icl_env

    def _generate_impossible_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
        impossible_type: str = "deterministic",
    ) -> MettaGridConfig:
        """Generate an impossible task (deterministic or noisy).

        These tasks are standalone and not combinatorial with normal task parameters.
        """
        # Sample basic parameters for the impossible task
        num_agents = rng.choice(self.config.num_agents)

        # Sample resources for the impossible task chain
        num_resources = rng.choice(self.config.num_resources)
        if num_resources == 0:
            resources = []
        else:
            num_resources = max(1, min(num_resources, len(self.resource_types)))
            resources = rng.sample(self.resource_types, num_resources)

        num_sinks = rng.choice(self.config.num_converters)

        # Use a random room size for dimensions
        room_size = rng.choice(["small", "medium", "large"])
        position = rng.choice(self.config.positions)
        num_chests = rng.choice(self.config.num_chests)
        chest_position = rng.choice(self.config.chest_positions)

        # Get terrain and dimensions using the chosen room_size
        original_room_sizes = self.config.room_sizes
        try:
            self.config.room_sizes = [room_size]
            (
                _,  # num_agents
                _,  # resources
                _,  # num_sinks
                _,  # room_size
                terrain,
                width,
                height,
                max_steps,
                _,  # position
                _,  # chest_position
                _,  # num_chests
            ) = self._setup_task(rng)
        finally:
            self.config.room_sizes = original_room_sizes

        # Generate the impossible task based on type
        if impossible_type == "deterministic":
            return self._generate_impossible_deterministic_task_internal(
                num_agents,
                resources,
                num_sinks,
                room_size,
                terrain,
                width,
                height,
                max_steps,
                position,
                chest_position,
                num_chests,
                num_instances or 24 // num_agents,
                rng,
            )
        elif impossible_type == "noisy":
            return self._generate_impossible_noisy_task_internal(
                num_agents,
                resources,
                num_sinks,
                room_size,
                terrain,
                width,
                height,
                max_steps,
                position,
                chest_position,
                num_chests,
                num_instances or 24 // num_agents,
                rng,
            )
        else:
            raise ValueError(f"Unknown impossible task type: {impossible_type}")

    def generate_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        """Generate a task. Handles both normal and impossible task variants."""
        return self._generate_task(task_id, rng, num_instances)

    def _generate_impossible_deterministic_task_internal(
        self,
        num_agents: int,
        resources: list[str],
        num_sinks: int,
        room_size: str,
        terrain: str,
        width: int,
        height: int,
        max_steps: int,
        position: list[Position],
        chest_position: list[Position],
        num_chests: int,
        num_instances: int,
        rng: random.Random,
    ) -> MettaGridConfig:
        """Generate an impossible task that always returns reward=0."""
        cfg = _BuildCfg()

        # Create impossible chain requiring unobtainable resource
        self._make_impossible_deterministic_chain(
            resources, width + height / 2, position, cfg, rng
        )
        self._make_sinks(num_sinks, position, cfg, rng)
        if num_chests > 0:
            self._make_chests(num_chests, cfg, chest_position)

        icl_env = make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            terrain=terrain,
        )

        # Add unobtainium to resource_names for impossible task
        if "unobtainium" not in icl_env.game.resource_names:
            icl_env.game.resource_names.append("unobtainium")

        icl_env.label = "impossible_deterministic"
        return icl_env

    def _generate_impossible_noisy_task_internal(
        self,
        num_agents: int,
        resources: list[str],
        num_sinks: int,
        room_size: str,
        terrain: str,
        width: int,
        height: int,
        max_steps: int,
        position: list[Position],
        chest_position: list[Position],
        num_chests: int,
        num_instances: int,
        rng: random.Random,
    ) -> MettaGridConfig:
        """Generate an impossible task with noisy returns but no learnable pattern."""
        cfg = _BuildCfg()

        # Create noisy impossible chain with inconsistent rules
        self._make_impossible_noisy_chain(
            resources, width + height / 2, position, cfg, rng
        )
        self._make_sinks(num_sinks, position, cfg, rng)
        if num_chests > 0:
            self._make_chests(num_chests, cfg, chest_position)

        icl_env = make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            terrain=terrain,
        )

        # Note: Noisy impossible tasks don't use unobtainium, but ensure resource_names are valid
        icl_env.label = "impossible_noisy"
        return icl_env


def train(
    curriculum_style: str = "train",
) -> TrainTool:
    task_generator_cfg = make_task_generator_cfg(
        **curriculum_args[curriculum_style], map_dir=None
    )
    from experiments.evals.in_context_learning.assemblers.assembly_lines import (
        make_assembly_line_eval_suite,
    )

    return train_icl(task_generator_cfg, make_assembly_line_eval_suite)


def play(curriculum_style: str = "test") -> PlayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    return play_icl(task_generator)


def play_eval() -> PlayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(
            num_agents=[2],
            chain_lengths=[5],
            num_sinks=[2],
            room_sizes=["large"],
            positions=[["Any", "Any"]],
        )
    )
    return play_icl(task_generator)


def replay(
    curriculum_style: str = "hard_eval",
) -> ReplayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    # Default to the research policy if none specified
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_4.2.2025-09-24/icl_resource_chain_terrain_4.2.2025-09-24:v2370.pt"
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_1.2.2025-09-24/icl_resource_chain_terrain_1.2.2025-09-24:v2070.pt"
    return replay_icl(task_generator, default_policy_uri)


def evaluate():
    from experiments.evals.in_context_learning.assemblers.assembly_lines import (
        make_assembly_line_eval_suite,
    )

    policy_uris = []

    for curriculum_style in curriculum_args:
        policy_uri = f"s3://softmax-public/policies/in_context.assembly_lines_{curriculum_style}.eval_local.2025-09-27/:latest"
        policy_uris.append(policy_uri)

    simulations = make_assembly_line_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.assemblers.assembly_lines.train",
                f"run=in_context.assembly_lines_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
