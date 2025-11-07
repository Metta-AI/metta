import random
import subprocess
import time
import dataclasses
import typing

import metta.agent.policies.vit_reset
import metta.cogworks.curriculum.curriculum
import metta.cogworks.curriculum.learning_progress_algorithm
import metta.cogworks.curriculum.task_generator
import metta.rl.training
import metta.sim.simulation_config
import metta.tools.play
import metta.tools.replay
import metta.tools.train
import mettagrid.builder
import mettagrid.builder.envs
import mettagrid.config.mettagrid_config

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
    "generator_red": mettagrid.builder.empty_assemblers.generator_red,
    "generator_blue": mettagrid.builder.empty_assemblers.generator_blue,
    "generator_green": mettagrid.builder.empty_assemblers.generator_green,
    "mine_red": mettagrid.builder.empty_assemblers.mine_red,
    "mine_blue": mettagrid.builder.empty_assemblers.mine_blue,
    "mine_green": mettagrid.builder.empty_assemblers.mine_green,
    "altar": mettagrid.builder.empty_assemblers.altar,
    "factory": mettagrid.builder.empty_assemblers.factory,
    "temple": mettagrid.builder.empty_assemblers.temple,
    "armory": mettagrid.builder.empty_assemblers.armory,
    "lab": mettagrid.builder.empty_assemblers.lab,
    "lasery": mettagrid.builder.empty_assemblers.lasery,
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


@dataclasses.dataclass
class _BuildCfg:
    used_objects: list[str] = dataclasses.field(default_factory=list)
    game_objects: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    map_builder_objects: dict[str, int] = dataclasses.field(default_factory=dict)
    input_resources: set[str] = dataclasses.field(default_factory=set)


class AssemblyLinesTaskGenerator(metta.cogworks.curriculum.task_generator.TaskGenerator):
    def __init__(self, config: "AssemblyLinesTaskGenerator.Config"):
        super().__init__(config)
        self.assembler_types = ASSEMBLER_TYPES.copy()
        self.resource_types = RESOURCE_TYPES.copy()
        self.config = config

    class Config(metta.cogworks.curriculum.task_generator.TaskGeneratorConfig["AssemblyLinesTaskGenerator"]):
        chain_lengths: list[int]
        num_sinks: list[int]
        room_sizes: list[str]
        terrains: list[str]

    def _choose_assembler_name(
        self, pool: dict[str, typing.Any], used: set[str], rng: random.Random
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

        protocol = mettagrid.config.mettagrid_config.ProtocolConfig(
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
    ) -> mettagrid.config.mettagrid_config.MettaGridConfig:
        cfg = _BuildCfg()

        self._make_resource_chain(chain_length, width + height / 2, cfg, rng)
        self._make_sinks(num_sinks, cfg, rng)

        return mettagrid.builder.envs.make_assembly_lines(
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
        num_instances: typing.Optional[int] = None,
    ) -> mettagrid.config.mettagrid_config.MettaGridConfig:
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


def train(
    curriculum_style: str = "level_0",
) -> metta.tools.train.TrainTool:
    import experiments.evals.assembly_lines

    task_generator_cfg = make_task_generator_cfg(**curriculum_args[curriculum_style])
    curriculum = metta.cogworks.curriculum.curriculum.CurriculumConfig(
        task_generator=task_generator_cfg, algorithm_config=metta.cogworks.curriculum.learning_progress_algorithm.LearningProgressConfig()
    )

    policy_config = metta.agent.policies.vit_reset.ViTResetConfig()

    return metta.tools.train.TrainTool(
        training_env=metta.rl.training.TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=metta.rl.training.EvaluatorConfig(simulations=experiments.evals.assembly_lines.make_assembly_line_eval_suite()),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_mettagrid(task_generator: AssemblyLinesTaskGenerator) -> mettagrid.config.mettagrid_config.MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "level_0") -> metta.tools.play.PlayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    return metta.tools.play.PlayTool(
        sim=metta.sim.simulation_config.SimulationConfig(
            env=make_mettagrid(task_generator), suite="assembly_lines", name="play"
        )
    )


def replay(
    curriculum_style: str = "level_0",
) -> metta.tools.replay.ReplayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    # Default to the research policy if none specified
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_1.2.2025-09-24/icl_resource_chain_terrain_1.2.2025-09-24:v2070.pt"
    return metta.tools.replay.ReplayTool(
        sim=metta.sim.simulation_config.SimulationConfig(
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
