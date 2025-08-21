import numpy as np
from typing import Optional
import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import ValueRange as vr
from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from metta.mettagrid.config import building

CONVERTER_TYPES = {
    "mine_red": building.mine_red,
    "mine_blue": building.mine_blue,
    "mine_green": building.mine_green,
    "generator_red": building.generator_red,
    "generator_blue": building.generator_blue,
    "generator_green": building.generator_green,
    "altar": building.altar,
    "lab": building.lab,
    "factory": building.factory,
    "temple": building.temple,
    "lasery": building.lasery,
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


class InContextResourceChain:
    def __init__(self, resource_chain, num_sinks):
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()
        self.num_sinks = num_sinks
        self.resource_chain = resource_chain
        self.converters = []
        self.used_objects = []
        self.all_input_resources = []

        self.map_builder_objects = {}

        self.game_objects = {}

        self.max_steps = 256

    def set_converter(self, input_resource, output_resource):
        converter_name = str(
            np.random.choice(
                [c for c in self.converter_types if c not in self.used_objects]
            )
        )
        self.used_objects.append(converter_name)
        self.converters.append(converter_name)

        converter = self.converter_types[converter_name]
        converter.output_resources = {output_resource: 1}

        if input_resource != "nothing":
            converter.input_resources = {input_resource: 1}

            self.all_input_resources.append(input_resource)

        self.game_objects[converter_name] = converter
        self.map_builder_objects[converter_name] = 1

    def set_sink(self):
        sink_name = str(
            np.random.choice(
                [c for c in self.converter_types if c not in self.used_objects]
            )
        )
        self.used_objects.append(sink_name)
        sink = self.converter_types[sink_name]

        for input_resource in self.all_input_resources:
            sink.input_resources[input_resource] = 1

        self.game_objects[sink_name] = sink
        self.map_builder_objects[sink_name] = 1

    def make_env(self):
        self.used_objects = []
        self.all_input_resources = []

        resource_chain = ["nothing"] + list(self.resource_chain) + ["heart"]

        chain_length = len(resource_chain)

        for i in range(chain_length - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            self.set_converter(input_resource, output_resource)

        for _ in range(self.num_sinks):
            self.set_sink()

        # longer episodes for longer chains
        if len(self.used_objects) > 4:
            self.max_steps = 512

        cooldown = 6 * (chain_length - 1)

        for obj in self.converters:
            self.game_objects[obj].cooldown = cooldown

        return self.game_objects, self.map_builder_objects


def make_env() -> EnvConfig:
    return InContextResourceChain(
        resource_chain=[
            "ore_red",
            "ore_blue",
            "ore_green",
            "battery_red",
            "battery_blue",
            "battery_green",
            "laser",
            "blueprint",
            "armor",
        ],
        num_sinks=1,
    ).make_env()


def make_curriculum(nav_env: Optional[EnvConfig] = None) -> CurriculumConfig:
    nav_env = make_env()

    # make a set of training tasks for navigation
    dense_tasks = cc.bucketed(nav_env)

    maps = ["terrain_maps_nohearts"]
    for size in ["large", "medium", "small"]:
        for terrain in ["balanced", "maze", "sparse", "dense", "cylinder-world"]:
            maps.append(f"varied_terrain/{terrain}_{size}")

    dense_tasks.add_bucket("game.map_builder.instance_map.dir", maps)
    dense_tasks.add_bucket(
        "game.map_builder.instance_map.objects.altar", [vr.vr(3, 50)]
    )

    sparse_nav_env = nav_env.model_copy()
    sparse_nav_env.game.map_builder = RandomMapBuilder.Config(
        agents=4,
        objects={"altar": 10},
    )
    sparse_tasks = cc.bucketed(sparse_nav_env)
    sparse_tasks.add_bucket("game.map_builder.width", [vr.vr(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.height", [vr.vr(60, 120)])
    sparse_tasks.add_bucket("game.map_builder.objects.altar", [vr.vr(1, 10)])

    nav_tasks = cc.merge([dense_tasks, sparse_tasks])

    return nav_tasks.to_curriculum()


def train(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            simulations=make_env(),
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="navigation",
        ),
    )


def eval() -> SimTool:
    return SimTool(simulations=make_env())
