import random
from typing import Optional

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.mettagrid.config import building
from metta.mettagrid.config.envs import make_icl_resource_chain
from metta.mettagrid.mettagrid_config import EnvConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from pydantic import Field

CONVERTER_TYPES = {
    "mine_red": building.mine_red,
    "mine_blue": building.mine_blue,
    "mine_green": building.mine_green,
    "generator_red": building.generator_red,
    "generator_blue": building.generator_blue,
    "generator_green": building.generator_green,
    "altar": building.altar,
    "lab": building.lab,
    # leave these out for evals
    # "factory": building.factory,
    # "temple": building.temple,
    # "lasery": building.lasery,
}

RESOURCE_TYPES = [
    "ore_red",
    "ore_blue",
    "ore_green",
    "battery_red",
    "battery_blue",
    "battery_green",
    # leave these out for evals
    # "laser",
    # "blueprint",
    # "armor",
]


class InContextCfg:
    """Configuration for InContextResourceChainEnv."""

    def __init__(self):
        self.used_objects = []
        self.all_input_resources = []
        self.converters = []
        self.game_objects = {}
        self.map_builder_objects = {}


class ConverterChainTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["ConverterChainTaskGenerator"]):
        """Configuration for ConverterChainTaskGenerator."""

        chain_lengths: list[int] = Field(
            default_factory=list, description="Chain lengths to sample from"
        )
        num_sinks: list[int] = Field(
            default_factory=list, description="Number of sinks to sample from"
        )

    def __init__(self, config: "ConverterChainTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()

    def set_converter(
        self,
        input_resource,
        output_resource,
        cfg,
        rng,
    ):
        converter_name = str(
            rng.choice([c for c in self.converter_types if c not in cfg.used_objects])
        )
        cfg.used_objects.append(converter_name)
        cfg.converters.append(converter_name)

        converter = self.converter_types[converter_name]
        converter.output_resources = {output_resource: 1}

        if input_resource != "nothing":
            converter.input_resources = {input_resource: 1}

            cfg.all_input_resources.append(input_resource)

        cfg.game_objects[converter_name] = converter
        cfg.map_builder_objects[converter_name] = 1

    def set_sink(self, cfg, rng):
        sink_name = str(
            rng.choice([c for c in self.converter_types if c not in cfg.used_objects])
        )
        cfg.used_objects.append(sink_name)
        sink = self.converter_types[sink_name]

        for input_resource in cfg.all_input_resources:
            sink.input_resources[input_resource] = 1

        cfg.game_objects[sink_name] = sink
        cfg.map_builder_objects[sink_name] = 1

    def make_env(self, resource_chain, num_sinks, rng, max_steps=256) -> EnvConfig:
        cfg = InContextCfg()

        resource_chain = ["nothing"] + list(resource_chain) + ["heart"]

        chain_length = len(resource_chain)

        for i in range(chain_length - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            self.set_converter(input_resource, output_resource, cfg, rng=rng)

        for _ in range(num_sinks):
            self.set_sink(cfg, rng=rng)

        # longer episodes for longer chains
        if len(cfg.used_objects) > 4:
            max_steps = 512

        cooldown = 6 * (chain_length - 1)

        for obj in cfg.converters:
            cfg.game_objects[obj].cooldown = cooldown

        return make_icl_resource_chain(
            num_agents=24,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
        )

    def _generate_task(self, task_id: int, rng: random.Random) -> EnvConfig:
        chain_length = rng.choice(self.config.chain_lengths)
        num_sinks = rng.choice(self.config.num_sinks)
        resource_chain = rng.sample(self.resource_types, chain_length)

        icl_env = self.make_env(resource_chain, num_sinks, rng=rng)

        return icl_env


def make_env() -> EnvConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        chain_lengths=[2, 3, 4, 5],
        num_sinks=[0, 1, 2],
    )
    task_generator = ConverterChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum() -> CurriculumConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        chain_lengths=[2, 3, 4, 5],
        num_sinks=[0, 1, 2],
    )
    task_generator = ConverterChainTaskGenerator(task_generator_cfg)

    return CurriculumConfig(task_generator=task_generator)


def train(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    trainer_cfg = TrainerConfig(
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(
            # simulations=[SimulationConfig(env=task_generator.get_task(0), name="icl")],
        ),
    )

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[EnvConfig] = None) -> PlayTool:
    eval_env = env or make_env()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
    )


def replay(env: Optional[EnvConfig] = None) -> ReplayTool:
    eval_env = env or make_env()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
    )


def eval() -> SimTool:
    return SimTool(
        simulations=[SimulationConfig(env=make_env(), name="in_context_resource_chain")]
    )
