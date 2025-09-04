import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.mettagrid.builder import empty_converters
from metta.mettagrid.builder.envs import make_icl_resource_chain
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from pydantic import Field

CONVERTER_TYPES = {
    "mine_red": empty_converters.mine_red,
    "mine_blue": empty_converters.mine_blue,
    "mine_green": empty_converters.mine_green,
    "generator_red": empty_converters.generator_red,
    "generator_blue": empty_converters.generator_blue,
    "generator_green": empty_converters.generator_green,
    "altar": empty_converters.altar,
    "lab": empty_converters.lab,
    "lasery": empty_converters.lasery,
    "factory": empty_converters.factory,
    "temple": empty_converters.temple,
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
    used_objects: List[str] = field(default_factory=list)
    all_input_resources: List[str] = field(default_factory=list)
    converters: List[str] = field(default_factory=list)
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)


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

    def _choose_converter_name(
        self, pool: Dict[str, Any], used: set[str], rng: random.Random
    ) -> str:
        choices = [name for name in pool.keys() if name not in used]
        if not choices:
            raise ValueError("No available converter names left to choose from.")
        return str(rng.choice(choices))

    def _add_converter(
        self,
        input_resource: str,
        output_resource: str,
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        converter_name = self._choose_converter_name(
            self.converter_types, cfg.used_objects, rng
        )
        cfg.used_objects.append(converter_name)
        cfg.converters.append(converter_name)

        converter = self.converter_types[converter_name].copy()
        converter.output_resources = {output_resource: 1}

        if input_resource == "nothing":
            converter.input_resources = {}
        else:
            converter.input_resources = {input_resource: 1}

            cfg.all_input_resources.append(input_resource)

        cfg.game_objects[converter_name] = converter
        cfg.map_builder_objects[converter_name] = 1

    def _add_sink(self, cfg: _BuildCfg, rng: random.Random):
        sink_name = self._choose_converter_name(
            self.converter_types, cfg.used_objects, rng
        )
        cfg.used_objects.append(sink_name)
        sink = self.converter_types[sink_name].copy()

        for input_resource in cfg.all_input_resources:
            sink.input_resources[input_resource] = 1

        cfg.game_objects[sink_name] = sink
        cfg.map_builder_objects[sink_name] = 1

    def _make_env_cfg(
        self, resource_chain, num_sinks, rng, max_steps=256
    ) -> MettaGridConfig:
        cfg = _BuildCfg()
        resource_chain = ["nothing"] + list(resource_chain) + ["heart"]

        chain_length = len(resource_chain)

        for i in range(chain_length - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            self._add_converter(input_resource, output_resource, cfg, rng=rng)

        for _ in range(num_sinks):
            self._add_sink(cfg, rng=rng)

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

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        chain_length = rng.choice(self.config.chain_lengths)
        num_sinks = rng.choice(self.config.num_sinks)
        resource_chain = rng.sample(self.resource_types, chain_length)

        icl_env = self._make_env_cfg(resource_chain, num_sinks, rng=rng)

        return icl_env


def make_mettagrid() -> MettaGridConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        chain_lengths=[6],
        num_sinks=[2],
    )
    task_generator = ConverterChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum() -> CurriculumConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        chain_lengths=[2, 3, 4, 5],
        num_sinks=[0, 1, 2],
    )
    return CurriculumConfig(task_generator=task_generator_cfg)


def train(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.icl_resource_chain import (
        make_icl_resource_chain_eval_suite,
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(simulations=make_icl_resource_chain_eval_suite()),
    )
    # for in context learning, we need episode length to be equal to bptt_horizon
    # which requires a large batch size
    trainer_cfg.batch_size = 2064384
    trainer_cfg.bptt_horizon = 256

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
        policy_uri="wandb://run/georgedeane.operant_conditioning.in_context_learning.all.0.1_progress_smoothing.08-19:v50",
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.icl_resource_chain import (
        make_icl_resource_chain_eval_suite,
    )

    simulations = simulations or make_icl_resource_chain_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )
