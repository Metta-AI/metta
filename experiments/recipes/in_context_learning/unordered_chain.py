import itertools
import random
from typing import List, Optional, Sequence

from .icl_resource_chain import (
    ICLTaskGenerator,
    _BuildCfg,
    CONVERTER_TYPES,
    RESOURCE_TYPES,
)
from metta.mettagrid.builder.envs import make_icl_resource_chain
from metta.mettagrid.builder.envs import MettaGridConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from metta.sim.simulation_config import SimulationConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig


class UnorderedChainTaskGenerator(ICLTaskGenerator):
    # can always add sinks in later

    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()

    def _add_source(self, output_resource: str, cfg: _BuildCfg, rng: random.Random):
        source_name = self._choose_converter_name(
            self.converter_types, set(cfg.used_objects), rng
        )
        cfg.used_objects.append(source_name)
        cfg.sources.append(source_name)

        converter = self.converter_types[source_name].copy()

        converter.output_resources = {output_resource: 1}
        converter.input_resources = {}
        cfg.all_input_resources.append(output_resource)
        cfg.game_objects[source_name] = converter
        cfg.map_builder_objects[source_name] = 1

    def _add_converter(
        self, cfg: _BuildCfg, rng: random.Random, max_input_resources: int = 6
    ):
        output_resource = "heart"  # can think about the output resource later

        # sample one multiset (combination with replacement) of source resources
        num_input_resources = rng.randint(1, max_input_resources)
        # this implies all sources are reusable, if we have non-reusable sources, we need to flag that
        # and impose a constraint on the number of non-reusable source resources we can have as input
        all_combos = list(
            itertools.combinations_with_replacement(
                cfg.all_input_resources, num_input_resources
            )
        )
        chosen_combo = rng.choice(all_combos) if all_combos else ()
        converter_name = self._choose_converter_name(
            self.converter_types, set(cfg.used_objects), rng
        )
        cfg.used_objects.append(converter_name)
        cfg.converters.append(converter_name)
        converter = self.converter_types[converter_name].copy()
        converter.input_resources = {}
        for resource in chosen_combo:
            converter.input_resources[resource] = (
                converter.input_resources.get(resource, 0) + 1
            )

        converter.output_resources = {output_resource: 1}

        cfg.game_objects[converter_name] = converter
        cfg.map_builder_objects[converter_name] = 1

    def _make_env_cfg(
        self,
        resources: List[str],
        num_converters: int,
        width,
        height,
        obstacle_type,
        density,
        rng,
        max_steps=256,
    ):
        cfg = _BuildCfg()

        for resource in resources:
            self._add_source(resource, cfg, rng=rng)

        for _ in range(num_converters):
            self._add_converter(cfg, rng=rng)

        return make_icl_resource_chain(
            num_agents=24,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            obstacle_type=obstacle_type,
            density=density,
        )

    def _generate_task(self, task_id: int, rng: random.Random):
        resources, num_converters, room_size, obstacle_type, density, width, height = (
            self._setup_task(rng)
        )

        icl_env = self._make_env_cfg(
            resources,
            num_converters,
            width=width,
            height=height,
            obstacle_type=obstacle_type,
            density=density,
            rng=rng,
            max_steps=self.config.max_steps,
        )

        icl_env.label = f"{len(resources)}resources_{num_converters}converters_{room_size}_{obstacle_type}_{density}"
        return icl_env


def make_mettagrid() -> MettaGridConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[3],
        num_sinks=[1],
        room_sizes=["large"],
        obstacle_types=["cross"],
        densities=["high"],
    )
    task_generator = UnorderedChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum(
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[LearningProgressConfig] = None,
    chain_lengths=[2, 3, 4, 5],
    num_sinks=[0, 1, 2],
    room_sizes=["small"],
    obstacle_types=[],
    densities=[],
) -> CurriculumConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=chain_lengths,
        num_sinks=num_sinks,
        room_sizes=room_sizes,
        obstacle_types=obstacle_types,
        densities=densities,
    )
    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,  # Enable bidirectional learning progress by default
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=3,
            progress_smoothing=0.1,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum: Optional[CurriculumConfig] = None,
) -> TrainTool:
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
        policy_uri="wandb://run/george.icl.reproduce.4gpus.09-12",
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    # Local import to   avoid circular import at module load time
    from experiments.evals.icl_resource_chain import (
        make_icl_resource_chain_eval_suite,
    )

    simulations = simulations or make_icl_resource_chain_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )
