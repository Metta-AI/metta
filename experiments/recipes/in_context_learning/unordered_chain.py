import itertools
import random
import subprocess
import time
from typing import List, Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training.evaluator import EvaluatorConfig
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_in_context_chains
from mettagrid.config.mettagrid_config import MettaGridConfig

from .icl_resource_chain import (
    ICLTaskGenerator,
    LPParams,
    _BuildCfg,
)


class UnorderedChainTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)

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
        self,
        cfg: _BuildCfg,
        rng: random.Random,
        max_recipe_inputs: Optional[int] = None,
    ):
        output_resource = "heart"  # can think about the output resource later

        # sample one multiset (combination with replacement) of source resources
        max_inputs = max_recipe_inputs if max_recipe_inputs else 6
        num_input_resources = rng.randint(1, max_inputs)
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
        max_steps=512,
        max_recipe_inputs=None,
        source_initial_resource_count=None,
        source_max_conversions=None,
        source_cooldown=25,
    ):
        cfg = _BuildCfg()

        for resource in resources:
            self._add_source(resource, cfg, rng=rng)

        # Configure source properties if specified
        if (
            source_initial_resource_count is not None
            or source_max_conversions is not None
        ):
            for source_name in cfg.sources:
                source = cfg.game_objects[source_name]
                if source_initial_resource_count is not None:
                    source.initial_resource_count = source_initial_resource_count
                if source_max_conversions is not None:
                    source.max_conversions = source_max_conversions
                source.cooldown = source_cooldown

        for _ in range(num_converters):
            self._add_converter(cfg, rng=rng, max_recipe_inputs=max_recipe_inputs)

        return make_in_context_chains(
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
        cfg = self.config

        # For unordered chains, use num_resources and num_sinks as converters
        if cfg.num_resources:
            num_resources = rng.choice(cfg.num_resources)
            resources = rng.sample(self.resource_types, num_resources)
        else:
            # Fallback to chain_lengths for compatibility
            num_resources = rng.choice(cfg.chain_lengths) if cfg.chain_lengths else 3
            resources = rng.sample(self.resource_types, num_resources)

        num_converters = rng.choice(cfg.num_sinks) if cfg.num_sinks else 1
        room_size = rng.choice(cfg.room_sizes)
        obstacle_type = (
            rng.choice(cfg.obstacle_types) if len(cfg.obstacle_types) > 0 else None
        )
        density = rng.choice(cfg.densities) if len(cfg.densities) > 0 else None

        max_recipe_inputs = (
            rng.choice(cfg.max_recipe_inputs) if cfg.max_recipe_inputs else None
        )

        size_range = (
            (8, 12)
            if room_size == "medium"
            else (12, 15)
            if room_size == "large"
            else (5, 8)
        )

        width, height = (
            rng.randint(size_range[0], size_range[1]),
            rng.randint(size_range[0], size_range[1]),
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
            max_recipe_inputs=max_recipe_inputs,
            source_initial_resource_count=cfg.source_initial_resource_count,
            source_max_conversions=cfg.source_max_conversions,
            source_cooldown=cfg.source_cooldown,
        )

        icl_env.label = (
            f"{len(resources)}resources_{num_converters}converters_{room_size}"
        )
        if max_recipe_inputs:
            icl_env.label += f"_maxinputs{max_recipe_inputs}"
        icl_env.label += "_terrain" if obstacle_type else ""
        icl_env.label += f"_{density}" if density else ""
        return icl_env


def make_mettagrid() -> MettaGridConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[4],
        num_sinks=[2],  # num_converters
        room_sizes=["large"],
        obstacle_types=["cross"],
        densities=["high"],
        max_recipe_inputs=[3],
    )
    task_generator = UnorderedChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum(
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[LearningProgressConfig] = None,
    num_resources=[2, 3, 4],
    num_converters=[1, 2, 3],
    room_sizes=["small"],
    obstacle_types=[],
    densities=[],
    max_recipe_inputs=[1, 2, 3],
    lp_params: LPParams = LPParams(),
) -> CurriculumConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=num_resources,
        num_sinks=num_converters,  # num_sinks used as num_converters
        room_sizes=room_sizes,
        obstacle_types=obstacle_types,
        densities=densities,
        max_recipe_inputs=max_recipe_inputs,
    )
    if algorithm_config is None:
        # Use LPParams to configure learning progress algorithm, matching ordered_chains
        params = LPParams(
            enable_detailed_slice_logging=enable_detailed_slice_logging,
            ema_timescale=lp_params.ema_timescale,
            exploration_bonus=lp_params.exploration_bonus,
            max_memory_tasks=lp_params.max_memory_tasks,
            max_slice_axes=lp_params.max_slice_axes,
            progress_smoothing=lp_params.progress_smoothing,
            num_active_tasks=lp_params.num_active_tasks,
            rand_task_rate=lp_params.rand_task_rate,
        )
        algorithm_config = LearningProgressConfig(**params.__dict__)

    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def small_curriculum():
    return make_curriculum(
        num_resources=[2, 3, 4],
        num_converters=[1, 2],
        room_sizes=["small"],
        max_recipe_inputs=[1, 2],
    )


def small_medium_curriculum():
    return make_curriculum(
        num_resources=[2, 3, 4],
        num_converters=[1, 2, 3],
        room_sizes=["small", "medium"],
        max_recipe_inputs=[1, 2, 3],
    )


def all_room_sizes_curriculum():
    return make_curriculum(
        num_resources=[3, 4, 5],
        num_converters=[1, 2, 3],
        room_sizes=["small", "medium", "large"],
        max_recipe_inputs=[1, 2, 3],
    )


def complex_recipes():
    return make_curriculum(
        num_resources=[4, 5, 6],
        num_converters=[2, 3, 4],
        room_sizes=["small", "medium", "large"],
        max_recipe_inputs=[2, 3, 4],
    )


def many_converters():
    return make_curriculum(
        num_resources=[4, 5, 6],
        num_converters=[3, 4, 5],
        room_sizes=["small", "medium", "large"],
        max_recipe_inputs=[2, 3, 4, 5],
    )


def terrain():
    return make_curriculum(
        num_resources=[3, 4, 5],
        num_converters=[1, 2, 3],
        room_sizes=["small", "medium", "large"],
        obstacle_types=["square", "cross", "L"],
        densities=["balanced", "sparse", "high"],
        max_recipe_inputs=[1, 2, 3],
    )


def train(
    curriculum: Optional[CurriculumConfig] = None,
    curriculum_style: str = "small",
    lp_params: LPParams = LPParams(),
) -> TrainTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.ordered_chains import (
        make_icl_resource_chain_eval_suite,
    )

    if curriculum is None:
        curriculum_args = {
            "small": {
                "num_resources": [2, 3, 4],
                "num_converters": [1, 2],
                "room_sizes": ["small"],
                "max_recipe_inputs": [1, 2],
                "lp_params": lp_params,
            },
            "small_medium": {
                "num_resources": [2, 3, 4],
                "num_converters": [1, 2, 3],
                "room_sizes": ["small", "medium"],
                "max_recipe_inputs": [1, 2, 3],
                "lp_params": lp_params,
            },
            "all_room_sizes": {
                "num_resources": [3, 4, 5],
                "num_converters": [1, 2, 3],
                "room_sizes": ["small", "medium", "large"],
                "max_recipe_inputs": [1, 2, 3],
                "lp_params": lp_params,
            },
            "complex_recipes": {
                "num_resources": [4, 5, 6],
                "num_converters": [2, 3, 4],
                "room_sizes": ["small", "medium", "large"],
                "max_recipe_inputs": [2, 3, 4],
                "lp_params": lp_params,
            },
            "terrain": {
                "num_resources": [3, 4, 5],
                "num_converters": [1, 2, 3],
                "obstacle_types": ["square", "cross", "L"],
                "densities": ["", "balanced", "sparse", "high"],
                "max_recipe_inputs": [1, 2, 3],
                "lp_params": lp_params,
                "room_sizes": ["small", "medium", "large"],
            },
        }
        curriculum = make_curriculum(**curriculum_args[curriculum_style])

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum,
        evaluation=EvaluatorConfig(
            simulations=make_icl_resource_chain_eval_suite(),
            evaluate_remote=True,
            evaluate_local=False,
        ),
    )
    # for in context learning, we need episode length to be equal to bptt_horizon
    # which requires a large batch size
    trainer_cfg.batch_size = 4128768
    trainer_cfg.bptt_horizon = 512

    training_env_cfg = TrainingEnvironmentConfig(curriculum=curriculum)
    return TrainTool(trainer=trainer_cfg, training_env=training_env_cfg)


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
        policy_uri="s3://softmax-public/policies/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19:v960.pt",
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    # Local import to   avoid circular import at module load time
    from experiments.evals.in_context_learning.ordered_chains import (
        make_icl_resource_chain_eval_suite,
    )

    simulations = simulations or make_icl_resource_chain_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    curriculum_styles = [
        "small",
        "small_medium",
        "all_room_sizes",
        "complex_recipes",
        "terrain",
    ]

    pretrained_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19:v960.pt"

    for curriculum_style in curriculum_styles:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.unordered_chain.train",
                f"run=icl_unordered_chain_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.unordered_chain.train",
                f"run=icl_unordered_chain_{curriculum_style}_pretrained.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                f"trainer.initial_policy.uri={pretrained_policy_uri}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )


if __name__ == "__main__":
    experiment()
