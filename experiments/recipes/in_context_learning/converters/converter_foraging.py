import random
import subprocess
import time
from typing import List, Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.eval import EvalTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_in_context_chains

from experiments.recipes.in_context_learning.in_context_learning import (
    ICLTaskGenerator,
    LPParams,
    _BuildCfg,
    play_icl,
    replay_icl,
    train_icl,
)

curriculum_args = {
    "small": {
        "num_resources": [2, 3, 4],
        "num_converters": [1, 2],
        "room_sizes": ["tiny", "small"],
        "max_recipe_inputs": [1, 2],
    },
    "small_medium": {
        "num_resources": [2, 3, 4],
        "num_converters": [1, 2],
        "room_sizes": ["tiny", "small", "medium"],
        "max_recipe_inputs": [1, 2, 3],
    },
    "all_room_sizes": {
        "num_resources": [3, 4, 5],
        "num_converters": [1, 2],
        "room_sizes": ["tiny", "small", "medium", "large"],
        "max_recipe_inputs": [1, 2, 3],
    },
    "complex_recipes": {
        "num_resources": [2, 3, 4, 5, 6],
        "num_converters": [1, 2, 3],
        "room_sizes": ["tiny", "small", "medium", "large"],
        "max_recipe_inputs": [1, 2, 3, 4],
    },
    "terrain": {
        "num_resources": [2, 3, 4, 5],
        "num_converters": [1, 2, 3],
        "max_recipe_inputs": [1, 2, 3],
        "room_sizes": ["tiny", "small", "medium", "large"],
    },
    # "test": {
    #     "num_resources": [4],
    #     "num_converters": [3],
    #     "room_sizes": ["medium"],
    #     "max_recipe_inputs": [2],
    # },
}


class ConverterForagingTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)

    def calculate_max_steps(
        self, num_altars: int, num_generators: int, width: int, height: int
    ) -> int:
        area = width * height
        max_steps = max(150, area * (num_altars + num_generators) * 2)
        return min(max_steps, 1800)

    def _make_env_cfg(
        self,
        resources: List[str],
        num_converters: int,
        room_size: str,
        width: int,
        height: int,
        density: str,
        rng: random.Random,
        max_input_resources: int,
        max_steps: int = 512,
        # keep explicit overrides if you still want them; otherwise computed below
        source_initial_resource_count: Optional[int] = None,
    ):
        cfg = _BuildCfg()
        avg_hop = width + height / 2

        # 1) add sources for each base resource
        for r in resources:
            self._add_converter(
                input_resources={},
                output_resources={r: 1},
                cfg=cfg,
                rng=rng,
                cooldown=int(avg_hop),
            )

        recipe_sizes: list[int] = []
        for _ in range(num_converters):
            # how many of the resources are used as input to get hearts
            resource_count = rng.randint(1, max_input_resources)
            resources = [
                rng.choice(list(cfg.all_output_resources))
                for _ in range(resource_count)
            ]
            converter_name = self._add_converter(
                input_resources={resource: 1 for resource in resources},
                output_resources={"heart": 1},
                cfg=cfg,
                rng=rng,
                cooldown=int(avg_hop * (1 + 0.5 * resource_count)),
            )
            cfg.converters.append(converter_name)
            recipe_sizes.append(resource_count)

        # build env
        return make_in_context_chains(
            num_agents=24,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            terrain=density,
        )

    def _generate_task(self, task_id: int, rng: random.Random):
        (
            _,
            resources,
            num_converters,
            room_size,
            density,
            width,
            height,
            max_steps,
            _,
        ) = super()._setup_task(rng)

        max_input_resources = rng.choice(self.config.max_recipe_inputs)

        env = self._make_env_cfg(
            resources=resources,
            num_converters=num_converters,
            room_size=room_size,
            width=width,
            height=height,
            density=density,
            rng=rng,
            max_steps=max_steps,
            max_input_resources=max_input_resources,
        )

        env.label = f"{len(resources)}resources_{num_converters}recipes_{room_size}"
        if max_input_resources:
            env.label += f"_maxinputs{max_input_resources}"
        env.label += f"_{density}" if density else ""
        return env


def make_curriculum(
    num_resources=[2, 3, 4],
    num_converters=[1, 2, 3],
    room_sizes=["small"],
    max_recipe_inputs=[1, 2, 3],
    lp_params: LPParams = LPParams(),
) -> CurriculumConfig:
    task_generator_cfg = ConverterForagingTaskGenerator.Config(
        num_resources=num_resources,
        num_converters=num_converters,
        room_sizes=room_sizes,
        max_recipe_inputs=max_recipe_inputs,
    )
    algorithm_config = LearningProgressConfig(**lp_params.__dict__)

    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum_style: str = "small",
    lp_params: LPParams = LPParams(),
) -> TrainTool:
    task_generator_cfg = ConverterForagingTaskGenerator.Config(
        **curriculum_args[curriculum_style]
    )
    from experiments.evals.in_context_learning.converters.converter_foraging import (
        make_unordered_chain_eval_suite,
    )

    return train_icl(task_generator_cfg, make_unordered_chain_eval_suite, lp_params)


def play(curriculum_style: str = "test") -> PlayTool:
    task_generator_cfg = ConverterForagingTaskGenerator.Config(
        **curriculum_args[curriculum_style]
    )
    task_generator = ConverterForagingTaskGenerator(task_generator_cfg)
    return play_icl(task_generator)


def replay(
    curriculum_style: str = "complex_recipes",
    policy_uri: str = "s3://softmax-public/policies/icl_unordered_chain_all_room_sizes_seed456/icl_unordered_chain_all_room_sizes_seed456:v900.pt",
) -> ReplayTool:
    task_generator_cfg = ConverterForagingTaskGenerator.Config(
        **curriculum_args[curriculum_style]
    )
    task_generator = ConverterForagingTaskGenerator(task_generator_cfg)
    return replay_icl(task_generator, policy_uri)


def eval(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> EvalTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.converters.converter_foraging import (
        make_unordered_chain_eval_suite,
    )

    simulations = simulations or make_unordered_chain_eval_suite()
    return EvalTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


# Backward compatibility alias
evaluate = eval


def experiment():
    curriculum_styles = [
        "small",
        "small_medium",
        "all_room_sizes",
        "complex_recipes",
        "terrain",
    ]

    for curriculum_style in curriculum_styles:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.converters.converter_foraging.train",
                f"run=icl_converter_foraging_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
