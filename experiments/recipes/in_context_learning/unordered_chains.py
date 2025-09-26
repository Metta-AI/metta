import random
import subprocess
import time
from typing import List, Optional, Sequence

from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_in_context_chains
from mettagrid.config.mettagrid_config import MettaGridConfig

from experiments.recipes.in_context_learning.icl_resource_chain import (
    ICLTaskGenerator,
    LPParams,
    calculate_avg_hop,
    _BuildCfg,
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
        "obstacle_types": ["square", "cross", "L"],
        "densities": ["", "balanced", "sparse", "high"],
        "max_recipe_inputs": [1, 2, 3],
        "room_sizes": ["tiny", "small", "medium", "large"],
    },
}


class UnorderedChainTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)

    def _add_source(self, output_resource: str, cfg: _BuildCfg, rng: random.Random):
        """
        Source: empty-input converter that emits one unit of `output_resource`.
        """
        name = self._choose_converter_name(
            self.converter_types, set(cfg.used_objects), rng
        )
        cfg.used_objects.append(name)
        cfg.sources.append(name)

        conv = self.converter_types[name].copy()
        conv.input_resources = {}
        conv.output_resources = {output_resource: 1}

        cfg.all_input_resources.append(output_resource)
        cfg.game_objects[name] = conv
        cfg.map_builder_objects[name] = 1

    def _add_recipe_converter(
        self,
        cfg: _BuildCfg,
        rng: random.Random,
        max_input_resources: Optional[int] = 6,
        output_resource: str = "heart",
    ):
        """
        Multiset-input converter:
          - Pick length L in [1, max_input_resources]
          - Sample L items with replacement from cfg.all_input_resources
          - Count them into input_resources
          - Output one `heart`
        """
        assert cfg.all_input_resources, "No resources available to build a recipe from."
        L = rng.randint(1, (max_input_resources or 6))

        # sample with replacement, then count
        picks = [rng.choice(cfg.all_input_resources) for _ in range(L)]
        inputs: dict[str, int] = {}
        for r in picks:
            inputs[r] = inputs.get(r, 0) + 1

        name = self._choose_converter_name(
            self.converter_types, set(cfg.used_objects), rng
        )
        cfg.used_objects.append(name)
        cfg.converters.append(name)

        conv = self.converter_types[name].copy()
        conv.input_resources = inputs
        conv.output_resources = {output_resource: 1}

        cfg.game_objects[name] = conv
        cfg.map_builder_objects[name] = 1

        # return L so caller can set cooldown proportional to recipe size
        return L

    def _make_env_cfg(
        self,
        resources: List[str],
        num_converters: int,
        room_size: str,
        width: int,
        height: int,
        obstacle_type: Optional[str],
        density: Optional[str],
        rng: random.Random,
        max_steps: int = 512,
        max_input_resources: Optional[int] = None,
        # keep explicit overrides if you still want them; otherwise computed below
        source_initial_resource_count: Optional[int] = None,
        source_max_conversions: Optional[int] = None,
        source_cooldown: Optional[int] = None,
    ):
        cfg = _BuildCfg()

        # 1) add sources for each base resource
        for r in resources:
            self._add_source(r, cfg, rng=rng)

        # 2) geometry-aware cooldowns
        avg_hop = calculate_avg_hop(room_size)
        default_source_cd = int(avg_hop) if source_cooldown is None else source_cooldown

        for s in cfg.sources:
            src = cfg.game_objects[s]
            if source_initial_resource_count is not None:
                src.initial_resource_count = source_initial_resource_count
            if source_max_conversions is not None:
                src.max_conversions = source_max_conversions
            src.cooldown = default_source_cd

        # 3) add N multiset recipe converters producing hearts
        recipe_sizes: list[int] = []
        for _ in range(num_converters):
            L = self._add_recipe_converter(
                cfg, rng=rng, max_input_resources=max_input_resources
            )
            recipe_sizes.append(L)

        # 4) set per-recipe cooldown ~ avg_hop and the recipe length
        for name, L in zip(cfg.converters, recipe_sizes):
            cfg.game_objects[name].cooldown = int(avg_hop * (1 + 0.5 * L))

        # 5) build env
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
        (
            resources,
            num_converters,
            room_size,
            obstacle_type,
            density,
            width,
            height,
            max_input_resources,
        ) = super()._setup_task(rng)

        # hardcode this for now
        max_steps = 512

        env = self._make_env_cfg(
            resources=resources,
            num_converters=num_converters,
            room_size=room_size,
            width=width,
            height=height,
            obstacle_type=obstacle_type,
            density=density,
            rng=rng,
            max_steps=max_steps,
            max_input_resources=max_input_resources,
            # optional explicit overrides still honored
            source_initial_resource_count=cfg.source_initial_resource_count,
            source_max_conversions=cfg.source_max_conversions,
            source_cooldown=cfg.source_cooldown,
        )

        env.label = f"{len(resources)}resources_{num_converters}recipes_{room_size}"
        if max_input_resources:
            env.label += f"_maxinputs{max_input_resources}"
        env.label += "_terrain" if obstacle_type else ""
        env.label += f"_{density}" if density else ""
        return env


def make_mettagrid(curriculum_style: str = "complex_recipes") -> MettaGridConfig:
    task_generator_cfg = ICLTaskGenerator.Config(**curriculum_args[curriculum_style])
    task_generator = UnorderedChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum(
    num_resources=[2, 3, 4],
    num_converters=[1, 2, 3],
    room_sizes=["small"],
    obstacle_types=[],
    densities=[],
    max_recipe_inputs=[1, 2, 3],
    lp_params: LPParams = LPParams(),
) -> CurriculumConfig:
    task_generator_cfg = UnorderedChainTaskGenerator.Config(
        num_resources=num_resources,
        num_converters=num_converters,
        room_sizes=room_sizes,
        obstacle_types=obstacle_types,
        densities=densities,
        max_recipe_inputs=max_recipe_inputs,
    )
    algorithm_config = LearningProgressConfig(**lp_params.__dict__)

    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum: Optional[CurriculumConfig] = None,
    curriculum_style: str = "small",
    lp_params: LPParams = LPParams(),
) -> TrainTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.unordered_chains import (
        make_unordered_chain_eval_suite,
    )

    curriculum = make_curriculum(
        **curriculum_args[curriculum_style], lp_params=lp_params
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    policy_config = FastLSTMResetConfig()

    training_env_cfg = TrainingEnvironmentConfig(curriculum=curriculum)
    return TrainTool(
        trainer=trainer_cfg,
        training_env=training_env_cfg,
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(
            simulations=make_unordered_chain_eval_suite(),
            evaluate_remote=True,
            evaluate_local=False,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def play(
    env: Optional[MettaGridConfig] = None, curriculum_style: str = "complex_recipes"
) -> PlayTool:
    eval_env = env or make_mettagrid(curriculum_style)
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_unordered_resource_chain",
            suite="in_context_learning",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None) -> ReplayTool:
    eval_env = env or make_mettagrid()
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_unordered_resource_chain",
            suite="in_context_learning",
        ),
        policy_uri="s3://softmax-public/policies/icl_unordered_chain_all_room_sizes_seed456/icl_unordered_chain_all_room_sizes_seed456:v900.pt",
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    # Local import to   avoid circular import at module load time
    from experiments.evals.in_context_learning.unordered_chains import (
        make_unordered_chain_eval_suite,
    )

    simulations = simulations or make_unordered_chain_eval_suite()
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

    for curriculum_style in curriculum_styles:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.unordered_chains.train",
                f"run=icl_unordered_chain_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
