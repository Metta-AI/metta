import random
import subprocess
import time
from bisect import bisect_left
from math import comb
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
    _BuildCfg,
)

curriculum_args = {
    "small": {
        "num_resources": [2, 3, 4],
        "num_converters": [1, 2],
        "room_sizes": ["small"],
        "max_recipe_inputs": [1, 2],
    },
    "small_medium": {
        "num_resources": [2, 3, 4],
        "num_converters": [1, 2, 3],
        "room_sizes": ["small", "medium"],
        "max_recipe_inputs": [1, 2, 3],
    },
    "all_room_sizes": {
        "num_resources": [3, 4, 5],
        "num_converters": [1, 2, 3],
        "room_sizes": ["small", "medium", "large"],
        "max_recipe_inputs": [1, 2, 3],
    },
    "complex_recipes": {
        "num_resources": [4, 5, 6],
        "num_converters": [2, 3, 4],
        "room_sizes": ["small", "medium", "large"],
        "max_recipe_inputs": [2, 3, 4],
    },
    "terrain": {
        "num_resources": [3, 4, 5],
        "num_converters": [1, 2, 3],
        "obstacle_types": ["square", "cross", "L"],
        "densities": ["", "balanced", "sparse", "high"],
        "max_recipe_inputs": [1, 2, 3],
        "room_sizes": ["small", "medium", "large"],
    },
}


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
        self, cfg: _BuildCfg, rng: random.Random, max_input_resources: Optional[int] = 6
    ):
        output_resource = "heart"  # can think about the output resource later

        # Build a pool that allows duplicates for reusable resources but not for non-reusables
        # non_reusable_resources is now auto-derived from sources with limited supply
        non_reusable: set[str] = set(self.config.non_reusable_resources)
        reusable: List[str] = [
            r for r in cfg.all_input_resources if r not in non_reusable
        ]
        unique_non_reusable: List[str] = list(
            {r for r in cfg.all_input_resources if r in non_reusable}
        )
        if not max_input_resources:
            max_input_resources = 6
        num_input_resources = rng.randint(1, max_input_resources)

        # Keep elegance and efficiency: sample a valid multiset without enumerating all
        def _sample_composition(total: int, parts: int) -> List[int]:
            if parts <= 0:
                return []
            if total == 0:
                return [0] * parts
            # Stars and bars: choose bar positions uniformly to sample a composition
            # over 'parts' non-negative integers summing to 'total'.
            if parts == 1:
                return [total]
            bars = sorted(rng.sample(range(total + parts - 1), parts - 1))
            prev = -1
            counts: List[int] = []
            for b in bars + [total + parts - 1]:
                counts.append(b - prev - 1)
                prev = b
            return counts

        def _weighted_choice(weights: List[int]) -> int:
            total = sum(weights)
            if total == 0:
                return len(weights) - 1
            cumsum: List[int] = []
            acc = 0
            for w in weights:
                acc += w
                cumsum.append(acc)
            x = rng.random() * total
            return bisect_left(cumsum, x)

        if len(non_reusable) == 0:
            # Back-compat: draw a combination with replacement of random length 1..num_input_resources
            L = rng.randint(1, num_input_resources)
            counts = _sample_composition(L, len(reusable))
            reusable_draw_list: List[str] = []
            for typ, c in zip(reusable, counts):
                reusable_draw_list.extend([typ] * c)
            chosen_combo = tuple(sorted(reusable_draw_list))
        else:
            L = rng.randint(1, num_input_resources)
            # Choose how many non-reusables to include (weighted by number of valid multisets)
            nR = len(reusable)
            nNR = len(unique_non_reusable)
            max_m = min(L, nNR)
            weights: List[int] = []
            for m in range(0, max_m + 1):
                r = L - m
                if r < 0:
                    weights.append(0)
                    continue
                if nR == 0 and r > 0:
                    weights.append(0)
                    continue
                # number of ways: choose m distinct non-reusables * compositions of r over nR types
                # C(nNR, m) * C(r + nR - 1, nR - 1)
                ways_nr = comb(nNR, m)
                ways_r = (
                    1
                    if r == 0 and nR >= 0
                    else (comb(r + nR - 1, nR - 1) if nR > 0 else 0)
                )
                weights.append(ways_nr * ways_r)
            # Fallback selection is handled in _weighted_choice
            m = _weighted_choice(weights)
            r = max(0, L - m)
            # Sample m distinct non-reusables
            chosen_nr = rng.sample(unique_non_reusable, m) if m > 0 else []
            # Sample composition for reusables
            counts = _sample_composition(r, nR)
            reusable_draw_list = list(chosen_nr)
            for typ, c in zip(reusable, counts):
                reusable_draw_list.extend([typ] * c)
            chosen_combo = tuple(sorted(reusable_draw_list))
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
        max_input_resources=None,
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
            self._add_converter(cfg, rng=rng, max_input_resources=max_input_resources)

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

        # Reuse superclass to sample common env geometry
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

        icl_env = self._make_env_cfg(
            resources,
            num_converters,
            width=width,
            height=height,
            obstacle_type=obstacle_type,
            density=density,
            rng=rng,
            max_steps=self.config.max_steps,
            max_input_resources=max_input_resources,
            source_initial_resource_count=cfg.source_initial_resource_count,
            source_max_conversions=cfg.source_max_conversions,
            source_cooldown=cfg.source_cooldown,
        )

        icl_env.label = (
            f"{len(resources)}resources_{num_converters}converters_{room_size}"
        )
        if max_input_resources:
            icl_env.label += f"_maxinputs{max_input_resources}"
        icl_env.label += "_terrain" if obstacle_type else ""
        icl_env.label += f"_{density}" if density else ""
        return icl_env


def make_mettagrid() -> MettaGridConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[4],
        num_converters=[2],
        room_sizes=["large"],
        obstacle_types=["cross"],
        densities=["high"],
        max_recipe_inputs=[3],
    )
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


def play(env: Optional[MettaGridConfig] = None) -> PlayTool:
    eval_env = env or make_mettagrid()
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
