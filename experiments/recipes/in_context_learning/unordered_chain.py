import random
from bisect import bisect_left
from math import comb
from typing import List, Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.mettagrid.builder.envs import MettaGridConfig, make_icl_resource_chain
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from .icl_resource_chain import (
    CONVERTER_TYPES,
    RESOURCE_TYPES,
    ICLTaskGenerator,
    _BuildCfg,
)


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
        # Configure regeneration/depletion only if overrides are provided
        if self.config.source_cooldown is not None:
            converter.cooldown = int(self.config.source_cooldown)
        if self.config.source_initial_resource_count is not None:
            converter.initial_resource_count = int(
                self.config.source_initial_resource_count
            )
        if self.config.source_max_conversions is not None:
            converter.max_conversions = int(self.config.source_max_conversions)

        # Derive non-reusable status: resource is non-reusable only if exactly 1 can ever be obtained
        initial_count = (
            converter.initial_resource_count
            if hasattr(converter, "initial_resource_count")
            else 1
        )
        max_conv = (
            converter.max_conversions if hasattr(converter, "max_conversions") else -1
        )

        # Calculate total possible resources from this source:
        # - If max_conversions = -1 (infinite), resource is reusable
        # - If max_conversions = 0 (preload only), total = initial_count
        # - If max_conversions > 0, total = initial_count + max_conversions
        if max_conv == 0:
            total_available = initial_count
        elif max_conv > 0:
            total_available = initial_count + max_conv
        else:
            total_available = float("inf")  # Infinite regeneration

        # Non-reusable only if exactly 1 resource can ever be obtained
        if total_available == 1:
            if output_resource not in self.config.non_reusable_resources:
                self.config.non_reusable_resources.append(output_resource)

        cfg.all_input_resources.append(output_resource)
        cfg.game_objects[source_name] = converter
        cfg.map_builder_objects[source_name] = 1

    def _add_converter(
        self, cfg: _BuildCfg, rng: random.Random, max_input_resources: int = 6
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

        num_input_resources = rng.randint(1, max_input_resources)

        # Keep elegance and efficiency: sample a valid multiset without enumerating all
        def _sample_composition(total: int, parts: int) -> List[int]:
            if parts <= 0:
                return []
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
        max_steps=256,
    ):
        cfg = _BuildCfg()

        # Clear non_reusable_resources to start fresh for this task
        self.config.non_reusable_resources.clear()

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
        most_efficient_optimal_reward, least_efficient_optimal_reward = 0, 0
        icl_env.game.reward_estimates = {
            "best_case_efficient_reward": most_efficient_optimal_reward,
            "worst_case_efficient_reward": least_efficient_optimal_reward,
        }

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
    num_resources=[2, 3, 4, 5],
    num_converters=[0, 1, 2],
    room_sizes=["small"],
    obstacle_types=[],
    densities=[],
) -> CurriculumConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=num_resources,
        num_sinks=num_converters,
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
    from experiments.evals.icl_unordered_chains import (
        make_icl_unordered_chain_eval_suite,
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum or make_curriculum(),
        evaluation=EvaluationConfig(simulations=make_icl_unordered_chain_eval_suite()),
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
    from experiments.evals.icl_unordered_chains import (
        make_icl_unordered_chain_eval_suite,
    )

    simulations = simulations or make_icl_unordered_chain_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )
