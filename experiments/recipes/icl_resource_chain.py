import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
    CurriculumAlgorithmConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
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
        room_sizes: list[str] = Field(
            default=["6x6"], description="Room size to sample from"
        )
        obstacle_types: list[str] = Field(
            default=[], description="Obstacle types to sample from"
        )
        densities: list[str] = Field(default=[], description="Density to sample from")
        # obstacle_complexity
        max_steps: int = Field(default=256, description="Episode length")

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
            self.converter_types, set(cfg.used_objects), rng
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
            self.converter_types, set(cfg.used_objects), rng
        )
        cfg.used_objects.append(sink_name)
        sink = self.converter_types[sink_name].copy()

        for input_resource in cfg.all_input_resources:
            sink.input_resources[input_resource] = 1

        cfg.game_objects[sink_name] = sink
        cfg.map_builder_objects[sink_name] = 1

    def _make_env_cfg(
        self,
        resources,
        num_sinks,
        width,
        height,
        obstacle_type,
        density,
        avg_hop,
        rng,
        max_steps=256,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()
        resource_chain = ["nothing"] + list(resources) + ["heart"]

        chain_length = len(resource_chain)

        for i in range(chain_length - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            self._add_converter(input_resource, output_resource, cfg, rng=rng)

        for _ in range(num_sinks):
            self._add_sink(cfg, rng=rng)

        # longer episodes for longer chains
        if len(cfg.used_objects) > 4:
            max_steps = self.config.max_steps * 2

        cooldown = avg_hop * (chain_length - 1)

        for obj in cfg.converters:
            cfg.game_objects[obj].cooldown = int(cooldown)

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

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        num_resources = rng.choice(self.config.chain_lengths)
        num_sinks = rng.choice(self.config.num_sinks)
        resources = rng.sample(self.resource_types, num_resources)
        room_size = rng.choice(self.config.room_sizes)
        obstacle_type = rng.choice(self.config.obstacle_types)
        density = rng.choice(self.config.densities)

        print(f"obstacle_type: {obstacle_type}, density: {density}")
        print()
        # by default, use a 6x6 room - to reproduce existing results
        if room_size == "6x6":
            width, height = 6, 6
        else:
            if room_size == "small":
                size_range = (5, 8)
            elif room_size == "medium":
                size_range = (8, 12)
            elif room_size == "large":
                size_range = (12, 15)

            width, height = (
                rng.randint(size_range[0], size_range[1]),
                rng.randint(size_range[0], size_range[1]),
            )

        max_steps = self.config.max_steps

        avg_hop = (width + height) / 2

        # optimal reward estimates for the task, to be used in evaluation
        most_efficient_optimal_reward, least_efficient_optimal_reward = (
            self._estimate_max_rewards(num_resources, num_sinks, max_steps, avg_hop)
        )

        icl_env = self._make_env_cfg(
            resources,
            num_sinks,
            width=width,
            height=height,
            obstacle_type=obstacle_type,
            density=density,
            avg_hop=avg_hop,
            max_steps=max_steps,
            rng=rng,
        )

        icl_env.game.reward_estimates = {
            "most_efficient_optimal_reward": most_efficient_optimal_reward,
            "least_efficient_optimal_reward": least_efficient_optimal_reward,
        }

        icl_env.label = f"{num_resources}resources_{num_sinks}sinks_{room_size}"

        return icl_env

    def _estimate_max_rewards(
        self,
        num_resources: int,
        num_sinks: int,
        max_steps: int,
        avg_hop: float,
    ) -> tuple[float, float]:
        """
        Returns (most_efficient_reward, least_efficient_reward).

        Updates vs prior:
          * Each converter interaction = 2 actions (put + get).
          * Both scenarios include average hop distance between perimeter objects.
          * Per-heart cycle time is the bottleneck of either converter cooldown or the
            movement+interaction cost to traverse the whole chain again.

        Definitions:
          - num_resources: number of *intermediate* resources between "nothing" and "heart".
          - n_converters = chain_length + 1 (edges: nothing->r1, ..., r_k->heart).
          - cooldown = avg_hop * n_converters (as set in _make_env_cfg).
        """
        # Number of converters in the chain (nothing->r1, ..., r_k->heart)
        n_converters = num_resources + 1
        total_objects = n_converters + num_sinks

        # Mirror _make_env_cfg’s episode-length extension
        effective_max_steps = max_steps * 2 if total_objects > 4 else max_steps

        # Converter cooldown applied uniformly
        cooldown = avg_hop * n_converters

        # Cost per attempt at any object = move there + (put + get)
        step_per_attempt = avg_hop + 2

        # Cost to traverse the *correct* chain once (movement + interactions at each stage)
        correct_chain_traverse_cost = n_converters * step_per_attempt

        # One full production cycle after the first heart is limited by either cooldown
        # or the time to traverse the chain again (including moving between stages).
        per_heart_cycle = max(cooldown, correct_chain_traverse_cost)

        def hearts_after(first_heart_steps: float) -> float:
            if first_heart_steps > effective_max_steps:
                return 0
            remaining = effective_max_steps - first_heart_steps
            return 1 + (remaining // per_heart_cycle)

        # ---------- Most efficient ----------
        # Immediately discover the correct chain; still pay average hop + (put+get) at each stage.
        best_first_heart_steps = correct_chain_traverse_cost
        most_efficient = hearts_after(best_first_heart_steps)

        # ---------- Least efficient ----------
        #   1. Find the first converter: (converters + sinks) attempts
        #   2. Find all sinks: ~(converters + 2 * sinks) attempts
        #      (every time you find a sink, you need to go get an item again)
        #   3. Find the right pattern: ~converters * (converters - 1) attempts
        find_first_converter_cost = (n_converters + num_sinks) * step_per_attempt
        find_all_sinks_cost = (n_converters + 2 * num_sinks) * step_per_attempt
        find_right_pattern_cost = n_converters * (n_converters - 1) * step_per_attempt

        worst_first_heart_steps = (
            find_first_converter_cost + find_all_sinks_cost + find_right_pattern_cost
        )
        least_efficient = hearts_after(worst_first_heart_steps)

        return int(most_efficient), int(least_efficient)


def make_mettagrid() -> MettaGridConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        chain_lengths=[3],
        num_sinks=[1],
        room_sizes=["large"],
        obstacle_types=["cross"],
        densities=["dense"],
    )
    print(task_generator_cfg)
    task_generator = ConverterChainTaskGenerator(task_generator_cfg)
    return task_generator.get_task(0)


def make_curriculum(
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        chain_lengths=[2, 3, 4, 5],
        num_sinks=[0, 1, 2],
        room_sizes=["small"],
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
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.icl_resource_chain import (
        make_icl_resource_chain_eval_suite,
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum
        or make_curriculum(enable_detailed_slice_logging=enable_detailed_slice_logging),
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
        policy_uri="wandb://run/georgedeane.operant_conditioning.in_context_learning.all.0.1.08-19:v50",
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
