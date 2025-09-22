import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.builder import empty_converters
from mettagrid.builder.envs import make_in_context_chains
from mettagrid.builder.envs import make_in_context_chains_with_numpy
from mettagrid.config.mettagrid_config import MettaGridConfig
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
    "armory": empty_converters.armory,
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

class LPParams:
    def __init__(
        self,
        ema_timescale: float = 0.001,
        exploration_bonus: float = 0.15,
        max_memory_tasks: int = 1000,
        max_slice_axes: int = 3,
        progress_smoothing: float = 0.15,
        enable_detailed_slice_logging: bool = False,
        num_active_tasks: int = 1000,
        rand_task_rate: float = 0.25,
    ):
        self.ema_timescale = ema_timescale
        self.exploration_bonus = exploration_bonus
        self.max_memory_tasks = max_memory_tasks
        self.max_slice_axes = max_slice_axes
        self.progress_smoothing = progress_smoothing
        self.enable_detailed_slice_logging = enable_detailed_slice_logging
        self.num_active_tasks = num_active_tasks
        self.rand_task_rate = rand_task_rate

curriculum_args = {
    "small": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small"],
    },
    "small_medium": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1],
        "room_sizes": ["small", "medium"],
    },
    "all_room_sizes": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
    },
    "longer_chains": {
        "chain_lengths": [2, 3, 4, 5, 6, 7],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
    },
    "terrain": {
        "chain_lengths": [2, 3, 4, 5, 6, 7],
        "num_sinks": [0, 1, 2],
        "obstacle_types": ["square", "cross", "L"],
        "densities": ["", "balanced", "sparse", "high"],
        "room_sizes": ["small", "medium", "large"],
    },
}

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
            default=["small"], description="Room size to sample from"
        )
        obstacle_types: list[str] = Field(
            default=[], description="Obstacle types to sample from"
        )
        densities: list[str] = Field(default=[], description="Density to sample from")
        # obstacle_complexity
        max_steps: int = Field(default=512, description="Episode length")

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
        room_size,
        obstacle_type,
        density,
        avg_hop,
        rng,
        max_steps=512,
        with_numpy=False,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()
        resource_chain = ["nothing"] + list(resources) + ["heart"]

        chain_length = len(resource_chain)

        for i in range(chain_length - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            self._add_converter(input_resource, output_resource, cfg, rng=rng)

        for _ in range(num_sinks):
            self._add_sink(cfg, rng=rng)

        cooldown = avg_hop * (chain_length - 1)

        for obj in cfg.converters:
            cfg.game_objects[obj].cooldown = int(cooldown)

        if with_numpy:
            return make_in_context_chains_with_numpy(
                num_agents=24,
                max_steps=max_steps,
                game_objects=cfg.game_objects,
                room_size=room_size,
                obstacle_type=obstacle_type,
                density=density,
                chain_length=chain_length,
                num_sinks=num_sinks,
            )

        return make_in_context_chains(
            num_agents=24,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            room_size=room_size,
            obstacle_type=obstacle_type,
            density=density,
            chain_length=chain_length,
            num_sinks=num_sinks,
        )

    def _generate_task(self, task_id: int, rng: random.Random, with_numpy: bool = True) -> MettaGridConfig:
        num_resources = rng.choice(self.config.chain_lengths)
        num_sinks = rng.choice(self.config.num_sinks)
        resources = rng.sample(self.resource_types, num_resources)
        room_size = rng.choice(self.config.room_sizes)
        obstacle_type = (
            rng.choice(self.config.obstacle_types)
            if len(self.config.obstacle_types) > 0
            else None
        )
        density = (
            rng.choice(self.config.densities)
            if len(self.config.densities) > 0
            else None
        )

        max_steps = self.config.max_steps

        if room_size == "tiny":
            avg_hop = 7
        elif room_size == "small":
            avg_hop = 10
        elif room_size == "medium":
            avg_hop = 13

        # optimal reward estimates for the task, to be used in evaluation
        best_case_optimal_reward, worst_case_optimal_reward = (
            self._estimate_max_rewards(num_resources, num_sinks, max_steps, avg_hop)
        )

        icl_env = self._make_env_cfg(
            resources,
            num_sinks,
            room_size,
            obstacle_type=obstacle_type,
            density=density,
            avg_hop=avg_hop,
            max_steps=max_steps,
            rng=rng,
            with_numpy=with_numpy,
        )

        icl_env.game.reward_estimates = {
            "best_case_optimal_reward": best_case_optimal_reward,
            "worst_case_optimal_reward": worst_case_optimal_reward,
        }

        icl_env.label = f"{num_resources}resources_{num_sinks}sinks_{room_size}"
        icl_env.label += "_terrain" if obstacle_type else ""
        icl_env.label += f"_{density}" if density else ""

        return icl_env

    def _estimate_max_rewards(
        self,
        num_resources: int,
        num_sinks: int,
        max_steps: int,
        avg_hop: float,
    ) -> tuple[float, float]:
        """
        Returns (best_case_optimal_reward, worst_case_optimal_reward).

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


def make_mettagrid(curriculum_style: str) -> MettaGridConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        **curriculum_args[curriculum_style],
    )
    task_generator = ConverterChainTaskGenerator(task_generator_cfg)

    env_cfg = task_generator.get_task(0)
    #TODO FINISHI THIS
    dir = f"icl_resource_chain/{curriculum_style}/{task_generator_cfg.chain_lengths[0]}chains_{task_generator_cfg.num_sinks[0]}sinks/"

    return env_cfg


def make_curriculum(
    curriculum_style: str,
    lp_params: LPParams = LPParams(),
) -> CurriculumConfig:
    task_generator_cfg = ConverterChainTaskGenerator.Config(
        **curriculum_args[curriculum_style],
    )
    algorithm_config = LearningProgressConfig(**lp_params.__dict__)

    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum_style: str = "terrain", lp_params: LPParams = LPParams()
) -> TrainTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.ordered_chains import (
        make_icl_resource_chain_eval_suite,
    )

    curriculum = make_curriculum(curriculum_style, lp_params)

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum,
        evaluation=EvaluationConfig(
            simulations=make_icl_resource_chain_eval_suite(),
            evaluate_remote=True,
            evaluate_local=False,
        ),
    )
    # for in context learning, we need episode length to be equal to bptt_horizon
    # which requires a large batch size
    trainer_cfg.batch_size = 4128768
    trainer_cfg.bptt_horizon = 512

    return TrainTool(trainer=trainer_cfg)


def play(env: Optional[MettaGridConfig] = None, curriculum_style: str = "terrain") -> PlayTool:
    eval_env = env or make_mettagrid(curriculum_style)
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
    )


def replay(env: Optional[MettaGridConfig] = None, curriculum_style: str = "terrain") -> ReplayTool:
    eval_env = env or make_mettagrid(curriculum_style)
    # Default to the research policy if none specified
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19:v960.pt"
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            name="in_context_resource_chain",
        ),
        policy_uri=default_policy_uri,
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
        "longer_chains",
        "terrain",
    ]

    pretrained_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19/icl_resource_chain_terrain_PS0.05_EB0.15_NAT1000_RTR0.25.09-19:v960.pt"

    for curriculum_style in curriculum_styles:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.ordered_chains.train",
                f"run=icl_resource_chain_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
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
                "experiments.recipes.in_context_learning.ordered_chains.train",
                f"run=icl_resource_chain_{curriculum_style}_pretrained.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                f"trainer.initial_policy.uri={pretrained_policy_uri}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )


def save_envs_to_numpy(num_envs: int = 1000):
    curriculum_styles = ["small", "small_medium", "all_room_sizes", "longer_chains", "terrain"]
    for curriculum_style in curriculum_styles:
        print(f"Generating {curriculum_style}...")
        for i in range(num_envs):
            print(f"Generating {i}...")
            task_generator_cfg = ConverterChainTaskGenerator.Config(
                **curriculum_args[curriculum_style],
            )
            task_generator = ConverterChainTaskGenerator(task_generator_cfg)
            env_cfg = task_generator._generate_task(i, random.Random(i))
            map_builder = env_cfg.game.map_builder.create()
            map_builder.build()

if __name__ == "__main__":
    generate_envs()
    # experiment()
