import random
from typing import Optional, Sequence

from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.mettagrid.builder.envs import MettaGridConfig, make_icl_resource_chain
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import EvaluationConfig, TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool

from .icl_resource_chain import ICLTaskGenerator, _BuildCfg


class OrderedChainTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)

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

    def _generate_task(self, task_id: int, rng: random.Random):
        resources, num_sinks, room_size, obstacle_type, density, width, height = (
            self._setup_task(rng)
        )
        avg_hop = (width + height) / 2
        max_steps = self.config.max_steps

        most_efficient_optimal_reward, least_efficient_optimal_reward = (
            self._estimate_max_rewards(len(resources), num_sinks, max_steps, avg_hop)
        )

        icl_env = self._make_env_cfg(
            resources,
            num_sinks,
            width=width,
            height=height,
            obstacle_type=obstacle_type,
            density=density,
            avg_hop=avg_hop,
            rng=rng,
            max_steps=max_steps,
        )

        icl_env.game.reward_estimates = {
            "best_case_efficient_reward": most_efficient_optimal_reward,
            "worst_case_efficient_reward": least_efficient_optimal_reward,
        }

        icl_env.label = f"{len(resources)}resources_{num_sinks}sinks_{room_size}_{obstacle_type}_{density}"

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


def make_mettagrid() -> MettaGridConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        num_resources=[3],
        num_sinks=[1],
        room_sizes=["large"],
        obstacle_types=["cross"],
        densities=["high"],
    )
    task_generator = OrderedChainTaskGenerator(task_generator_cfg)
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


def small_curriculum():
    return make_curriculum(
        chain_lengths=[2, 3, 4, 5],
        num_sinks=[0, 1, 2],
        room_sizes=["small"],
    )


def small_medium_curriculum():
    return make_curriculum(
        chain_lengths=[2, 3, 4, 5],
        num_sinks=[0, 1, 2],
        room_sizes=["small", "medium"],
    )


def all_room_sizes_curriculum():
    return make_curriculum(
        chain_lengths=[2, 3, 4, 5],
        num_sinks=[0, 1, 2],
        room_sizes=["small", "medium", "large"],
    )


def longer_chains():
    return make_curriculum(
        chain_lengths=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        num_sinks=[0, 1, 2],
        room_sizes=["small", "medium", "large"],
    )


def longer_chains_more_sinks():
    return make_curriculum(
        chain_lengths=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        num_sinks=[0, 1, 2, 3, 4],
        room_sizes=["small", "medium", "large"],
    )


def terrain():
    return make_curriculum(
        chain_lengths=[2, 3, 4, 5, 6],
        num_sinks=[0, 1, 2],
        room_sizes=["small", "medium", "large"],
        obstacle_types=["square", "cross", "L"],
        densities=["balanced", "sparse", "high"],
    )


def train(
    curriculum: Optional[CurriculumConfig] = None,
) -> TrainTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.icl_ordered_chains import (
        make_icl_resource_chain_eval_suite,
    )

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
        curriculum=curriculum or small_curriculum(),
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
    from experiments.evals.icl_ordered_chains import (
        make_icl_resource_chain_eval_suite,
    )

    simulations = simulations or make_icl_resource_chain_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )
