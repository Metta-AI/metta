import random
import subprocess
import time
from typing import Optional, Sequence
from metta.cogworks.curriculum.curriculum import (
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss.loss_config import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from metta.agent.policies.fast import FastConfig
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from mettagrid.builder.envs import make_icl_with_numpy, make_in_context_chains
from mettagrid.config.mettagrid_config import MettaGridConfig
import json
import os

from experiments.evals.in_context_learning.ordered_chains import (
    make_icl_resource_chain_eval_suite,
)

from experiments.recipes.in_context_learning.icl_resource_chain import (
    ICLTaskGenerator,
    LPParams,
    _BuildCfg,
)


curriculum_args = {
    "level_0": {
        "chain_lengths": [2],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny"],
    },
    "level_1": {
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny"],
    },
    "level_2": {
        "chain_lengths": [2, 3, 4],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny"],
    },
    "tiny": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny"],
    },
    "tiny_small": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny", "small"],
    },
    "all_room_sizes": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
    },
    "longer_chains": {
        "chain_lengths": [2, 3, 4, 5, 6, 7],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
    },
    "terrain_1": {
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "obstacle_types": ["square"],
        "densities": ["", "balanced", "sparse"],
        "room_sizes": ["tiny", "small"],
    },
    "terrain_2": {
        "chain_lengths": [2, 3, 4],
        "num_sinks": [0, 1, 2],
        "obstacle_types": ["square", "cross", "L"],
        "densities": ["", "balanced", "sparse"],
        "room_sizes": ["tiny", "small"],
    },
    "terrain_3": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "obstacle_types": ["square", "cross", "L"],
        "densities": ["", "balanced", "sparse"],
        "room_sizes": ["tiny", "small", "medium"],
    },
    "terrain_4": {
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "obstacle_types": ["square", "cross", "L"],
        "densities": ["", "balanced", "sparse", "high"],
        "room_sizes": ["tiny", "small", "medium"],
    },
}


def get_reward_estimates(
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
        if first_heart_steps > max_steps:
            return 0
        remaining = max_steps - first_heart_steps
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


class OrderedChainsTaskGenerator(ICLTaskGenerator):
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
        room_size,
        width,
        height,
        obstacle_type,
        density,
        avg_hop,
        rng,
        max_steps=512,
        numpy_dir: str | None = "icl_ordered_chains",
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        resource_chain = ["nothing"] + list(resources) + ["heart"]

        for i in range(len(resource_chain) - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            self._add_converter(input_resource, output_resource, cfg, rng=rng)

        for _ in range(num_sinks):
            self._add_sink(cfg, rng=rng)

        cooldown = avg_hop * (len(resource_chain) - 1)

        for obj in cfg.converters:
            cfg.game_objects[obj].cooldown = int(cooldown)

        if numpy_dir is not None:  # load from s3
            from metta.map.terrain_from_numpy import InContextLearningFromNumpy

            terrain = "simple-" if obstacle_type is None else f"terrain-{density}"
            dir = f"{numpy_dir}/{room_size}/{len(resources) + 1}chains_{num_sinks}sinks/{terrain}"
            env = make_icl_with_numpy(
                num_agents=1,
                num_instances=24,
                max_steps=max_steps,
                game_objects=cfg.game_objects,
                instance_map=InContextLearningFromNumpy.Config(
                    dir=dir,
                    object_names=cfg.used_objects,
                    rng=rng,
                ),
            )
            if os.path.exists(f"{dir}/reward_estimates.json"):
                reward_estimates = json.load(open(f"{dir}/reward_estimates.json"))
                env.game.reward_estimates = reward_estimates[dir]
            return env

        return make_in_context_chains(
            num_agents=24,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            obstacle_type=obstacle_type,
            density=density,
            chain_length=len(resources) + 1,
            num_sinks=num_sinks,
        )

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
        numpy_dir: str | None = "icl_ordered_chains",
        estimate_max_rewards: bool = False,
    ) -> MettaGridConfig:
        resources, num_sinks, room_size, obstacle_type, density, width, height = (
            self._setup_task(rng)
        )

        max_steps = self.config.max_steps

        # estimate average hop for cooldowns
        avg_hop = 7 if room_size == "tiny" else 10 if room_size == "small" else 13

        icl_env = self._make_env_cfg(
            resources,
            num_sinks,
            room_size,
            width,
            height,
            obstacle_type=obstacle_type,
            density=density,
            avg_hop=avg_hop,
            rng=rng,
            numpy_dir=numpy_dir,
        )

        # for numpy generated maps, we just load these rewards from a file
        if numpy_dir is None and estimate_max_rewards:
            # optimal reward estimates for the task, to be used in evaluation
            best_case_optimal_reward, worst_case_optimal_reward = get_reward_estimates(
                len(resources), num_sinks, max_steps, avg_hop
            )
            icl_env.game.reward_estimates = {
                "best_case_optimal_reward": best_case_optimal_reward,
                "worst_case_optimal_reward": worst_case_optimal_reward,
            }

        icl_env.label = f"{len(resources)}resources_{num_sinks}sinks_{room_size}"
        icl_env.label += "_terrain" if obstacle_type else ""
        icl_env.label += f"_{density}" if density else ""

        return icl_env


def make_mettagrid(curriculum_style: str) -> MettaGridConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        **curriculum_args[curriculum_style],
    )
    task_generator = OrderedChainsTaskGenerator(task_generator_cfg)

    env_cfg = task_generator.get_task(0)

    return env_cfg


def make_curriculum(
    curriculum_style: str,
    lp_params: LPParams = LPParams(),
) -> CurriculumConfig:
    task_generator_cfg = ICLTaskGenerator.Config(
        **curriculum_args[curriculum_style],
    )
    algorithm_config = LearningProgressConfig(**lp_params.__dict__)

    return CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )


def train(
    curriculum_style: str = "tiny",
    lp_params: LPParams = LPParams(),
    use_fast_lstm_reset: bool = True,
) -> TrainTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.ordered_chains import (
        make_icl_resource_chain_eval_suite,
    )

    curriculum = make_curriculum(curriculum_style, lp_params)

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    if use_fast_lstm_reset:
        policy_config = FastLSTMResetConfig()
    else:
        policy_config = FastConfig()
        trainer_cfg.batch_size = 4177920
        trainer_cfg.bptt_horizon = 512

    return TrainTool(
        trainer=trainer_cfg,
        policy_architecture=policy_config,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(
            simulations=make_icl_resource_chain_eval_suite(),
            evaluate_remote=True,
            evaluate_local=False,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def play(
    env: Optional[MettaGridConfig] = None, curriculum_style: str = "tiny"
) -> PlayTool:
    eval_env = env or make_mettagrid(curriculum_style)
    return PlayTool(
        sim=SimulationConfig(
            env=eval_env,
            suite="in_context_learning",
            name="eval",
        ),
    )


def replay(
    env: Optional[MettaGridConfig] = None, curriculum_style: str = "tiny_small"
) -> ReplayTool:
    eval_env = env or make_mettagrid(curriculum_style)
    # Default to the research policy if none specified
    # default_policy_uri = "s3://softmax-public/policies/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22:v500.pt"
    return ReplayTool(
        sim=SimulationConfig(
            env=eval_env,
            suite="in_context_learning",
            name="eval",
        ),
        policy_uri="s3://softmax-public/policies/icl_resource_chain_tiny_small.2025-09-22/icl_resource_chain_tiny_small.2025-09-22:v500.pt",
    )


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    # Local import to   avoid circular import at module load time

    simulations = simulations or make_icl_resource_chain_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=[policy_uri],
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    curriculum_styles = [
        "level_0",
        "level_1",
        "level_2",
        "tiny_small",
        "all_room_sizes",
        "longer_chains",
        "terrain_1",
        "terrain_2",
        "terrain_3",
        "terrain_4",
    ]

    for curriculum_style in curriculum_styles:
        for use_fast_lstm_reset in [True, False]:
            subprocess.run(
                [
                    "./devops/skypilot/launch.py",
                    "experiments.recipes.in_context_learning.ordered_chains.train",
                    f"run=icl_resource_chain_{curriculum_style}.newarchitecture{use_fast_lstm_reset}.{time.strftime('%Y-%m-%d')}",
                    f"curriculum_style={curriculum_style}",
                    f"use_fast_lstm_reset={use_fast_lstm_reset}",
                    "--gpus=4",
                    "--heartbeat-timeout=3600",
                    "--skip-git-check",
                ]
            )
        time.sleep(1)


def save_envs_to_numpy(dir="icl_ordered_chains/", num_envs: int = 100):
    for chain_length in range(
        2, 8
    ):  # chain length should be equal to the number of converters, which is equal to the number of resources + 1
        for n_sinks in range(0, 3):
            for room_size in ["tiny", "small", "medium"]:
                for terrain_type in ["", "terrain"]:
                    for density in ["", "balanced", "sparse", "high"]:
                        for i in range(num_envs):
                            print(
                                f"Generating {i} for {chain_length} chains, {n_sinks} sinks, {room_size}, {terrain_type}, {density}"
                            )
                            if terrain_type == "terrain":
                                obstacle_type = random.choice(["square", "cross", "L"])
                            else:
                                obstacle_type = ""
                            task_generator_cfg = ICLTaskGenerator.Config(
                                chain_lengths=[chain_length],
                                num_sinks=[n_sinks],
                                room_sizes=[room_size],
                                obstacle_types=[obstacle_type],
                                densities=[density],
                            )
                            task_generator = OrderedChainsTaskGenerator(
                                config=task_generator_cfg
                            )
                            env_cfg = task_generator._generate_task(
                                i, random.Random(i), numpy_dir=None
                            )
                            map_builder = env_cfg.game.map_builder.create()
                            map_builder.build()

    generate_reward_estimates(dir=dir)


def generate_reward_estimates(dir="icl_ordered_chains"):
    # TODO: Eventually we want to make the reward estimates more accurate, per actual map and including terrain.
    # For now we just use the average hop distance.
    import json
    import os

    import numpy as np

    room_sizes = os.listdir(dir)
    reward_estimates = {}
    for room_size in room_sizes:
        # Delete all .DS_Store files in the directory tree
        chains = os.listdir(f"{dir}/{room_size}")
        for chain_dir in chains:
            num_resources = int(chain_dir[0])
            num_sinks = int(chain_dir[1:].strip("chains_")[0])
            for terrain in os.listdir(f"{dir}/{room_size}/{chain_dir}"):
                files = os.listdir(f"{dir}/{room_size}/{chain_dir}/{terrain}")
                for file in files:
                    grid = np.load(f"{dir}/{room_size}/{chain_dir}/{terrain}/{file}")
                    avg_hop = (grid.shape[0] + grid.shape[1]) / 2
                    best_case_optimal_reward, worst_case_optimal_reward = (
                        get_reward_estimates(num_resources, num_sinks, 512, avg_hop)
                    )
                    reward_estimates[f"{dir}/{room_size}/{chain_dir}/{terrain}"] = {
                        "best_case_optimal_reward": best_case_optimal_reward,
                        "worst_case_optimal_reward": worst_case_optimal_reward,
                    }
    # Save the reward_estimates dictionary to a JSON file
    with open(f"{dir}/reward_estimates.json", "w") as f:
        json.dump(reward_estimates, f, indent=2)


if __name__ == "__main__":
    # experiment()
    # save_envs_to_numpy()
    generate_reward_estimates()
