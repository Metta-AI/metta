import json
import os
import random
import subprocess
import time
from typing import Optional, Sequence
from experiments.sweeps.protein_configs import make_custom_protein_config, PPO_CORE
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.protein_config import ParameterConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.sweep import SweepTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_icl_with_numpy, make_in_context_chains
from mettagrid.config.mettagrid_config import MettaGridConfig
from experiments.recipes.in_context_learning.in_context_learning import (
    ICLTaskGenerator,
    LPParams,
    train_icl,
    _BuildCfg,
    calculate_avg_hop,
    play_icl,
    replay_icl,
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
    "full": {
        "chain_lengths": [2, 3, 4, 5, 6],
        "num_sinks": [0, 1, 2],
        "obstacle_types": ["square", "cross", "L"],
        "densities": ["", "balanced", "sparse", "high"],
        "room_sizes": ["tiny", "small", "medium", "large"],
    },
    "test": {
        "chain_lengths": [4],
        "num_sinks": [2],
        "room_sizes": ["medium"],
    },
}


def make_task_generator_cfg(
    chain_lengths,
    num_sinks,
    room_sizes,
    map_dir,
    obstacle_types=[],
    densities=[],
):
    return ConverterChainTaskGenerator.Config(
        num_resources=[c - 1 for c in chain_lengths],
        num_converters=num_sinks,
        room_sizes=room_sizes,
        obstacle_types=obstacle_types,
        densities=densities,
        map_dir=map_dir,
    )


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


def calculate_max_steps(avg_hop: float, chain_length: int, num_sinks: int) -> int:
    """
    Calculate maximum steps for an episode based on environment parameters.

    This calculation ensures enough time for:
    1. Finding all sinks through exploration
    2. Completing the chain at least 10 times

    Formula breakdown:
    - steps_per_attempt = 2 * avg_hop (movement to object + interaction costs)
    - Finding sinks: steps_per_attempt * num_sinks
    - Chain completion: steps_per_attempt * chain_length (traverse full chain once)
    - Target: Complete chain 10 times minimum

    Total = sink_exploration + 5 * chain_completion
    """
    steps_per_attempt = 2 * avg_hop
    sink_exploration_cost = steps_per_attempt * num_sinks
    chain_completion_cost = steps_per_attempt * chain_length
    target_completions = 10

    return int(sink_exploration_cost + target_completions * chain_completion_cost)


class ConverterChainTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)
        self.map_dir = config.map_dir

    def load_from_numpy(
        self,
        room_size,
        obstacle_type,
        density,
        resources,
        num_sinks,
        max_steps,
        cfg,
        rng,
    ) -> MettaGridConfig:
        from metta.map.terrain_from_numpy import InContextLearningFromNumpy

        terrain = "simple-" if obstacle_type is None else f"terrain-{density}"
        dir = f"{self.map_dir}/{room_size}/{len(resources) + 1}chains_{num_sinks}sinks/{terrain}"
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
        if os.path.exists(f"./train_dir/{dir}/reward_estimates.json"):
            reward_estimates = json.load(
                open(f"./train_dir/{dir}/reward_estimates.json")
            )
            env.game.reward_estimates = reward_estimates[dir]
        return env

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
        max_steps,
        rng,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        resource_chain = ["nothing"] + list(resources) + ["heart"]
        cooldown = avg_hop * (len(resource_chain) - 1)

        for i in range(len(resource_chain) - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            converter_name = self._add_converter(
                input_resources={input_resource: 1},
                output_resources={output_resource: 1},
                cfg=cfg,
                rng=rng,
                cooldown=cooldown,
            )
            cfg.converters.append(converter_name)

        for _ in range(num_sinks):
            self._add_converter(
                input_resources={
                    input_resource: 1 for input_resource in cfg.all_input_resources
                },
                output_resources={},
                cfg=cfg,
                rng=rng,
            )

        if self.map_dir is not None:  # load from s3
            return self.load_from_numpy(
                room_size,
                obstacle_type,
                density,
                resources,
                num_sinks,
                max_steps,
                cfg,
                rng,
            )
            if os.path.exists(f"./train_dir/{dir}/reward_estimates.json"):
                reward_estimates = json.load(
                    open(f"./train_dir/{dir}/reward_estimates.json")
                )
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
            dir=self.map_dir,
        )

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
        estimate_max_rewards: bool = False,
    ) -> MettaGridConfig:
        resources, num_sinks, room_size, obstacle_type, density, width, height, _ = (
            self._setup_task(rng)
        )

        # estimate average hop for cooldowns
        avg_hop = calculate_avg_hop(room_size)
        max_steps = calculate_max_steps(avg_hop, len(resources) + 1, num_sinks)
        icl_env = self._make_env_cfg(
            resources,
            num_sinks,
            room_size,
            width,
            height,
            obstacle_type=obstacle_type,
            density=density,
            avg_hop=avg_hop,
            max_steps=max_steps,
            rng=rng,
        )

        # for numpy generated maps, we just load these rewards from a file
        if self.map_dir is None and estimate_max_rewards:
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


def train(
    curriculum_style: str = "tiny",
    lp_params: LPParams = LPParams(),
    map_dir: str = "icl_ordered_chains",
) -> TrainTool:
    task_generator_cfg = make_task_generator_cfg(
        **curriculum_args[curriculum_style], map_dir=map_dir
    )
    from experiments.evals.in_context_learning.converter_chains import (
        make_converter_chain_eval_suite,
    )

    return train_icl(
        task_generator_cfg,
        evaluator_fn=make_converter_chain_eval_suite,
        lp_params=lp_params,
    )


def play(curriculum_style: str = "tiny", map_dir="icl_ordered_chains") -> PlayTool:
    task_generator = ConverterChainTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style], map_dir=map_dir)
    )
    return play_icl(task_generator)


def replay(
    curriculum_style: str = "hard_eval",
    map_dir=None,
    policy_uri: str = "s3://softmax-public/policies/icl_resource_chain_terrain_4.2.2025-09-24/icl_resource_chain_terrain_4.2.2025-09-24:v2370.pt",
) -> ReplayTool:
    task_generator = ConverterChainTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style], map_dir=map_dir)
    )
    return replay_icl(task_generator, policy_uri)


def evaluate(
    simulations: Optional[Sequence[SimulationConfig]] = None,
) -> SimTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.converter_chains import (
        make_converter_chain_eval_suite,
    )

    curriculum_styles = [
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
    simulations = simulations or make_converter_chain_eval_suite()
    policy_uris = []
    for curriculum_style in curriculum_styles:
        policy_uris.append(
            f"s3://softmax-public/policies/icl_resource_chain_{curriculum_style}.2.2025-09-24/icl_resource_chain_{curriculum_style}.2.2025-09-24:latest.pt"
        )
    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    curriculum_styles = ["full"]

    for curriculum_style in curriculum_styles:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.converters.converter_chains.train",
                f"run=icl_resource_chain_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


def save_envs_to_numpy(dir="icl_ordered_chains/", num_envs: int = 100):
    for chain_length in range(
        2, 10
    ):  # chain length should be equal to the number of converters, which is equal to the number of resources + 1
        for n_sinks in range(0, 4):
            for room_size in ["medium", "large"]:
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
                            task_generator_cfg = make_task_generator_cfg(
                                chain_lengths=[chain_length],
                                num_sinks=[n_sinks],
                                room_sizes=[room_size],
                                obstacle_types=[obstacle_type],
                                densities=[density],
                                map_dir=None,
                            )
                            task_generator = ConverterChainTaskGenerator(
                                config=task_generator_cfg
                            )
                            env_cfg = task_generator._generate_task(i, random.Random(i))
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


def sweep(
    total_timesteps: int = 1000000,
) -> SweepTool:
    lp_protein_config = make_custom_protein_config(
        base_config=PPO_CORE,
        parameters={
            "lp_params.progress_smoothing": ParameterConfig(
                distribution="uniform",  # Changed from logit_normal - more appropriate for 0.05-0.15 range
                min=0.05,
                max=0.15,
                mean=0.1,
                scale="auto",
            ),
            "lp_params.exploration_bonus": ParameterConfig(
                distribution="uniform",  # Changed from logit_normal - more appropriate for 0.03-0.15 range
                min=0.03,
                max=0.15,
                mean=0.09,
                scale="auto",
            ),
            "lp_params.ema_timescale": ParameterConfig(
                distribution="log_normal",  # Changed to log_normal for better exploration of small values
                min=0.001,
                max=0.01,
                mean=0.00316,  # Geometric mean: sqrt(0.001 * 0.01) ≈ 0.00316
                scale="auto",
            ),
            "lp_params.num_active_tasks": ParameterConfig(
                distribution="int_uniform",  # Changed to int_uniform since this is a count of tasks
                min=1000,
                max=5000,
                mean=3000,  # Arithmetic mean for uniform distribution
                scale="auto",
            ),
            "lp_params.rand_task_rate": ParameterConfig(
                distribution="uniform",
                min=0.1,
                max=0.25,
                mean=0.175,
                scale="auto",
            ),
        },
    )

    lp_protein_config.metric = "evaluator/eval_in_context_learning/in_context_learning"

    return SweepTool(
        protein_config=lp_protein_config,
        recipe_module="experiments.recipes.in_context_learning.ordered_chains",
        train_entrypoint="train",
        eval_entrypoint="evaluate",
        train_overrides={
            "trainer.total_timesteps": total_timesteps,
        },
    )


if __name__ == "__main__":
    # experiment()
    save_envs_to_numpy()
