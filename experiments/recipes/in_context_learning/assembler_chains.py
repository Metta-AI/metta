from mettagrid.builder.envs import make_icl_assembler
from mettagrid.config.mettagrid_config import MettaGridConfig
from experiments.recipes.in_context_learning.in_context_learning import (
    ICLTaskGenerator,
    _BuildCfg,
    calculate_avg_hop,
    train_icl,
    play_icl,
    replay_icl,
)

# different positions for the generators and altars

from experiments.recipes.in_context_learning.converter_chains import (
    ConverterChainTaskGenerator,
    calculate_max_steps,
    curriculum_args,
)
from metta.tools.train import TrainTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
import random

# TODO add terrain to these environments

# TODO make room size adjust to number of objects

curriculum_args = {
    "single_agent_easy": {
        "num_agents": [1],
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["medium"],
        "positions": [["Any"]],
    },
    "single_agent_hard": {
        "num_agents": [1],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["medium", "large"],
        "positions": [["Any"]],
    },
    "two_agent_easy": {
        "num_agents": [2],
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["medium"],
        "positions": [["Any", "Any"]],
    },
    "two_agent_medium": {
        "num_agents": [2],
        "chain_lengths": [2, 3, 4],
        "num_sinks": [0, 1],
        "room_sizes": ["medium", "large"],
        "positions": [["Any", "Any"]],
    },
    "two_agent_hard": {
        "num_agents": [2],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["medium", "large"],
        "positions": [["Any", "Any"]],
    },
    "multi_agent_easy": {
        "num_agents": [1, 2],
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["medium"],
        "positions": [["Any", "Any"], ["Any"]],
    },
    "multi_agent_medium": {
        "num_agents": [1, 2],
        "chain_lengths": [2, 3, 4],
        "num_sinks": [0, 1],
        "room_sizes": ["medium", "large"],
        "positions": [["Any", "Any"], ["Any"]],
    },
    "multi_agent_hard": {
        "num_agents": [1, 2],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["medium", "large"],
        "positions": [["Any", "Any"], ["Any"]],
    },
    "test": {
        "num_agents": [2],
        "chain_lengths": [5],
        "num_sinks": [2],
        "room_sizes": ["medium"],
        "positions": [["Any", "Any"]],
    },
}


def make_task_generator_cfg(
    num_agents,
    chain_lengths,
    num_sinks,
    room_sizes,
    positions,
    obstacle_types=[],
    densities=[],
):
    return AssemblerConverterChainTaskGenerator.Config(
        num_agents=num_agents,
        num_resources=[c - 1 for c in chain_lengths],
        num_converters=num_sinks,
        room_sizes=room_sizes,
        positions=positions,
        obstacle_types=obstacle_types,
        densities=densities,
    )


class AssemblerConverterChainTaskGenerator(ConverterChainTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)

    def _make_env_cfg(
        self,
        num_agents,
        resources,
        num_sinks,
        width,
        room_size,
        height,
        position,
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
            input_resources = {} if input_resource == "nothing" else {input_resource: 1}
            self._add_assembler(
                input_resources=input_resources,
                output_resources={output_resource: 1},
                position=position,
                cfg=cfg,
                cooldown=cooldown,
                rng=rng,
            )

        for _ in range(num_sinks):
            self._add_assembler(
                input_resources={
                    input_resource: 1 for input_resource in cfg.all_input_resources
                },
                output_resources={},
                position=position,
                cfg=cfg,
                rng=rng,
            )
        num_instances = 24 // num_agents
        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            width=width,
            height=height,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
        )

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
    ) -> MettaGridConfig:
        num_agents = rng.choice(self.config.num_agents)
        resources, num_sinks, room_size, obstacle_type, density, width, height, _ = (
            self._setup_task(rng)
        )
        position = rng.choice(self.config.positions)

        avg_hop = calculate_avg_hop(room_size)
        max_steps = calculate_max_steps(avg_hop, len(resources) + 1, num_sinks)

        return self._make_env_cfg(
            num_agents,
            resources,
            num_sinks,
            width,
            room_size,
            height,
            position,
            obstacle_type,
            density,
            avg_hop,
            max_steps,
            rng,
        )


def make_mettagrid(curriculum_style: str) -> MettaGridConfig:
    # Update config to support map_dir from main
    task_generator = AssemblerConverterChainTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )

    env_cfg = task_generator.get_task(random.randint(0, 1000000))

    return env_cfg


def train(
    curriculum_style: str = "tiny",
) -> TrainTool:
    task_generator_cfg = make_task_generator_cfg(**curriculum_args[curriculum_style])
    from experiments.evals.in_context_learning.assembler_chains import (
        make_icl_assembler_resource_chain_eval_suite,
    )

    return train_icl(task_generator_cfg, make_icl_assembler_resource_chain_eval_suite)


def play(curriculum_style: str = "tiny", map_dir=None) -> PlayTool:
    task_generator = AssemblerConverterChainTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    return play_icl(task_generator)


def replay(
    curriculum_style: str = "hard_eval",
) -> ReplayTool:
    task_generator = AssemblerConverterChainTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    # Default to the research policy if none specified
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_4.2.2025-09-24/icl_resource_chain_terrain_4.2.2025-09-24:v2370.pt"
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_1.2.2025-09-24/icl_resource_chain_terrain_1.2.2025-09-24:v2070.pt"
    return replay_icl(task_generator, default_policy_uri)
