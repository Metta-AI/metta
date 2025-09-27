from mettagrid.builder.envs import make_icl_assembler, make_icl_with_numpy
from experiments.recipes.in_context_learning.in_context_learning import (
    ICLTaskGenerator,
    _BuildCfg,
    calculate_avg_hop,
    train_icl,
    play_icl,
    replay_icl,
    room_size_templates,
)
from experiments.recipes.in_context_learning.converters.converter_chains import (
    calculate_max_steps,
    curriculum_args,
)
from metta.map.terrain_from_numpy import InContextLearningFromNumpy
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Position,
)
from metta.tools.train import TrainTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
import random
from typing import Optional
import os

curriculum_args = {
    "single_agent_easy": {
        "num_agents": [1],
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny", "small"],
        "positions": [["Any"]],
    },
    "single_agent_hard": {
        "num_agents": [1],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
        "positions": [["Any"]],
    },
    "two_agent_easy": {
        "num_agents": [2],
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny", "small"],
        "positions": [["Any", "Any"]],
    },
    "two_agent_hard": {
        "num_agents": [2],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
        "positions": [["Any", "Any"]],
    },
    "multi_agent_easy": {
        "num_agents": [1, 2],
        "chain_lengths": [2, 3],
        "num_sinks": [0, 1],
        "room_sizes": ["tiny", "small"],
        "positions": [["Any", "Any"], ["Any"]],
    },
    "multi_agent_hard": {
        "num_agents": [1, 2],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["tiny", "small", "medium"],
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
    num_agents: list[int],
    chain_lengths: list[int],
    num_sinks: list[int],
    room_sizes: list[str],
    positions: list[list[Position]],
    map_dir: Optional[str] = "in_context_assembly_lines",
):
    return AssemblerConverterChainTaskGenerator.Config(
        num_agents=num_agents,
        num_resources=[c - 1 for c in chain_lengths],
        num_converters=num_sinks,
        room_sizes=room_sizes,
        positions=positions,
        map_dir=map_dir,
    )


class AssemblerConverterChainTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)

    def _make_env_cfg(
        self,
        num_agents,
        resources,
        num_sinks,
        width,
        height,
        position,
        terrain,
        avg_hop,
        max_steps,
        num_instances,
        rng,
        dir=None,
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
        num_instances = 24 // num_agents if num_instances is None else num_instances

        if dir is not None:
            if os.path.exists(dir):
                return make_icl_with_numpy(
                    num_agents=num_agents,
                    num_instances=num_instances,
                    max_steps=max_steps,
                    game_objects=cfg.game_objects,
                    instance_map=InContextLearningFromNumpy.Config(
                        agents=num_agents,
                        dir=dir,
                        objects=cfg.map_builder_objects,
                        rng=rng,
                    ),
                )

        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            terrain=terrain,
        )

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        num_agents = rng.choice(self.config.num_agents)
        resources, num_sinks, room_size, _, _, width, height, _ = self._setup_task(rng)
        terrain = rng.choice(room_size_templates[room_size]["terrain"])
        width, height = self._set_width_and_height(
            room_size, num_agents, len(resources) + 1, num_sinks, rng
        )

        position = rng.choice(self.config.positions)

        avg_hop = calculate_avg_hop(room_size)
        max_steps = calculate_max_steps(avg_hop, len(resources) + 1, num_sinks)

        dir = (
            f"{self.config.map_dir}/{room_size}/{len(resources)}chain/{num_sinks}sinks/{terrain}"
            if self.config.map_dir is not None
            else None
        )

        return self._make_env_cfg(
            num_agents,
            resources,
            num_sinks,
            width,
            height,
            position,
            terrain,
            avg_hop,
            max_steps,
            num_instances,
            rng,
            dir,
        )


def train(
    curriculum_style: str = "multi_agent_easy",
) -> TrainTool:
    task_generator_cfg = make_task_generator_cfg(**curriculum_args[curriculum_style])
    from experiments.evals.in_context_learning.assembly_lines import (
        make_icl_assembler_resource_chain_eval_suite,
    )

    return train_icl(task_generator_cfg, make_icl_assembler_resource_chain_eval_suite)


def play(curriculum_style: str = "test") -> PlayTool:
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


def save_envs_to_numpy(dir="in_context_assembly_lines/", num_envs: int = 100):
    import os
    import numpy as np

    for chain_length in range(2, 7):
        for num_sinks in range(0, 3):
            for room_size in room_size_templates:
                for terrain_type in room_size_templates[room_size]["terrain"]:
                    for i in range(num_envs):
                        task_generator_cfg = make_task_generator_cfg(
                            num_agents=[4],
                            chain_lengths=[chain_length],
                            num_sinks=[num_sinks],
                            room_sizes=[room_size],
                            positions=[["Any", "Any"]],
                            map_dir=None,
                        )
                        task_generator = AssemblerConverterChainTaskGenerator(
                            config=task_generator_cfg
                        )
                        random_number = random.randint(0, 1000000)
                        terrain_type = "simple" if terrain_type == "" else terrain_type
                        filename = f"{dir}/{room_size}/{chain_length}chain/{num_sinks}sinks/{terrain_type}/{random_number}.npy"
                        os.makedirs(os.path.dirname(filename), exist_ok=True)
                        env_cfg = task_generator._generate_task(
                            i, random.Random(i), num_instances=1
                        )
                        map_builder = env_cfg.game.map_builder.create()
                        grid = map_builder.build().grid

                        print(f"saving to {filename}")
                        np.save(filename, grid)


if __name__ == "__main__":
    save_envs_to_numpy()
