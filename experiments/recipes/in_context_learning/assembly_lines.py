from mettagrid.builder.envs import make_icl_assembler
from experiments.recipes.in_context_learning.in_context_learning import (
    ICLTaskGenerator,
    _BuildCfg,
    train_icl,
    play_icl,
    replay_icl,
    room_size_templates,
)
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

    def _make_resource_chain(
        self,
        resources: list[str],
        avg_hop: float,
        position: list[Position],
        cfg: _BuildCfg,
        rng: random.Random,
    ):
        cooldown = avg_hop * (len(resources) + 1)
        resource_chain = ["nothing"] + list(resources) + ["heart"]
        for i in range(len(resource_chain) - 1):
            input_resource, output_resource = resource_chain[i], resource_chain[i + 1]
            input_resources = {} if input_resource == "nothing" else {input_resource: 1}
            self._add_assembler(
                input_resources=input_resources,
                output_resources={output_resource: 1},
                position=position,
                cfg=cfg,
                cooldown=int(cooldown),
                rng=rng,
            )

    def _make_sinks(
        self,
        num_sinks: int,
        position: list[Position],
        cfg: _BuildCfg,
        rng: random.Random,
    ) -> list[str]:
        for _ in range(num_sinks):
            self._add_assembler(
                input_resources={},
                output_resources={},
                position=position,
                cfg=cfg,
                rng=rng,
            )

    def _make_env_cfg(
        self,
        num_agents,
        resources,
        num_sinks,
        width,
        height,
        position,
        terrain,
        max_steps,
        num_instances,
        rng,
        dir=None,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        self._make_resource_chain(resources, width + height / 2, position, cfg, rng)
        self._make_sinks(num_sinks, position, cfg, rng)

        if dir is not None and os.path.exists(dir):
            return self.load_from_numpy(
                num_agents,
                max_steps,
                cfg.game_objects,
                cfg.map_builder_objects,
                dir,
                rng,
                num_instances,
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

    def calculate_max_steps(
        self, chain_length: int, num_sinks: int, width: int, height: int
    ) -> int:
        avg_hop = width + height / 2

        steps_per_attempt = 4 * avg_hop
        sink_exploration_cost = steps_per_attempt * num_sinks
        chain_completion_cost = steps_per_attempt * chain_length
        target_completions = 10

        return int(sink_exploration_cost + target_completions * chain_completion_cost)

    def _generate_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        (
            num_agents,
            resources,
            num_sinks,
            room_size,
            terrain,
            width,
            height,
            max_steps,
            position,
        ) = self._setup_task(rng)

        dir = (
            f"{self.config.map_dir}/{room_size}/{len(resources)}chain/{num_sinks}sinks/{terrain}"
            if self.config.map_dir is not None
            else None
        )

        icl_env = self._make_env_cfg(
            num_agents=num_agents,
            resources=resources,
            num_sinks=num_sinks,
            width=width,
            height=height,
            position=position,
            terrain=terrain,
            max_steps=max_steps,
            num_instances=num_instances or 24 // num_agents,
            rng=rng,
            dir=dir,
        )

        icl_env.label = f"{room_size}_{len(resources)}chain_{num_sinks}sinks_{terrain}"
        return icl_env


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


def save_envs_to_numpy(dir="in_context_assembly_lines/", num_envs: int = 500):
    import os
    import numpy as np

    for chain_length in range(2, 6):
        for num_sinks in range(0, 3):
            for room_size in ["tiny", "small", "medium"]:
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
