import os
import random
import subprocess
import time
from typing import Optional

from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_icl_assembler
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Position,
)
from experiments.recipes.in_context_learning.in_context_learning import (
    num_agents_to_positions,
    ICLTaskGenerator,
    _BuildCfg,
    num_agents_to_positions,
    play_icl,
    replay_icl,
)
import os

curriculum_args = {
    "train": {
        "num_agents": [1, 2, 6, 12],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3],
        "chest_positions": [["N"], ["N", "S"], ["N", "S", "E"]],
        "num_chests": [2, 5, 8],
    },
    "train_pairs": {
        "num_agents": [2, 6, 12],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[2],
        "chest_positions": [["N"]],
        "num_chests": [2, 5, 8],
    },
    "train_triplets": {
        "num_agents": [3, 6, 12],
        "chain_lengths": [2, 3, 4, 5],
        "num_sinks": [0, 1, 2],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[3],
        "chest_positions": [["N"]],
        "num_chests": [2, 5, 8],
    },
    # "test": {
    #     "num_agents": [2],
    #     "chain_lengths": [2],
    #     "num_sinks": [0],
    #     "chest_positions": [["N"]],
    #     "num_chests": [1],
    #     "room_sizes": ["medium"],
    #     "positions": [["Any", "Any"]],
    # }
}


def make_task_generator_cfg(
    num_agents: list[int],
    chain_lengths: list[int],
    num_sinks: list[int],
    room_sizes: list[str],
    positions: list[list[Position]],
    map_dir: Optional[str] = None,
    num_chests: list[int] = [0],
    chest_positions: list[list[Position]] = [["N"]],
):
    return AssemblyLinesTaskGenerator.Config(
        num_agents=num_agents,
        num_resources=[c - 1 for c in chain_lengths],
        num_converters=num_sinks,
        room_sizes=room_sizes,
        positions=positions,
        map_dir=map_dir,
        num_chests=num_chests,
        chest_positions=chest_positions,
    )


class AssemblyLinesTaskGenerator(ICLTaskGenerator):
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
        cooldown = avg_hop * len(resources)
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
    ):
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
        chest_position,
        num_chests,
        terrain,
        max_steps,
        num_instances,
        rng,
        dir=None,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        self._make_resource_chain(resources, width + height / 2, position, cfg, rng)
        self._make_sinks(num_sinks, position, cfg, rng)
        if num_chests > 0:
            self._make_chests(num_chests, cfg, chest_position)

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
            chest_position,
            num_chests,
        ) = self._setup_task(rng)

        dir = (
            f"./train_dir/{self.config.map_dir}/{room_size}/{len(resources)}chain/{num_sinks}sinks/{terrain}"
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
            chest_position=chest_position,
            num_chests=num_chests,
            terrain=terrain,
            max_steps=max_steps,
            num_instances=num_instances or 24 // num_agents,
            rng=rng,
            dir=dir,
        )

        icl_env.label = f"{room_size}_{len(resources)}chain_{num_sinks}sinks_{terrain}"
        return icl_env

    def generate_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        return self._generate_task(task_id, rng, num_instances)


def train(
    curriculum_style: str = "multi_agent_easy",
) -> TrainTool:
    task_generator_cfg = make_task_generator_cfg(
        **curriculum_args[curriculum_style], map_dir=None
    )
    from experiments.evals.in_context_learning.assemblers.assembly_lines import (
        make_assembly_line_eval_suite,
    )

    return train_icl(task_generator_cfg, make_assembly_line_eval_suite)


def play(curriculum_style: str = "test") -> PlayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    return play_icl(task_generator)


def play_eval() -> PlayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(
            num_agents=[2],
            chain_lengths=[5],
            num_sinks=[2],
            room_sizes=["large"],
            positions=[["Any", "Any"]],
        )
    )
    return play_icl(task_generator)


def replay(
    curriculum_style: str = "hard_eval",
) -> ReplayTool:
    task_generator = AssemblyLinesTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    # Default to the research policy if none specified
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_4.2.2025-09-24/icl_resource_chain_terrain_4.2.2025-09-24:v2370.pt"
    default_policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_1.2.2025-09-24/icl_resource_chain_terrain_1.2.2025-09-24:v2070.pt"
    return replay_icl(task_generator, default_policy_uri)


def evaluate():
    from experiments.evals.in_context_learning.assemblers.assembly_lines import (
        make_assembly_line_eval_suite,
    )

    policy_uris = []

    for curriculum_style in curriculum_args:
        policy_uri = f"s3://softmax-public/policies/in_context.assembly_lines_{curriculum_style}.eval_local.2025-09-27/:latest"
        policy_uris.append(policy_uri)

    simulations = make_assembly_line_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.assemblers.assembly_lines.train",
                f"run=in_context.assembly_lines_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
