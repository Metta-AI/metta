import random
from typing import Dict, Any
from dataclasses import dataclass, field
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.builder.envs import make_icl_assembler
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.agent.policies.vit_reset import ViTResetConfig
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    Position,
)
from experiments.recipes.cogs_v_clips.utils import (
    assembly_lines_curriculum_args,
    size_ranges,
    RESOURCES,
    make_assembler,
    make_extractor,
    make_chest,
    make_agent,
)


@dataclass
class _BuildCfg:
    game_objects: Dict[str, Any] = field(default_factory=dict)
    map_builder_objects: Dict[str, int] = field(default_factory=dict)


class AssemblyLinesTaskGenerator(TaskGenerator):
    """Pure foraging, no energy or chargers"""

    class Config(TaskGeneratorConfig["AssemblyLinesTaskGenerator"]):
        num_cogs: list[int]
        chain_length: list[int]
        room_size: list[str]
        positions: list[list[Position]]

    def __init__(self, config: "AssemblyLinesTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.used_resources = set()

    def _get_width_and_height(self, room_size: str, rng: random.Random):
        lo, hi = size_ranges[room_size]
        return rng.randint(lo, hi), rng.randint(lo, hi)

    def _set_width_and_height(self, room_size, num_agents, num_objects, rng):
        """Set the width and height of the environment to be at least the minimum area required for the number of agents, altars, and generators."""
        width, height = self._get_width_and_height(room_size, rng)
        area = width * height
        minimum_area = num_agents + num_objects * 2
        if area < minimum_area:
            width, height = minimum_area // 2, minimum_area // 2
        return width, height

    def _calculate_max_steps(self, chain_length: int, width: int, height: int) -> int:
        avg_hop = width + height / 2

        steps_per_attempt = 4 * avg_hop
        chain_completion_cost = steps_per_attempt * chain_length
        target_completions = 5

        return int(target_completions * chain_completion_cost)

    def add_to_game_cfg(self, object, cfg):
        # add a single object to the game cfg
        cfg.game_objects[object.name] = object
        cfg.map_builder_objects[object.name] = 1

    def _make_env_cfg(
        self,
        num_agents,
        num_instances,
        chain_length,
        width,
        height,
        recipe_position,
        max_steps,
        rng: random.Random = random.Random(),
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        if chain_length == 1:
            # only one assembler and one sink
            assembler_inputs = {}
            assembler_outputs = {"heart": 1}
        else:
            num_extractors = chain_length - 1
            assert num_extractors <= len(RESOURCES), (
                "We do not currently support more than 4 extractors"
            )
            resource_chain = rng.sample(RESOURCES, num_extractors)
            self.used_resources.update(resource_chain)
            for i in range(len(resource_chain) - 1):
                input_resource, output_resource = (
                    resource_chain[i],
                    resource_chain[i + 1],
                )
                input_resources = {} if i == 0 else {input_resource: 1}
                extractor = make_extractor(
                    resource=resource_chain[i],
                    inputs=input_resources,
                    outputs=output_resource,
                    position=recipe_position,
                )
                self.add_to_game_cfg(extractor, cfg)

            # the assembler takes as input the last resource in the extractor chain
            assembler_inputs = {resource_chain[-1]: 1}
            assembler_outputs = {"heart": 1}

        assembler = make_assembler(
            inputs=assembler_inputs,
            outputs=assembler_outputs,
            positions=recipe_position,
        )
        self.add_to_game_cfg(assembler, cfg)
        chest = make_chest(position_deltas=[("N", 1), ("S", 1), ("E", 1), ("W", 1)])
        self.add_to_game_cfg(chest, cfg)

        inventory_rewards = {"heart": 0}
        stat_rewards = {"chest.heart.amount": 1}
        resource_limits = {"heart": 1}
        agent = make_agent(
            stat_rewards=stat_rewards,
            inventory_rewards=inventory_rewards,
            resource_limits=resource_limits,
        )
        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
            resources=list(self.used_resources) + ["heart", "energy"],
            agent=agent,
            inventory_regen_interval=0,
            terrain=rng.choice(["sparse", "balanced", "dense", "no-terrain"]),
        )

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        num_cogs = rng.choice(self.config.num_cogs)
        chain_length = rng.choice(self.config.chain_length)
        room_size = rng.choice(self.config.room_size)
        recipe_position = rng.choice(self.config.positions)

        width, height = self._set_width_and_height(
            room_size, num_cogs, chain_length, rng
        )
        max_steps = self._calculate_max_steps(chain_length, width, height)

        icl_env = self._make_env_cfg(
            num_agents=num_cogs,
            num_instances=24 // num_cogs,
            chain_length=chain_length,
            width=width,
            height=height,
            recipe_position=recipe_position,
            max_steps=max_steps,
            rng=rng,
        )

        icl_env.label = f"{room_size}_{chain_length}_chain"
        return icl_env


def train(curriculum_style: str = "pairs", architecture: str = "lstm_reset") -> TrainTool:
    from experiments.evals.cogs_v_clips.foraging import make_foraging_eval_suite

    task_generator_cfg = AssemblyLinesTaskGenerator.Config(
        **assembly_lines_curriculum_args[curriculum_style]
    )
    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=LearningProgressConfig(
            num_active_tasks=1000,
        ),
    )

    if architecture == "lstm_reset":
        policy_architecture = FastLSTMResetConfig()
    elif architecture == "vit_reset":
        policy_architecture = ViTResetConfig()

    return TrainTool(
        trainer=TrainerConfig(
            losses=LossConfig(),
        ),
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_architecture,
        evaluator=EvaluatorConfig(
            simulations=make_foraging_eval_suite(),
            evaluate_remote=False,
            evaluate_local=True,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_env(
    num_cogs=1,
    position=["Any"],
    chain_length=3,
    sizes="small",
):
    task_generator = AssemblyLinesTaskGenerator(
        config=AssemblyLinesTaskGenerator.Config(
            num_cogs=[num_cogs],
            positions=[position],
            room_size=[sizes],
            chain_length=[chain_length],
        )
    )
    return task_generator.get_task(random.randint(0, 1000000))


def replay() -> ReplayTool:
    eval_env = make_env()
    policy_uri = "s3://softmax-public/policies/cogs_v_clips.level_1.eval_local.multi_agent_pairs_bases_vit_reset.2025-10-02/:latest"

    return ReplayTool(
        policy_uri=policy_uri,
        sim=SimulationConfig(suite="cogs_v_clips", env=eval_env, name="eval"),
    )


def experiment():
    import subprocess
    import time

    for curriculum_style in assembly_lines_curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.cogs_v_clips.assembly_lines.train",
                f"run=cogs_v_clips.assembly_lines.lstmresets.{curriculum_style}.{random.randint(0, 10000)}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


def make_mettagrid(task_generator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "pairs") -> PlayTool:
    task_generator = AssemblyLinesTaskGenerator(
        config=AssemblyLinesTaskGenerator.Config(
            **assembly_lines_curriculum_args[curriculum_style]
        )
    )
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="cogs_vs_clippies", name="play"
        )
    )


if __name__ == "__main__":
    experiment()
