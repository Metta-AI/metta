import os
import random
import subprocess
import time
from typing import Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval_remote import EvalRemoteTool
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
    ICLTaskGenerator,
    LPParams,
    _BuildCfg,
    num_agents_to_positions,
    play_icl,
    replay_icl,
    room_size_templates,
    train_icl,
)


def make_curriculum_args(
    num_agents: list[int],
    num_altars: list[int],
    num_generators: list[int],
    room_sizes: list[str],
    positions: list[list[Position]],
    num_chests: list[int],
    chest_positions: list[list[Position]],
) -> dict:
    return {
        "num_agents": num_agents,
        "num_altars": num_altars,
        "num_generators": num_generators,
        "room_sizes": room_sizes,
        "positions": positions,
        "num_chests": num_chests,
        "chest_positions": chest_positions,
    }


curriculum_args = {
    "two_agent_many_altars": {
        "num_agents": [2],
        "num_altars": list(range(5, 20, 5)),
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[2],
        "num_chests": [2, 5, 8],
        "chest_positions": [["Any"]],
    },
    "three_agent_many_altars": {
        "num_agents": [3],
        "num_altars": list(range(5, 20, 5)),
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
        "positions": num_agents_to_positions[2] + num_agents_to_positions[3],
        "num_chests": [2, 5, 8],
        "chest_positions": [["Any"]],
    },
    "multi_agent_multi_altars": {
        "num_agents": [1, 2, 3],
        "num_altars": list(range(5, 20, 5)),
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
        "num_chests": [2, 5, 8],
        "positions": num_agents_to_positions[1]
        + num_agents_to_positions[2]
        + num_agents_to_positions[3],
        "chest_positions": [["Any"]],
    },
    "two_agents_1g_1a_small": {
        "num_agents": [2],
        "num_altars": [1],
        "num_generators": [1],
        "num_chests": [2],
        "room_sizes": ["small", "medium"],
        "positions": num_agents_to_positions[2],
        "chest_positions": [["Any"]],
    },
    # "two_agents_1g_1a_medium": {
    #     "num_agents": [2],
    #     "num_altars": [1, 2, 5],
    #     "num_generators": [1, 2, 5],
    #     "room_sizes": ["medium", "large"],
    #     "positions": num_agents_to_positions[2],
    # },
    # "multi_agents_1g_1a": {
    #     "num_agents": [1, 2, 3],
    #     "num_altars": [5],
    #     "num_generators": [5],
    #     "room_sizes": ["small", "medium", "large"],
    #     "positions": num_agents_to_positions[1]
    #     + num_agents_to_positions[2]
    #     + num_agents_to_positions[3],
    # },
    # "multi_agents_1g_1a_terrain": {
    #     "num_agents": [1, 2, 3],
    #     "num_altars": list(range(5, 20, 5)),
    #     "num_generators": list(range(5, 20, 5)),
    #     "room_sizes": ["small", "medium", "large"],
    #     "positions": num_agents_to_positions[1] + num_agents_to_positions[2],
    # },
    "test": {
        "num_agents": [3],
        "num_altars": [4],
        "num_generators": [0],
        "num_chests": [2],
        "chest_positions": [["N"]],
        "room_sizes": ["small"],
        "positions": [["N", "S"]],
    },
}


def make_task_generator_cfg(
    num_agents: list[int],
    num_altars: list[int],
    num_generators: list[int],
    room_sizes: list[str],
    positions: list[list[Position]],
    num_chests: list[int],
    chest_positions: list[list[Position]],
    map_dir: Optional[str] = None,
) -> ICLTaskGenerator.Config:
    return ForagingTaskGenerator.Config(
        num_agents=num_agents,
        num_converters=num_altars,
        num_resources=num_generators,
        positions=positions,
        room_sizes=room_sizes,
        map_dir=map_dir,
        num_chests=num_chests,
        chest_positions=chest_positions,
    )


class ForagingTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.used_resources = set()

    def _make_generators(self, num_generators, cfg, position, rng: random.Random):
        """Make generators that input nothing and output resources for the altar"""

        for _ in range(num_generators):
            resource = rng.choice(self.resource_types)
            self.used_resources.add(resource)
            self._add_assembler(
                input_resources={},
                output_resources={resource: 1},
                position=position,
                cfg=cfg,
                rng=rng,
                replacement=True,
            )

    def _make_altars(
        self, num_altars, cfg, position, num_generators, rng: random.Random
    ):
        altar_cooldown = 25 + num_altars * 10 if num_generators == 0 else 1

        if num_generators == 0:
            input_resources = {}
            assembler_name = "altar"
        else:
            input_resources = {
                resource: 1
                for resource in rng.sample(
                    list(self.used_resources),
                    rng.randint(1, len(self.config.max_recipe_inputs)),
                )
            }
            # if we want multiple altars with different recipes, we need to give them different names
            assembler_name = "altar" if num_generators == 1 else None

        for _ in range(num_altars):
            self._add_assembler(
                input_resources=input_resources,
                output_resources={"heart": 1},
                position=position,
                cfg=cfg,
                assembler_name=assembler_name,
                cooldown=altar_cooldown,
                rng=rng,
            )

    def _make_env_cfg(
        self,
        num_agents,
        num_instances,
        num_altars,
        num_generators,
        terrain,
        width,
        height,
        recipe_position,
        num_chests,
        chest_position,
        max_steps,
        rng: random.Random = random.Random(),
        dir=None,
    ) -> MettaGridConfig:
        cfg = _BuildCfg()

        if num_generators > 3 and num_altars > 6:
            num_altars = 6

        self._make_generators(num_generators, cfg, recipe_position, rng)

        self._make_altars(num_altars, cfg, recipe_position, num_generators, rng)

        # self._make_chests(num_chests, cfg, chest_position)

        if dir is not None and os.path.exists(dir):
            print(f"Loading from {dir}")
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
        self, num_altars: int, num_generators: int, width: int, height: int
    ) -> int:
        area = width * height
        max_steps = max(150, area * (num_altars + num_generators) * 2)
        return min(max_steps, 1800)

    def _generate_task(
        self, task_id: int, rng: random.Random, num_instances: Optional[int] = None
    ) -> MettaGridConfig:
        (
            num_agents,
            generators,
            num_altars,
            room_size,
            terrain,
            width,
            height,
            max_steps,
            recipe_position,
            chest_position,
            num_chests,
        ) = self._setup_task(rng)
        # Find the smallest value in templates["num_objects"] that is >= num_objects for correct dir

        if self.config.map_dir is not None:
            num_object_reference = min(
                obj
                for obj in room_size_templates[room_size]["num_objects"]
                if obj >= (num_altars + len(generators))
            )
            dir = f"./train_dir/{self.config.map_dir}/{room_size}/{num_object_reference}objects/{terrain}"
        else:
            dir = None

        icl_env = self._make_env_cfg(
            num_agents=num_agents,
            num_instances=num_instances or 24 // num_agents,
            num_altars=num_altars,
            num_generators=len(generators),
            terrain=terrain,
            width=width,
            height=height,
            recipe_position=recipe_position,
            num_chests=num_chests,
            chest_position=chest_position,
            max_steps=max_steps,
            rng=rng,
            dir=dir,
        )

        icl_env.label = (
            f"{room_size}_{num_altars}altars_{len(generators)}generators_{terrain}"
        )
        return icl_env

    def generate_task(
        self,
        task_id: int,
        rng: random.Random,
        num_instances: Optional[int] = None,
    ) -> MettaGridConfig:
        return self._generate_task(task_id, rng, num_instances)


def make_mettagrid(
    curriculum_style: str = "single_agent_two_altars",
) -> MettaGridConfig:
    task_generator_cfg = ForagingTaskGenerator.Config(
        **make_curriculum_args(**curriculum_args[curriculum_style])
    )
    task_generator = ForagingTaskGenerator(task_generator_cfg)
    return task_generator.get_task(random.randint(0, 1000000))


def make_assembler_env(
    num_agents: int,
    num_altars: int,
    num_generators: int,
    num_chests: int,
    chest_positions: list[Position],
    room_size: str,
    position: list[Position] = ["Any"],
    chest_position: list[Position] = ["N"],
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        num_agents=[num_agents],
        num_altars=[num_altars],
        num_generators=[num_generators],
        positions=[position],
        room_sizes=[room_size],
        num_chests=[num_chests],
        chest_positions=[chest_positions],
    )
    task_generator = ForagingTaskGenerator(task_generator_cfg)
    return task_generator.get_task(random.randint(0, 1000000))


def make_curriculum(
    num_agents: list[int] = [1, 2],
    num_altars: list[int] = [2],
    num_generators: list[int] = [0, 1, 2],
    room_sizes: list[str] = ["small", "medium", "large"],
    positions: list[list[Position]] = [["Any"], ["Any", "Any"]],
    num_chests: list[int] = [2],
    chest_positions: list[list[Position]] = [["Any"]],
) -> CurriculumConfig:
    task_generator_cfg = make_task_generator_cfg(
        num_agents=num_agents,
        num_altars=num_altars,
        num_generators=num_generators,
        room_sizes=room_sizes,
        positions=positions,
        num_chests=num_chests,
        chest_positions=chest_positions,
    )
    return CurriculumConfig(task_generator=task_generator_cfg)


def train(
    curriculum_style: str = "single_agent_two_altars", lp_params: LPParams = LPParams()
) -> TrainTool:
    task_generator_cfg = make_task_generator_cfg(
        **curriculum_args[curriculum_style], map_dir=None
    )
    from experiments.evals.in_context_learning.assemblers.foraging import (
        make_foraging_eval_suite,
    )

    return train_icl(task_generator_cfg, make_foraging_eval_suite, lp_params)


def evaluate(simulations: Optional[Sequence[SimulationConfig]] = None) -> SimTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.assemblers.foraging import (
        make_foraging_eval_suite,
    )

    policy_uris = [
        "s3://softmax-public/policies/in_context.all_assemblers.eval_remote.2025-09-29/in_context.all_assemblers.eval_remote.2025-09-29/:latest.pt",
        "s3://softmax-public/policies/in_context.foraging_train.2025-09-30/in_context.foraging_train.2025-09-30/:latest.pt",
        "s3://softmax-public/policies/in_context.assembly_lines_train.2025-09-29/in_context.assembly_lines_train.2025-09-29/:latest.pt",
        "s3://softmax-public/policies/in_context.all_assemblers.2025-09-29/in_context.all_assemblers.2025-09-29/:latest.pt",
    ]
    for curriculum_style in curriculum_args:
        policy_uris.append(
            f"s3://softmax-public/policies/in_context.foraging_{curriculum_style}.eval_local.2025-09-27/in_context.foraging_{curriculum_style}.eval_local.2025-09-27/:latest.pt"
        )

    print(f"Policy uris:{policy_uris}")

    simulations = simulations or make_foraging_eval_suite()
    return SimTool(
        simulations=simulations,
        policy_uris=policy_uris,
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def play_eval() -> PlayTool:
    env = make_assembler_env(
        num_agents=1,
        num_altars=2,
        num_generators=0,
        num_chests=2,
        chest_positions=["W"],
        room_size="small",
        position=["W"],
    )

    return PlayTool(
        sim=SimulationConfig(
            env=env,
            name="in_context_assemblers",
            suite="in_context_learning",
        ),
    )


def replay(curriculum_style: str = "test") -> ReplayTool:
    task_generator = ForagingTaskGenerator(
        make_task_generator_cfg(
            **make_curriculum_args(**curriculum_args[curriculum_style])
        )
    )
    policy_uri = "s3://softmax-public/policies/icl_resource_chain_terrain_4.2.2025-09-24/icl_resource_chain_terrain_4.2.2025-09-24:v2370.pt"
    return replay_icl(task_generator, policy_uri)


def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.foraging.train",
                f"run=in_context.foraging_{curriculum_style}.eval_local.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


def play(
    curriculum_style: str = "test",
) -> PlayTool:
    task_generator = ForagingTaskGenerator(
        make_task_generator_cfg(
            **make_curriculum_args(**curriculum_args[curriculum_style])
        )
    )
    return play_icl(task_generator)


def evaluate_remote(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> EvalRemoteTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.assemblers.foraging import (
        make_foraging_eval_suite,
    )

    simulations = simulations or make_foraging_eval_suite()
    return EvalRemoteTool(
        simulations=simulations,
        policy_uri=policy_uri,
    )


if __name__ == "__main__":
    evaluate()
