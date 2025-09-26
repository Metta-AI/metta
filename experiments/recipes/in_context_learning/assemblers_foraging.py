import random
import subprocess
import time
from typing import Optional, Sequence

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.sim.simulation_config import SimulationConfig
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
    train_icl,
    play_icl,
    replay_icl,
    _BuildCfg,
    num_agents_to_positions,
)


def make_curriculum_args(
    num_agents: list[int],
    num_altars: list[int],
    num_generators: list[int],
    room_sizes: list[str],
    positions: list[list[Position]],
) -> dict:
    return {
        "num_agents": num_agents,
        "num_altars": num_altars,
        "num_generators": num_generators,
        "room_sizes": room_sizes,
        "positions": positions,
    }


def calculate_max_steps(num_objects: int, width: int, height: int) -> int:
    area = width * height
    max_steps = max(150, area * num_objects * 2)
    return min(max_steps, 1800)


curriculum_args = {
    "single_agent_two_altars": {
        "num_agents": [1],
        "num_altars": [2],
        "num_generators": [0],
        "room_sizes": ["small"],
    },
    "single_agent_many_altars": {
        "num_agents": [1],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
    },
    "single_agent_many_altars_with_any": {
        "num_agents": [1],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
        "include_Any": True,
    },
    "two_agent_5_altars_pattern": {
        "num_agents": [2],
        "num_altars": [5],
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
    },
    "two_agent_two_altars_pattern": {
        "num_agents": [2],
        "num_altars": [2],
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
    },
    "two_agent_two_altars_progressive_pattern": {
        "num_agents": [2],
        "num_altars": [2],
        "num_generators": [0],
        "room_sizes": ["small"],
        "generator_positions": [["Any"]],
        "altar_positions": [
            ["N", "S", "E"],
            ["N", "W", "E"],
            ["S", "E", "W"],
            ["S", "W", "E"],
            ["Any"],
            ["N", "S"],
            ["E", "W"],
            ["N", "E"],
            ["N", "W"],
            ["S", "E"],
            ["S", "W"],
        ],
    },
    "two_agent_two_altars_any": {
        "num_agents": [2],
        "num_altars": [2],
        "num_generators": [0],
        "room_sizes": ["small"],
        "generator_positions": [["Any"]],
        "altar_positions": [["Any"]],
    },
    "three_agents_two_altars": {
        "num_agents": [3],
        "num_altars": [2, 4],
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
        "include_Any": False,
    },
    "three_agent_many_altars_with_any": {
        "num_agents": [3],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
        "include_Any": True,
    },
    "multi_agent_multi_altars": {
        "num_agents": [1, 2, 3],
        "num_altars": list(range(4, 16, 2)),
        "num_generators": [0],
        "room_sizes": ["small", "medium", "large"],
    },
    "two_agents_1g_1a": {
        "num_agents": [2],
        "num_altars": [1],
        "num_generators": [1],
        "room_sizes": ["small", "medium", "large"],
    },
    "multi_agents_1g_1a": {
        "num_agents": [1, 2, 3],
        "num_altars": [1],
        "num_generators": [1],
        "room_sizes": ["small", "medium", "large"],
    },
    "multi_agents_1g_1a_with_any": {
        "num_agents": [1, 2, 3],
        "num_altars": [1],
        "num_generators": [1],
        "room_sizes": ["small", "medium", "large"],
        "include_Any": True,
    },
    "three_agents_1g_1a": {
        "num_agents": [3],
        "num_altars": [1],
        "num_generators": [1],
        "room_sizes": ["small", "medium", "large"],
    },
    "test": {
        "num_agents": [10],
        "num_altars": [10],
        "num_generators": [0],
        "room_sizes": ["xlarge"],
        "positions": [["N", "S"]],
    },
}


def make_task_generator_cfg(
    num_agents: list[int],
    num_altars: list[int],
    num_generators: list[int],
    room_sizes: list[str],
    positions: list[list[Position]],
) -> ICLTaskGenerator.Config:
    return AssemblerTaskGenerator.Config(
        num_agents=num_agents,
        num_converters=num_altars,
        num_resources=num_generators,
        positions=positions,
        room_sizes=room_sizes,
        obstacle_types=[],
        densities=[],
        map_dir=None,
    )


class AssemblerTaskGenerator(ICLTaskGenerator):
    def __init__(self, config: "ICLTaskGenerator.Config"):
        super().__init__(config)
        self.config = config
        self.num_altars = config.num_converters
        self.num_generators = config.num_resources

    def _make_generators(self, num_generators, cfg, position, rng: random.Random):
        """Make generators that input nothing and output resources for the altar"""

        for _ in range(num_generators):
            resource = rng.choice(self.resource_types)
            self._add_assembler(
                input_resources={},
                output_resources={resource: 1},
                position=position,
                cfg=cfg,
                rng=rng,
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
                    self.resource_types, rng.randint(1, len(self.resource_types))
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

    def make_env_cfg(
        self,
        num_agents,
        num_instances,
        num_altars,
        num_generators,
        width,
        height,
        recipe_position,
        max_steps,
        rng: random.Random = random.Random(),
    ) -> MettaGridConfig:
        cfg = _BuildCfg()
        self._make_generators(num_generators, cfg, recipe_position, rng)

        print(f"Num altars: {num_altars}")

        self._make_altars(num_altars, cfg, recipe_position, num_generators, rng)

        return make_icl_assembler(
            num_agents=num_agents,
            num_instances=num_instances,
            max_steps=max_steps,
            game_objects=cfg.game_objects,
            map_builder_objects=cfg.map_builder_objects,
            width=width,
            height=height,
        )

    # TODO
    def _set_width_and_height(
        self, room_size, num_agents, num_altars, num_generators, rng
    ):
        width, height = self._get_width_and_height(room_size, rng)
        area = width * height
        minimum_area = (num_agents + num_altars + num_generators) * 2
        if area < minimum_area:
            width, height = minimum_area // 2, minimum_area // 2
        return width, height

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        num_agents = rng.choice(self.config.num_agents)

        # positions must be the same length as the number of agents
        recipe_position = rng.choice(
            [p for p in self.config.positions if len(p) <= num_agents]
        )

        num_altars = rng.choice(self.num_altars)
        num_generators = rng.choice(self.num_generators)
        room_size = rng.choice(self.config.room_sizes)
        width, height = self._set_width_and_height(
            room_size, num_agents, num_altars, num_generators, rng
        )
        max_steps = calculate_max_steps(
            num_agents + num_altars + num_generators, width, height
        )

        num_instances = 24 // num_agents

        return self.make_env_cfg(
            num_agents,
            num_instances,
            num_altars,
            num_generators,
            width,
            height,
            recipe_position,
            max_steps,
            rng,
        )


def make_mettagrid(
    curriculum_style: str = "single_agent_two_altars",
) -> MettaGridConfig:
    task_generator_cfg = AssemblerTaskGenerator.Config(
        **make_curriculum_args(**curriculum_args[curriculum_style])
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(random.randint(0, 1000000))


def make_assembler_env(
    num_agents: int,
    num_altars: int,
    num_generators: int,
    room_size: str,
    position: list[Position] = ["Any"],
) -> MettaGridConfig:
    task_generator_cfg = make_task_generator_cfg(
        num_agents=[num_agents],
        num_altars=[num_altars],
        num_generators=[num_generators],
        positions=[position],
        room_sizes=[room_size],
    )
    task_generator = AssemblerTaskGenerator(task_generator_cfg)
    return task_generator.get_task(random.randint(0, 1000000))


def make_curriculum(
    num_agents: list[int] = [1, 2],
    num_altars: list[int] = [2],
    num_generators: list[int] = [0, 1, 2],
    room_sizes: list[str] = ["small", "medium", "large"],
    positions: list[list[Position]] = [["Any"], ["Any", "Any"]],
) -> CurriculumConfig:
    task_generator_cfg = make_task_generator_cfg(
        num_agents=num_agents,
        num_altars=num_altars,
        num_generators=num_generators,
        room_sizes=room_sizes,
        positions=positions,
    )
    return CurriculumConfig(task_generator=task_generator_cfg)


def train(
    curriculum_style: str = "single_agent_two_altars", lp_params: LPParams = LPParams()
) -> TrainTool:
    task_generator_cfg = make_task_generator_cfg(
        **make_curriculum_args(**curriculum_args[curriculum_style])
    )
    from experiments.evals.in_context_learning.assembler_foraging import (
        make_assembler_eval_suite,
    )

    return train_icl(task_generator_cfg, make_assembler_eval_suite, lp_params)


def evaluate(
    policy_uri: str, simulations: Optional[Sequence[SimulationConfig]] = None
) -> SimTool:
    # Local import to avoid circular import at module load time
    from experiments.evals.in_context_learning.assembler_foraging import (
        make_assembler_eval_suite,
    )

    policy_uris = []
    for curriculum_style in curriculum_args:
        policy_uris.append(
            f"s3://softmax-public/policies/george.icl_assemblers_{curriculum_style}.2025-09-25/george.icl_assemblers_{curriculum_style}.2025-09-25:latest.pt"
        )

    simulations = simulations or make_assembler_eval_suite()
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


def replay(curriculum_style: str = "single_agent_two_altars") -> ReplayTool:
    task_generator = AssemblerTaskGenerator(
        make_task_generator_cfg(
            **make_curriculum_args(**curriculum_args[curriculum_style])
        )
    )
    policy_uri = "s3://softmax-public/policies/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22/icl_assemblers3_two_agent_two_altars_pattern.2025-09-22:v500.pt"
    return replay_icl(task_generator, policy_uri)


def experiment():
    for curriculum_style in curriculum_args:
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.in_context_learning.assemblers.train",
                f"run=george.icl_assemblers_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
                "--gpus=4",
                "--heartbeat-timeout=3600",
                "--skip-git-check",
            ]
        )
        time.sleep(1)


def play(
    curriculum_style: str = "multi_agent_multi_altars",
) -> PlayTool:
    task_generator = AssemblerTaskGenerator(
        make_task_generator_cfg(
            **make_curriculum_args(**curriculum_args[curriculum_style])
        )
    )
    return play_icl(task_generator)


if __name__ == "__main__":
    experiment()
