from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission
from metta.agent.policies.vit import ViTDefaultConfig
from metta.tools.play import PlayTool
from metta.sim.simulation_config import SimulationConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
from metta.rl.loss import LossConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.tools.train import TrainTool
import random
from pathlib import Path
import yaml
from cogames.cogs_vs_clips import vibes
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    AssemblerConfig,
    ProtocolConfig,
    ChestConfig,
)
from cogames.cogs_vs_clips.stations import CvCChestConfig, resources, ChargerConfig
import copy
import subprocess
import time
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.builder import empty_assemblers
from mettagrid.config.mettagrid_config import (
    GameConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeGlyphActionConfig,
    ActionConfig,
)


curriculum_args = {
    "assembler_with_chest": {
        "extractors": [[]],
        "use_charger": [False],
        "use_chest": [True],
        "use_extractor_glyphs": [False],
        "efficiences": [100],
        "max_uses": [1000],
    },
    "one_extractor": {
        "extractors": [["carbon"], ["oxygen"], ["germanium"], ["silicon"]],
        "use_charger": [False],
        "use_chest": [True],
        "use_extractor_glyphs": [False],
        "efficiences": [100],
        "max_uses": [1000],
    },
    "two_extractors": {
        "extractors": [["carbon"], ["oxygen"], ["germanium"], ["silicon"]]
        + [
            ["carbon", "oxygen"],
            ["carbon", "germanium"],
            ["carbon", "silicon"],
            ["oxygen", "germanium"],
            ["oxygen", "silicon"],
            ["germanium", "silicon"],
        ],
        "use_charger": [False],
        "use_chest": [True],
        "use_extractor_glyphs": [False],
        "efficiences": [100],
        "max_uses": [1000],
    },
    "three_extractors": {
        "extractors": [["carbon"], ["oxygen"], ["germanium"], ["silicon"]]
        + [
            ["carbon", "oxygen"],
            ["carbon", "germanium"],
            ["carbon", "silicon"],
            ["oxygen", "germanium"],
            ["oxygen", "silicon"],
            ["germanium", "silicon"],
        ]
        + [
            ["carbon", "oxygen", "germanium"],
            ["carbon", "oxygen", "silicon"],
            ["carbon", "germanium", "silicon"],
            ["oxygen", "germanium", "silicon"],
        ],
        "use_charger": [False],
        "use_chest": [True],
        "use_extractor_glyphs": [False],
        "efficiences": [100],
        "max_uses": [1000],
    },
    "four_extractors": {
        "extractors": [["carbon"], ["oxygen"], ["germanium"], ["silicon"]]
        + [
            ["carbon", "oxygen"],
            ["carbon", "germanium"],
            ["carbon", "silicon"],
            ["oxygen", "germanium"],
            ["oxygen", "silicon"],
            ["germanium", "silicon"],
        ]
        + [
            ["carbon", "oxygen", "germanium"],
            ["carbon", "oxygen", "silicon"],
            ["carbon", "germanium", "silicon"],
            ["oxygen", "germanium", "silicon"],
        ]
        + [["carbon", "oxygen", "germanium", "silicon"]],
        "use_charger": [False],
        "use_chest": [True],
        "use_extractor_glyphs": [False],
        "efficiences": [100],
        "max_uses": [1000],
    },
}


base_map_data = """
...................
..###############..
..#.............#..
..#.c.........o.#..
..#.............#..
..#......+......#..
..#......@......#..
..#.............#..
.......@.&.@.......
...................
.........@.........
..#......=......#..
..#.............#..
..#.g.........s.#..
..#.............#..
..###############..
...................
"""


def get_map(data: str) -> MapBuilderConfig:
    map_path = Path("packages/cogames/src/cogames/maps/machina1/base.map")
    with map_path.open("r", encoding="utf-8") as f:
        map_data = f.read()
    parsed = yaml.safe_load(map_data)
    print(f"Parsed: {parsed}")
    parsed["map_data"] = data.replace(" ", "")[1:-1]
    print(f"Parsed: {parsed}")
    map = MapBuilderConfig.model_validate(parsed)
    return map


def edit_map(extractors, use_charger, use_chest):
    map_data = copy.deepcopy(base_map_data)
    if not "c" in extractors:
        map_data = map_data.replace("c", ".")
    if not "o" in extractors:
        map_data = map_data.replace("o", ".")
    if not "g" in extractors:
        map_data = map_data.replace("g", ".")
    if not "s" in extractors:
        map_data = map_data.replace("s", ".")
    if not use_chest:
        map_data = map_data.replace("=", ".")
    if not use_charger:
        map_data = map_data.replace("+", ".")

    # Remove 3 out of 4 "@" symbols, randomly
    agent_indices = [i for i, c in enumerate(map_data) if c == "@"]
    remove_indices = random.sample(agent_indices, 3)
    map_data = "".join(
        c if i not in remove_indices else "." for i, c in enumerate(map_data)
    )

    print(f"Map data: {map_data}")

    map = get_map(map_data)

    print(f"Map: {map}")
    return map


def load_machina_1():
    machina1 = Machina1OpenWorldMission()
    map_builder = get_map("machina_100_stations.map")
    return machina1.instantiate(map_builder, 2).make_env()


def make_assembler(inputs: dict[str, int]) -> AssemblerConfig:
    # when we have more agents, we will adjust the inputs based on the number of agents in the recipe
    assembler = AssemblerConfig(
        name="assembler",
        type_id=8,
        map_char="&",
        render_symbol="🔄",
        recipes=[
            (
                ["heart"],
                ProtocolConfig(input_resources=inputs, output_resources={"heart": 1}),
            )
        ],
    )
    return assembler


def make_extractor(
    name, inputs, outputs, max_uses, cooldown, allow_partial_usage
) -> AssemblerConfig:
    name_to_type_id = {
        "carbon_extractor": 2,
        "oxygen_extractor": 3,
        "germanium_extractor": 15,
        "silicon_extractor": 15,
    }
    name_to_map_char = {
        "carbon_extractor": "C",
        "oxygen_extractor": "O",
        "germanium_extractor": "G",
        "silicon_extractor": "S",
    }

    return AssemblerConfig(
        name=name,
        type_id=name_to_type_id[name],
        map_char=name_to_map_char[name],
        render_symbol=vibes.VIBE_BY_NAME[name].symbol,
        max_uses=max_uses,
        allow_partial_usage=allow_partial_usage,
        recipes=[
            (
                [],
                ProtocolConfig(
                    input_resources=inputs, output_resources=outputs, cooldown=cooldown
                ),
            )
        ],
    )


def make_chest() -> ChestConfig:
    return CvCChestConfig().station_cfg()


def make_charger() -> AssemblerConfig:
    return ChargerConfig().station_cfg()


def machina1_assembler() -> AssemblerConfig:
    inputs = {
        "carbon": 20,
        "oxygen": 20,
        "germanium": 5,
        "silicon": 50,
        "energy": 20,
    }
    return make_assembler(inputs)


def machina1_carbon_extractor(
    efficiency, max_uses, cooldown=10, allow_partial_usage=False
) -> AssemblerConfig:
    outputs = {"carbon": 4 * efficiency // 100}
    return make_extractor(
        "carbon_extractor",
        inputs={},
        outputs=outputs,
        max_uses=max_uses,
        cooldown=cooldown,
        allow_partial_usage=allow_partial_usage,
    )


def machina1_oxygen_extractor(
    efficiency, max_uses, allow_partial_usage=True
) -> AssemblerConfig:
    outputs = {"oxygen": 20}
    cooldown = int(10_000 / efficiency)
    return make_extractor(
        "oxygen_extractor",
        inputs={},
        outputs=outputs,
        max_uses=max_uses,
        cooldown=cooldown,
        allow_partial_usage=allow_partial_usage,
    )


def machina1_germanium_extractor(
    efficiency, max_uses=1, allow_partial_usage=False, cooldown=0
) -> AssemblerConfig:
    outputs = {"germanium": efficiency}
    return make_extractor(
        "germanium_extractor",
        inputs={},
        outputs=outputs,
        max_uses=max_uses,
        cooldown=cooldown,
        allow_partial_usage=allow_partial_usage,
    )


def machina1_silicon_extractor(
    efficiency, max_uses, cooldown=0, allow_partial_usage=False, use_charger=True
) -> AssemblerConfig:
    outputs = {"silicon": max(1, int(25 * efficiency // 100))}
    if use_charger:
        inputs = {"energy": 25}
    else:
        inputs = {}
    max_uses = max(1, max_uses // 10)
    return make_extractor(
        "silicon_extractor",
        inputs=inputs,
        outputs=outputs,
        max_uses=max_uses,
        cooldown=cooldown,
        allow_partial_usage=allow_partial_usage,
    )


class Machina1BaseTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["Machina1BaseTaskGenerator"]):
        extractors: list[list[str]]
        use_charger: list[bool]
        use_chest: list[bool]
        use_extractor_glyphs: list[
            bool
        ]  # False - generally no glyphs except for germanium, function of synergy (?)
        efficiences: list[int]  # default 100
        max_uses: list[int]  # default 1000

    def __init__(self, config: "Machina1BaseTaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _setup_task(self, rng: random.Random):
        extractors = rng.choice(self.config.extractors)
        use_charger = rng.choice(self.config.use_charger)
        use_chest = rng.choice(self.config.use_chest)
        use_extractor_glyphs = rng.choice(self.config.use_extractor_glyphs)
        efficiences = rng.choice(self.config.efficiences)
        max_uses = rng.choice(self.config.max_uses)
        return (
            extractors,
            use_charger,
            use_chest,
            use_extractor_glyphs,
            efficiences,
            max_uses,
        )

    def _get_assembler_inputs(self, extractors, use_charger):
        inputs = {}
        if "carbon" in extractors:
            inputs["carbon"] = 20
        if "oxygen" in extractors:
            inputs["oxygen"] = 20
        if "germanium" in extractors:
            inputs["germanium"] = 5
        if "silicon" in extractors:
            inputs["silicon"] = 50
        if use_charger:
            inputs["energy"] = 20
        return inputs

    def _get_agent_rewards(self, use_chest: bool):
        if use_chest:
            return {"chest.heart.amount": 1}
        else:
            return {"heart.amount": 1}

    def _make_env_cfg(
        self,
        map,
        max_steps: int,
        game_objects: dict,
        agent_rewards: dict,
        consumed_resources: dict,
        heart_capacity: int = 1,
        energy_capacity: int = 100,
        cargo_capacity: int = 100,
        energy_regen_amount: int = 1,
        inventory_regen_interval: int = 1,
    ) -> MettaGridConfig:
        cfg = MettaGridConfig(
            game=GameConfig(
                max_steps=max_steps,
                map_builder=map,
                num_agents=1,
                resource_names=resources,
                vibe_names=[vibe.name for vibe in vibes.VIBES],
                actions=ActionsConfig(
                    move=ActionConfig(consumed_resources=consumed_resources),
                    noop=ActionConfig(),
                    change_glyph=ChangeGlyphActionConfig(
                        number_of_glyphs=len(vibes.VIBES)
                    ),
                ),
                agent=AgentConfig(
                    resource_limits={
                        "heart": heart_capacity,
                        "energy": energy_capacity,
                        ("carbon", "oxygen", "germanium", "silicon"): cargo_capacity,
                    },
                    rewards=AgentRewards(stats=agent_rewards),
                    initial_inventory={"energy": energy_capacity},
                    shareable_resources=["energy"],
                    inventory_regen_amounts={"energy": energy_regen_amount},
                ),
                inventory_regen_interval=inventory_regen_interval,
                objects=game_objects,
            )
        )

        return cfg

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        game_objects = {}
        game_objects["wall"] = empty_assemblers.wall
        (extractors, use_charger, use_chest, _, efficiency, max_uses) = (
            self._setup_task(rng)
        )
        map = edit_map(extractors, use_charger, use_chest)
        assembler_inputs = self._get_assembler_inputs(extractors, use_charger)
        assembler = make_assembler(assembler_inputs)
        game_objects["assembler"] = assembler

        extractor_configs = []

        for extractor_name in extractors:
            if extractor_name == "carbon":
                extractor = machina1_carbon_extractor(efficiency, max_uses)
            elif extractor_name == "oxygen":
                extractor = machina1_oxygen_extractor(efficiency, max_uses)
            elif extractor_name == "germanium":
                extractor = machina1_germanium_extractor(efficiency, max_uses)
            elif extractor_name == "silicon":
                extractor = machina1_silicon_extractor(
                    efficiency, max_uses, use_charger=use_charger
                )
            extractor_configs.append(extractor)
            game_objects[extractor_name] = extractor

        if use_chest:
            game_objects["chest"] = make_chest()
        if use_charger:
            game_objects["charger"] = make_charger()
            consumed_resources = {"energy": 25}
        else:
            consumed_resources = {}

        if len(extractors) == 0:
            max_steps = 200
        elif len(extractors) == 1:
            max_steps = 300
        elif len(extractors) == 2:
            max_steps = 500
        elif len(extractors) == 3:
            max_steps = 750
        elif len(extractors) == 4:
            max_steps = 1000

        cfg = self._make_env_cfg(
            map,
            max_steps=max_steps,
            game_objects=game_objects,
            agent_rewards=self._get_agent_rewards(use_chest),
            consumed_resources=consumed_resources,
            heart_capacity=1,
            energy_capacity=100,
            cargo_capacity=100,
            energy_regen_amount=1,
            inventory_regen_interval=1,
        )
        return cfg


def make_task_generator_cfg(
    extractors, use_charger, use_chest, use_extractor_glyphs, efficiences, max_uses
) -> Machina1BaseTaskGenerator.Config:
    return Machina1BaseTaskGenerator.Config(
        extractors=extractors,
        use_charger=use_charger,
        use_chest=use_chest,
        use_extractor_glyphs=use_extractor_glyphs,
        efficiences=efficiences,
        max_uses=max_uses,
    )


def make_mettagrid(task_generator: Machina1BaseTaskGenerator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "only_assembler"):
    task_generator = Machina1BaseTaskGenerator(
        make_task_generator_cfg(**curriculum_args[curriculum_style])
    )
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="machina1", name="play"
        )
    )


def train(curriculum_style: str = "only_assembler"):
    from experiments.evals.machina1.single_agent_base import (
        make_single_agent_base_eval_suite,
    )

    task_generator_cfg = make_task_generator_cfg(**curriculum_args[curriculum_style])
    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=LearningProgressConfig(),
    )
    policy_config = ViTDefaultConfig()

    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(simulations=make_single_agent_base_eval_suite()),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    for curriculum_style in curriculum_args.keys():
        subprocess.run(
            [
                "./devops/skypilot/launch.py",
                "experiments.recipes.machina1.single_agent_base.train",
                f"run=machina1_{curriculum_style}.{time.strftime('%Y-%m-%d')}",
                f"curriculum_style={curriculum_style}",
            ]
        )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
