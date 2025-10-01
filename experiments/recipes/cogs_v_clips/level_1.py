# Level 1:
# no buildings give you energy
# resource give you energy of some amount
# running out of energy is not super punishing


# Level 2: now you can go out into the wilderness
# resources can run out in a way that matters -> things in your base have max_uses,
# slow regneration, different cooldowns,

# Level 3: clipping happens, always have exactly the same item needed to unclip, always a magnetizer

# Level 4: four potential resources that you need to do unclipping with


# First experimrent: just a map with assemblers (input energy output hearts), chests (input hearts), and chargers (output energy)
# curriculum over depletion rate, and number of these objects
# if multiagent, option for agents to share energy with each other


# resource types
# inventory resources: energy, heart, carbon, oxygen, silicon, germanium
# map builder resources: any resource type
# energy + four resource types (carbon, oxygen, silicon, germanium)

# easy recipe: takes energy to make a heart
# hardest recipe: some number of some resources and four cogs to make a heart

# resource extractors
import subprocess
import time

from mettagrid.config.mettagrid_config import MettaGridConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
import random
from mettagrid.config.mettagrid_config import Field
from mettagrid.config.mettagrid_config import Position
from cogames.cogs_vs_clips.scenarios import make_game
from metta.tools.play import PlayTool
from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import ChestConfig
from metta.tools.train import TrainTool
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.trainer_config import TrainerConfig
from metta.rl.loss import LossConfig
from metta.agent.policies.vit_reset import ViTResetConfig
from metta.agent.policies.fast_lstm_reset import FastLSTMResetConfig
from metta.agent.policies.vit_sliding_trans import ViTSlidingTransConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from mettagrid.mapgen.mapgen import MapGen
from metta.map.terrain_from_numpy import CogsVClippiesFromNumpy

# ADDING TERRAIN

# base has all the assemblers, in the middle, surrounded by walls
# nano-assembler which maeks hearts
# extractors are also in the base -> everything in the base

# second -> in all the cardinal directions there are holes in the walls so you can go out and forage for resources
# outside of the base there are assemblers with better recipes than assemblers in the base

curriculum_args = {
    "multi_agent_singles": {
        "num_cogs": [2, 4, 6, 8, 12],
        "assembler_positions": [["Any"]],
        "num_chargers": [3, 5, 10, 15],
        "charger_positions": [["Any"]],
        "carbon_extractor_positions": [["Any"]],
        "oxygen_extractor_positions": [["Any"]],
        "germanium_extractor_positions": [["Any"]],
        "silicon_extractor_positions": [["Any"]],
        "num_chests": [1, 5, 10],
        "chest_positions": [["N"]],
        "regeneration_rate": [1, 2, 3, 4],
        "shareable_energy": [False],
        "use_terrain": [True, False],
        "sizes": ["small", "medium"],
    },
    "multi_agent_pairs": {
        "num_cogs": [2, 4, 6, 8, 12],
        "assembler_positions": [["Any", "Any"]],
        "num_chargers": [3, 5, 10, 15],
        "charger_positions": [["Any", "Any"]],
        "carbon_extractor_positions": [["Any", "Any"]],
        "oxygen_extractor_positions": [["Any", "Any"]],
        "germanium_extractor_positions": [["Any", "Any"]],
        "silicon_extractor_positions": [["Any", "Any"]],
        "num_chests": [1, 5, 10],
        "chest_positions": [["N", "S"]],
        "regeneration_rate": [1, 2, 3, 4],
        "shareable_energy": [True],
        "use_terrain": [True, False],
        "sizes": ["small", "medium"],
    },
    "multi_agent_triplets": {
        "num_cogs": [3, 6, 8, 12, 24],
        "assembler_positions": [["Any", "Any"]],
        "num_chargers": [3, 5, 10, 15],
        "charger_positions": [["Any", "Any", "Any"]],
        "carbon_extractor_positions": [["Any", "Any", "Any"]],
        "oxygen_extractor_positions": [["Any", "Any", "Any"]],
        "num_germanium_extractors": [0, 0, 0, 5, 10],
        "germanium_extractor_positions": [["Any", "Any", "Any"]],
        "silicon_extractor_positions": [["Any", "Any", "Any"]],
        "num_chests": [1, 5, 10],
        "chest_positions": [["N", "S", "E"]],
        "regeneration_rate": [1, 2, 3, 4],
        "shareable_energy": [True],
        "use_terrain": [True, False],
        "sizes": ["small", "medium"],
    },
    # "test":
    #     {
    #     "num_cogs": [2],
    #     "assembler_positions": [["Any", "Any"]],
    #     "num_chargers": [3],
    #     "charger_positions": [["Any"]],
    #     "carbon_extractor_positions": [["Any"]],
    #     "oxygen_extractor_positions": [["Any"]],
    #     "germanium_extractor_positions": [["Any"]],
    #     "silicon_extractor_positions": [["Any"]],
    #     "num_chests": [1],
    #     "chest_positions": [["N"]],
    #     "regeneration_rate": [5],
    #     "shareable_energy": [True],
    #     "use_terrain": [True],
    #     "sizes": ["small"]
    #     }
}


# agent.inventory.resource_limits:
# base resources to have a limit that would be a coule 100, second level things like pickaxe have 1 or 2
# hearts have a limit of 5

# resource limits should be a function of how many agents and how many assemblers are in the env

evals = {
    "single_agent_no_terrain": {
        "num_cogs": 1,
        "assembler_positions": ["Any"],
        "num_chargers": 5,
        "charger_positions": ["Any"],
        "carbon_extractor_positions": ["Any"],
        "oxygen_extractor_positions": ["Any"],
        "germanium_extractor_positions": ["Any"],
        "silicon_extractor_positions": ["Any"],
        "num_chests": 5,
        "chest_positions": ["N"],
        "regeneration_rate": 10,
        "shareable_energy": False,
        "use_terrain": False,
    },
    "two_agent_pairs_no_terrain": {
        "num_cogs": 2,
        "assembler_positions": ["Any", "Any"],
        "num_chargers": 5,
        "charger_positions": ["Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any"],
        "num_chests": 5,
        "chest_positions": ["N", "S"],
        "regeneration_rate": 10,
        "shareable_energy": True,
        "use_terrain": False,
    },
    "three_agent_triplets_no_terrain": {
        "num_cogs": 3,
        "assembler_positions": ["Any", "Any"],
        "num_chargers": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "num_chests": 5,
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 10,
        "shareable_energy": True,
        "use_terrain": False,
    },
    "many_agent_triplets_no_terrain": {
        "num_cogs": 12,
        "assembler_positions": ["Any", "Any"],
        "num_chargers": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "num_chests": 5,
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 10,
        "shareable_energy": True,
        "use_terrain": False,
    },
    "single_agent_terrain_small": {
        "num_cogs": 1,
        "assembler_positions": ["Any"],
        "num_chargers": 5,
        "charger_positions": ["Any"],
        "carbon_extractor_positions": ["Any"],
        "oxygen_extractor_positions": ["Any"],
        "germanium_extractor_positions": ["Any"],
        "silicon_extractor_positions": ["Any"],
        "num_chests": 5,
        "chest_positions": ["N"],
        "regeneration_rate": 10,
        "shareable_energy": False,
        "use_terrain": True,
    },
    "two_agent_pairs_terrain_small": {
        "num_cogs": 2,
        "assembler_positions": ["Any", "Any"],
        "num_chargers": 5,
        "charger_positions": ["Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any"],
        "num_chests": 5,
        "chest_positions": ["N", "S"],
        "regeneration_rate": 10,
        "shareable_energy": True,
        "use_terrain": True,
    },
    "three_agent_triplets_terrain_small": {
        "num_cogs": 3,
        "assembler_positions": ["Any", "Any", "Any"],
        "num_chargers": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "num_chests": 5,
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 10,
        "shareable_energy": True,
        "use_terrain": True,
    },
    "many_agent_triplets_terrain_small": {
        "num_cogs": 12,
        "assembler_positions": ["Any", "Any", "Any"],
        "num_chargers": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "num_chests": 5,
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 10,
        "shareable_energy": True,
        "use_terrain": True,
    },
}


class CogsVsClippiesTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["CogsVsClippiesTaskGenerator"]):
        num_cogs: list[int] = Field(default=[1])
        num_assemblers: list[int] = Field(default=[1])
        assembler_positions: list[list[Position]] = Field(default=[["Any"]])
        num_chargers: list[int] = Field(default=[1])
        charger_positions: list[list[Position]] = Field(default=[["Any"]])
        num_carbon_extractors: list[int] = Field(default=[0])
        carbon_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        num_oxygen_extractors: list[int] = Field(default=[0])
        oxygen_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        num_germanium_extractors: list[int] = Field(default=[0])
        germanium_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        num_silicon_extractors: list[int] = Field(default=[0])
        silicon_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        num_chests: list[int] = Field(default=[1])
        chest_positions: list[list[Position]] = Field(default=[["N"]])
        regeneration_rate: list[int] = Field(default=[5])
        shareable_energy: list[bool] = Field(default=[True])
        use_terrain: list[bool] = Field(default=[False])
        sizes: list[str] = Field(default=["small"])

    def __init__(self, config: "TaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _set_width_and_height(self, num_cogs, num_objects, rng):
        """Set the width and height of the environment to be at least the minimum area required for the number of agents, altars, and generators."""
        minimum_area = num_cogs + num_objects * rng.choice([3, 5, 7])
        width, height = minimum_area // 2, minimum_area // 2
        return width, height

    def _make_env_cfg(self, rng: random.Random):
        num_cogs = rng.choice(self.config.num_cogs)
        num_chargers = rng.choice(self.config.num_chargers)
        num_chests = rng.choice(self.config.num_chests)
        regeneration_rate = rng.choice(self.config.regeneration_rate)
        num_assemblers = num_chests * 3
        max_num_extractors = num_assemblers // 2
        include_extractors = rng.choice([True, False])
        chest_position = rng.choice(self.config.chest_positions)
        charger_position = rng.choice(self.config.charger_positions)
        assembler_position = rng.choice(self.config.assembler_positions)
        carbon_extractor_position = rng.choice(self.config.carbon_extractor_positions)
        oxygen_extractor_position = rng.choice(self.config.oxygen_extractor_positions)
        germanium_extractor_position = rng.choice(
            self.config.germanium_extractor_positions
        )
        silicon_extractor_position = rng.choice(self.config.silicon_extractor_positions)
        if include_extractors:
            num_carbon_extractors = rng.choice(
                [0] + list(range(1, max_num_extractors + 1, 2))
            )
            num_oxygen_extractors = rng.choice(
                [0] + list(range(max_num_extractors + 1, 2))
            )
            num_germanium_extractors = rng.choice(
                [0] + list(range(max_num_extractors + 1, 2))
            )
            num_silicon_extractors = rng.choice(
                [0] + list(range(max_num_extractors + 1, 2))
            )
        else:
            num_carbon_extractors = num_oxygen_extractors = num_germanium_extractors = (
                num_silicon_extractors
            ) = 0

        width, height = self._set_width_and_height(
            num_cogs,
            num_assemblers
            + num_chargers
            + num_carbon_extractors
            + num_oxygen_extractors
            + num_germanium_extractors
            + num_silicon_extractors
            + num_chests,
            rng,
        )
        shareable_energy = rng.choice(self.config.shareable_energy)

        num_instances = 24 // num_cogs

        env = make_game(
            num_cogs=num_cogs,
            width=width,
            height=height,
            num_assemblers=num_assemblers,
            num_chargers=num_chargers,
            num_carbon_extractors=num_carbon_extractors,
            num_oxygen_extractors=num_oxygen_extractors,
            num_germanium_extractors=num_germanium_extractors,
            num_silicon_extractors=num_silicon_extractors,
            num_chests=num_chests,
        )
        self._overwrite_positions(env.game.objects["assembler"], assembler_position)
        self._overwrite_positions(env.game.objects["charger"], charger_position)
        self._overwrite_positions(
            env.game.objects["carbon_extractor"], carbon_extractor_position
        )
        self._overwrite_positions(
            env.game.objects["oxygen_extractor"], oxygen_extractor_position
        )
        self._overwrite_positions(
            env.game.objects["germanium_extractor"], germanium_extractor_position
        )
        self._overwrite_positions(
            env.game.objects["silicon_extractor"], silicon_extractor_position
        )

        env.game.objects["chest"] = ChestConfig(
            type_id=17,
            resource_type="heart",
            deposit_positions=chest_position,
        )

        env.game.inventory_regen_interval = regeneration_rate
        env.game.inventory_regen_amounts = {"energy": 2}
        if shareable_energy:
            env.game.agent.shareable_resources = ["energy"]
        env.label = f"{env.game.num_agents}_cogs_{num_assemblers}_assemblers_{num_chargers}_chargers_{num_carbon_extractors + num_oxygen_extractors + num_germanium_extractors + num_silicon_extractors}_extractors_{num_chests}_chests_{env.game.inventory_regen_interval}_regeneration_rate"

        if self.config.use_terrain:
            terrain_density = rng.choice(["sparse", "balanced", "dense"])
            size = rng.choice(self.config.sizes)
            map_builder = MapGen.Config(
                instances=num_instances,
                border_width=6,
                instance_border_width=3,
                instance=CogsVClippiesFromNumpy.Config(
                    agents=num_cogs,
                    objects={
                        "assembler": num_assemblers,
                        "charger": num_chargers,
                        "carbon_extractor": num_carbon_extractors,
                        "oxygen_extractor": num_oxygen_extractors,
                        "germanium_extractor": num_germanium_extractors,
                        "silicon_extractor": num_silicon_extractors,
                        "chest": num_chests,
                    },
                    remove_altars=True,
                    dir=f"varied_terrain/{terrain_density}_{size}",
                    rng=rng,
                ),
            )
            env.game.map_builder = map_builder
        else:
            env.game.map_builder = MapGen.Config(
                instances=num_instances,
                instance_map=env.game.map_builder,
                num_agents=24,
            )
        env.game.num_agents = 24
        return env

    def _overwrite_positions(self, object, positions):
        for i, recipe in enumerate(object.recipes):
            object.recipes[i] = (positions, recipe[1])

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        env = self._make_env_cfg(rng)

        return env


def make_mettagrid(task_generator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "test") -> PlayTool:
    task_generator = CogsVsClippiesTaskGenerator(
        config=CogsVsClippiesTaskGenerator.Config(**curriculum_args[curriculum_style])
    )
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="cogs_vs_clippies", name="play"
        )
    )


def train(
    curriculum_style: str = "multi_agent_pairs", architecture="vit_reset"
) -> TrainTool:
    task_generator_cfg = CogsVsClippiesTaskGenerator.Config(
        **curriculum_args[curriculum_style]
    )
    algorithm_config = LearningProgressConfig(
        ema_timescale=0.001,
        exploration_bonus=0.15,
        max_memory_tasks=1000,
        max_slice_axes=3,
        progress_smoothing=0.15,
        num_active_tasks=1000,
        rand_task_rate=0.25,
    )
    trainer_cfg = TrainerConfig(
        losses=LossConfig(),
    )
    if architecture == "vit_reset":
        policy_config = ViTResetConfig()
    elif architecture == "lstm_reset":
        policy_config = FastLSTMResetConfig()
    elif architecture == "transformer":
        policy_config = ViTSlidingTransConfig()
        trainer_cfg.batch_size = 131072
        trainer_cfg.minibatch_size = 4096
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    curriculum = CurriculumConfig(
        task_generator=task_generator_cfg,
        algorithm_config=algorithm_config,
    )

    return TrainTool(
        trainer=trainer_cfg,
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(
            simulations=make_eval_suite(),
            evaluate_remote=True,
            evaluate_local=False,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_env(
    num_cogs=1,
    num_assemblers=1,
    num_chargers=1,
    num_carbon_extractors=1,
    num_oxygen_extractors=1,
    num_germanium_extractors=1,
    num_silicon_extractors=1,
    num_chests=1,
    chest_positions=["N"],
    assembler_positions=["Any"],
    charger_positions=["Any"],
    carbon_extractor_positions=["Any"],
    oxygen_extractor_positions=["Any"],
    germanium_extractor_positions=["Any"],
    silicon_extractor_positions=["Any"],
    regeneration_rate=10,
    shareable_energy=False,
    use_terrain=False,
):
    task_generator = CogsVsClippiesTaskGenerator(
        config=CogsVsClippiesTaskGenerator.Config(
            num_cogs=[num_cogs],
            num_assemblers=[num_assemblers],
            num_chargers=[num_chargers],
            num_carbon_extractors=[num_carbon_extractors],
            num_oxygen_extractors=[num_oxygen_extractors],
            num_germanium_extractors=[num_germanium_extractors],
            num_silicon_extractors=[num_silicon_extractors],
            num_chests=[num_chests],
            chest_positions=[chest_positions],
            assembler_positions=[assembler_positions],
            charger_positions=[charger_positions],
            carbon_extractor_positions=[carbon_extractor_positions],
            oxygen_extractor_positions=[oxygen_extractor_positions],
            germanium_extractor_positions=[germanium_extractor_positions],
            silicon_extractor_positions=[silicon_extractor_positions],
            regeneration_rate=[regeneration_rate],
            shareable_energy=[shareable_energy],
            use_terrain=[use_terrain],
        )
    )
    return task_generator.get_task(random.randint(0, 1000000))


def make_eval_suite():
    return [
        SimulationConfig(
            env=make_env(**evals[curriculum_style]),
            suite="cogs_vs_clippies",
            name=f"eval_{curriculum_style}",
        )
        for curriculum_style in evals
    ]


def experiment():
    for curriculum_style in curriculum_args:
        for architecture in ["vit_reset", "lstm_reset", "transformer"]:
            subprocess.run(
                [
                    "./devops/skypilot/launch.py",
                    "experiments.recipes.cogs_v_clips.level_1.train",
                    f"run=daphne.cogs_v_clips.level_1.{curriculum_style}_{architecture}.{time.strftime('%Y-%m-%d')}",
                    f"curriculum_style={curriculum_style}",
                    f"architecture={architecture}",
                    "--gpus=4",
                    "--heartbeat-timeout=3600",
                ]
            )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
