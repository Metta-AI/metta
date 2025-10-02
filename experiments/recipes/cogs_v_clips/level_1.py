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
from mettagrid.config.mettagrid_config import RecipeConfig
# ADDING TERRAIN

# base has all the assemblers, in the middle, surrounded by walls
# nano-assembler which maeks hearts
# extractors are also in the base -> everything in the base

# second -> in all the cardinal directions there are holes in the walls so you can go out and forage for resources
# outside of the base there are assemblers with better recipes than assemblers in the base


#integrate facility training environments
curriculum_args = {
    "multi_agent_singles": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"]],
        "charger_positions": [["Any"]],
        "carbon_extractor_positions": [["Any"]],
        "oxygen_extractor_positions": [["Any"]],
        "germanium_extractor_positions": [["Any"]],
        "silicon_extractor_positions": [["Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True, False],
    },
    "multi_agent_pairs": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"], ["Any", "Any"]],
        "charger_positions": [["Any"], ["Any", "Any"]],
        "carbon_extractor_positions": [["Any"], ["Any", "Any"]],
        "oxygen_extractor_positions": [["Any"], ["Any", "Any"]],
        "germanium_extractor_positions": [["Any"], ["Any", "Any"]],
        "silicon_extractor_positions": [["Any"], ["Any", "Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N", "S"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True, False],
    },
    "multi_agent_triplets": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "charger_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "carbon_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "oxygen_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "germanium_extractor_positions": [
            ["Any"],
            ["Any", "Any"],
            ["Any", "Any", "Any"],
        ],
        "silicon_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N", "S", "E"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True, False],
    },
    "multi_agent_singles_bases": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"]],
        "charger_positions": [["Any"]],
        "carbon_extractor_positions": [["Any"]],
        "oxygen_extractor_positions": [["Any"]],
        "germanium_extractor_positions": [["Any"]],
        "silicon_extractor_positions": [["Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True],
    },
    "multi_agent_pairs_bases": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"], ["Any", "Any"]],
        "charger_positions": [["Any"], ["Any", "Any"]],
        "carbon_extractor_positions": [["Any"], ["Any", "Any"]],
        "oxygen_extractor_positions": [["Any"], ["Any", "Any"]],
        "germanium_extractor_positions": [["Any"], ["Any", "Any"]],
        "silicon_extractor_positions": [["Any"], ["Any", "Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N", "S"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True],
    },
    "multi_agent_triplets_bases": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "charger_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "carbon_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "oxygen_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "germanium_extractor_positions": [
            ["Any"],
            ["Any", "Any"],
            ["Any", "Any", "Any"],
        ],
        "silicon_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N", "S", "E"]],
        "regeneration_rate": [2, 4, 6],
        "sizes": ["small", "medium"],
        "use_base": [True],
    },
    "multi_agent_singles_uniform": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"]],
        "charger_positions": [["Any"]],
        "carbon_extractor_positions": [["Any"]],
        "oxygen_extractor_positions": [["Any"]],
        "germanium_extractor_positions": [["Any"]],
        "silicon_extractor_positions": [["Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N"]],
        "regeneration_rate": [2,4,6],
        "shareable_energy": [False],
        "use_terrain": [True],
        "sizes": ["small", "medium"],
        "use_base": [False],
    },
    "multi_agent_pairs_uniform": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"], ["Any", "Any"]],
        "charger_positions": [["Any"], ["Any", "Any"]],
        "carbon_extractor_positions": [["Any"], ["Any", "Any"]],
        "oxygen_extractor_positions": [["Any"], ["Any", "Any"]],
        "germanium_extractor_positions": [["Any"], ["Any", "Any"]],
        "silicon_extractor_positions": [["Any"], ["Any", "Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N", "S"]],
        "regeneration_rate": [2,4,6],
        "shareable_energy": [True],
        "use_terrain": [True],
        "sizes": ["small", "medium"],
        "use_base": [False],
    },
    "multi_agent_triplets_uniform": {
        "num_cogs": [2, 4, 8, 12, 24],
        "assembler_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "charger_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "carbon_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "oxygen_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "germanium_extractor_positions": [
            ["Any"],
            ["Any", "Any"],
            ["Any", "Any", "Any"],
        ],
        "silicon_extractor_positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
        "num_obj_distribution": [2, 4, 8, 10, 15],
        "chest_positions": [["N", "S", "E"]],
        "regeneration_rate": [2, 3, 4, 5],
        "shareable_energy": [True],
        "use_terrain": [True],
        "sizes": ["small", "medium"],
        "use_base": [False],
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
    "single_agent_small": {
        "num_cogs": 1,
        "assembler_positions": ["Any"],
        "num_obj_distribution": 5,
        "charger_positions": ["Any"],
        "carbon_extractor_positions": ["Any"],
        "oxygen_extractor_positions": ["Any"],
        "germanium_extractor_positions": ["Any"],
        "silicon_extractor_positions": ["Any"],
        "chest_positions": ["N"],
        "regeneration_rate": 2,
        "shareable_energy": False,
        "use_terrain": True,
        "sizes": "small",
    },
    "two_agent_pairs_small": {
        "num_cogs": 2,
        "assembler_positions": ["Any", "Any"],
        "num_obj_distribution": 2,
        "charger_positions": ["Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any"],
        "chest_positions": ["N", "S"],
        "regeneration_rate": 4,
        "shareable_energy": True,
        "use_terrain": True,
        "sizes": "small",
    },
    "three_agent_triplets_small": {
        "num_cogs": 3,
        "assembler_positions": ["Any", "Any", "Any"],
        "num_obj_distribution": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 2,
        "shareable_energy": True,
        "use_terrain": True,
        "sizes": "small",
    },
    "many_agent_triplets_small": {
        "num_cogs": 12,
        "assembler_positions": ["Any", "Any"],
        "num_obj_distribution": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 2,
        "shareable_energy": True,
        "use_terrain": True,
        "sizes": "small",
    },
    "single_agent_medium": {
        "num_cogs": 1,
        "assembler_positions": ["Any"],
        "num_obj_distribution": 5,
        "charger_positions": ["Any"],
        "carbon_extractor_positions": ["Any"],
        "oxygen_extractor_positions": ["Any"],
        "germanium_extractor_positions": ["Any"],
        "silicon_extractor_positions": ["Any"],
        "chest_positions": ["N"],
        "regeneration_rate": 2,
        "shareable_energy": False,
        "use_terrain": True,
        "sizes": "medium",
    },
    "two_agent_pairs_medium": {
        "num_cogs": 2,
        "assembler_positions": ["Any", "Any"],
        "num_obj_distribution": 2,
        "charger_positions": ["Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any"],
        "chest_positions": ["N", "S"],
        "regeneration_rate": 4,
        "shareable_energy": True,
        "use_terrain": True,
        "sizes": "medium",
    },
    "three_agent_triplets_medium": {
        "num_cogs": 3,
        "assembler_positions": ["Any", "Any", "Any"],
        "num_obj_distribution": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 2,
        "shareable_energy": True,
        "use_terrain": True,
        "sizes": "medium",
    },
    "many_agent_triplets_medium": {
        "num_cogs": 12,
        "assembler_positions": ["Any", "Any"],
        "num_obj_distribution": 5,
        "charger_positions": ["Any", "Any", "Any"],
        "carbon_extractor_positions": ["Any", "Any", "Any"],
        "oxygen_extractor_positions": ["Any", "Any", "Any"],
        "germanium_extractor_positions": ["Any", "Any", "Any"],
        "silicon_extractor_positions": ["Any", "Any", "Any"],
        "chest_positions": ["N", "S", "E"],
        "regeneration_rate": 2,
        "shareable_energy": True,
        "use_terrain": True,
        "sizes": "medium",
    },
    #ADD DAVIDS ENVS
}


class CogsVsClippiesTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["CogsVsClippiesTaskGenerator"]):
        num_cogs: list[int] = Field(default=[1])
        num_obj_distribution: list[int] = Field(default=[10])
        assembler_positions: list[list[Position]] = Field(default=[["Any"]])
        charger_positions: list[list[Position]] = Field(default=[["Any"]])
        carbon_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        oxygen_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        germanium_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        silicon_extractor_positions: list[list[Position]] = Field(default=[["Any"]])
        chest_positions: list[list[Position]] = Field(default=[["N"]])
        regeneration_rate: list[int] = Field(default=[5])
        shareable_energy: list[bool] = Field(default=[True])
        use_terrain: list[bool] = Field(default=[True])
        sizes: list[str] = Field(default=["small"])
        use_base: list[bool] = Field(default=[True])

    def __init__(self, config: "TaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _set_width_and_height(self, num_cogs, num_objects, rng):
        """Set the width and height of the environment to be at least the minimum area required for the number of agents, altars, and generators."""
        minimum_area = num_cogs + num_objects * rng.choice([1, 2, 4])
        width, height = minimum_area // 2, minimum_area // 2
        return width, height

    def _make_env_cfg(self, rng: random.Random):
        num_cogs = rng.choice(self.config.num_cogs)
        regeneration_rate = rng.choice(self.config.regeneration_rate)
        include_extractors = rng.choice([True, False])
        chest_position = rng.choice(self.config.chest_positions)
        charger_position = rng.choice(self.config.charger_positions)
        assembler_position = rng.choice(self.config.assembler_positions)
        carbon_extractor_position = rng.choice(self.config.carbon_extractor_positions)
        oxygen_extractor_position = rng.choice(self.config.oxygen_extractor_positions)
        germanium_extractor_position = rng.choice(
            self.config.germanium_extractor_positions
        )
        use_terrain = rng.choice(self.config.use_terrain)

        if use_terrain:
            terrain_density = rng.choice(["sparse", "balanced", "dense"])
            room_size = rng.choice(self.config.sizes)
            if room_size == "small":
                num_obj_distribution = [2, 4, 6]
            elif room_size == "medium":
                num_obj_distribution = [5, 8, 10]
            elif room_size == "large":
                num_obj_distribution = [10, 20, 25]
        else:
            num_obj_distribution = self.config.num_obj_distribution

        num_assemblers = rng.choice(num_obj_distribution)
        num_chargers = rng.choice(num_obj_distribution)
        num_chests = rng.choice(num_obj_distribution)
        num_chests = max(4, num_chests)

        silicon_extractor_position = rng.choice(self.config.silicon_extractor_positions)
        if include_extractors:
            num_extractors = {
                "carbon": rng.choice([0] + num_obj_distribution),
                "oxygen": rng.choice([0] + num_obj_distribution),
                "germanium": rng.choice([0] + num_obj_distribution),
                "silicon": rng.choice([0] + num_obj_distribution),
            }
        else:
            num_extractors = {"carbon": 0, "oxygen": 0, "germanium": 0, "silicon": 0}

        if not use_terrain:
            width, height = self._set_width_and_height(
                num_cogs,
                num_assemblers
                + num_chargers
                + num_extractors["carbon"]
                + num_extractors["oxygen"]
                + num_extractors["germanium"]
                + num_extractors["silicon"]
                + num_chests,
                rng,
            )
        else:
            width, height = 10, 10  # won't use for terrain

        use_base = rng.choice(self.config.use_base)
        num_instances = 24 // num_cogs

        env = make_game(
            num_cogs=num_cogs,
            width=width,
            height=height,
            num_assemblers=num_assemblers,
            num_chargers=num_chargers,
            num_carbon_extractors=num_extractors["carbon"],
            num_oxygen_extractors=num_extractors["oxygen"],
            num_germanium_extractors=num_extractors["germanium"],
            num_silicon_extractors=num_extractors["silicon"],
            num_chests=num_chests,
        )
        self._make_assembler_recipes(
            env.game.objects["assembler"], rng, num_extractors, assembler_position
        )
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
        env.game.agent.inventory_regen_amounts = {"energy": 1}
        if rng.choice(self.config.shareable_energy):
            env.game.agent.shareable_resources = ["energy"]
        env.label = f"{env.game.num_agents}_cogs_{room_size}_{env.game.inventory_regen_interval}_regeneration_rate_{use_base}_base"

        map_builder = MapGen.Config(
            instances=num_instances,
            border_width=6,
            instance_border_width=3,
            instance=CogsVClippiesFromNumpy.Config(
                agents=num_cogs,
                objects={
                    "assembler": num_assemblers,
                    "charger": num_chargers,
                    "carbon_extractor": num_extractors["carbon"],
                    "oxygen_extractor": num_extractors["oxygen"],
                    "germanium_extractor": num_extractors["germanium"],
                    "silicon_extractor": num_extractors["silicon"],
                    "chest": num_chests,
                },
                remove_altars=True,
                dir=f"varied_terrain/{terrain_density}_{room_size}",
                mass_in_center=use_base,
                rng=rng,
            ),
        )
        env.game.map_builder = map_builder
        # else:
        #     env.game.map_builder = MapGen.Config(
        #         instances=num_instances,
        #         instance_map=env.game.map_builder,
        #         num_agents=24,
        #     )
        env.game.num_agents = 24

        return env

    def _make_assembler_recipes(
        self,
        assembler,
        rng: random.Random,
        num_extractors: dict[str, int],
        assembler_position: list[Position],
    ):
        input_resources = {"energy": 3}
        for resource, num_extractor in num_extractors.items():
            if num_extractor > 0 and rng.choice([True, False]):
                input_resources[resource] = 1
        assembler.recipes = [
            (
                assembler_position,
                RecipeConfig(
                    input_resources=input_resources,
                    output_resources={"heart": 1},
                    cooldown=1,
                ),
            )
        ]
        return assembler

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
    curriculum_style: str = "multi_agent_pairs_uniform", architecture="vit_reset"
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
        trainer_cfg.batch_size = 516096
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
            evaluate_remote=False,
            evaluate_local=True,
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def make_env(
    num_cogs=1,
    num_obj_distribution=4,
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
    use_base=True,
    sizes="small",
):
    task_generator = CogsVsClippiesTaskGenerator(
        config=CogsVsClippiesTaskGenerator.Config(
            num_cogs=[num_cogs],
            chest_positions=[chest_positions],
            assembler_positions=[assembler_positions],
            num_obj_distribution=[num_obj_distribution],
            charger_positions=[charger_positions],
            carbon_extractor_positions=[carbon_extractor_positions],
            oxygen_extractor_positions=[oxygen_extractor_positions],
            germanium_extractor_positions=[germanium_extractor_positions],
            silicon_extractor_positions=[silicon_extractor_positions],
            regeneration_rate=[regeneration_rate],
            shareable_energy=[shareable_energy],
            use_terrain=[use_terrain],
            use_base=[use_base],
            sizes=[sizes],
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
        for architecture in ["vit_reset", "lstm_reset"]:
            subprocess.run(
                [
                    "./devops/skypilot/launch.py",
                    "experiments.recipes.cogs_v_clips.level_1.train",
                    f"run=cogs_v_clips.level_1.eval_local.{curriculum_style}_{architecture}.{time.strftime('%Y-%m-%d')}",
                    f"curriculum_style={curriculum_style}",
                    f"architecture={architecture}",
                    "--gpus=4",
                    "--heartbeat-timeout=3600",
                    "--skip-git-check",
                ]
            )
        time.sleep(1)


if __name__ == "__main__":
    experiment()
