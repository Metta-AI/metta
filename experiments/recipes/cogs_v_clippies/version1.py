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
from mettagrid.config.mettagrid_config import MettaGridConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
import random
from mettagrid.config.mettagrid_config import Field
from mettagrid.config.mettagrid_config import Position
from cogames.cogs_vs_clips.scenarios import make_game
from metta.tools.play import PlayTool
from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import ChestConfig

curriculum_args = {
    "single_agent": {
        "num_cogs": [1],
        "assembler_positions": [["Any"]],
        "num_chargers": [3, 5, 10, 15],
        "charger_positions": [["Any"]],
        "carbon_extractor_positions": [["Any"]],
        "oxygen_extractor_positions": [["Any"]],
        "germanium_extractor_positions": [["Any"]],
        "silicon_extractor_positions": [["Any"]],
        "num_chests": [1, 5, 10],
        "chest_positions": [["N"]],
        "regeneration_rate": [5, 10, 15],
        "shareable_energy": [True, False],
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
        "regeneration_rate": [5, 10, 15],
        "shareable_energy": [True, False],
    },
    "multi_agent_triplets": {
        "num_cogs": [3, 6, 9, 12, 21],
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
        "regeneration_rate": [5, 10, 15],
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
    #     }
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
                [0] + list(range(max_num_extractors + 1, 2))
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

        print(
            f"Building env with {num_cogs} agents, {num_assemblers} assemblers, {num_chargers} chargers, {num_carbon_extractors} carbon extractors, {num_oxygen_extractors} oxygen extractors, {num_germanium_extractors} germanium extractors, {num_silicon_extractors} silicon extractors, {num_chests} chests, {regeneration_rate} regeneration rate"
        )

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
        env.game.inventory_regen_amounts = {"energy": 10}
        if shareable_energy:
            env.game.agent.shareable_resources = ["energy"]
        env.label = f"{env.game.num_agents}_cogs_{num_assemblers}_assemblers_{num_chargers}_chargers_{num_carbon_extractors + num_oxygen_extractors + num_germanium_extractors + num_silicon_extractors}_extractors_{num_chests}_chests_{env.game.inventory_regen_interval}_regeneration_rate"
        return env

    def _overwrite_positions(self, object, positions):
        for recipe in object.recipes:
            recipe = (positions, recipe[1])

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        env = self._make_env_cfg(rng)

        return env


def make_mettagrid(task_generator) -> MettaGridConfig:
    return task_generator.get_task(random.randint(0, 1000000))


def play(curriculum_style: str = "multi_agent_pairs") -> PlayTool:
    task_generator = CogsVsClippiesTaskGenerator(
        config=CogsVsClippiesTaskGenerator.Config(**curriculum_args[curriculum_style])
    )
    return PlayTool(
        sim=SimulationConfig(
            env=make_mettagrid(task_generator), suite="cogs_vs_clippies", name="play"
        )
    )
