from cogames.cogs_vs_clips.missions import Machina1OpenWorldMission, get_map
from metta.tools.play import PlayTool
from metta.sim.simulation_config import SimulationConfig
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig
import random
from mettagrid.config.mettagrid_config import MettaGridConfig

def load_machina_1():
    machina1 = Machina1OpenWorldMission()
    map_builder = get_map("machina_100_stations.map")
    return machina1.instantiate(map_builder, 2).make_env()


def play():
    env = load_machina_1()
    return PlayTool(
        sim=SimulationConfig(
            env=env, suite="machina1", name="play"
        )
    )

#parameters
# extractors
# use_charger
# assembler_inputs
# use_chest
#use_glyphs

class Machina1BaseTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["Machina1BaseTaskGenerator"]):
        extractors: list[str]
        use_charger: list[bool]
        assembler_inputs: dict[str, int]
        use_chest: list[bool]
        use_glyphs: list[bool]

    def __init__(self, config: "Machina1BaseTaskGenerator.Config"):
        super().__init__(config)
        self.config = config

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        pass
