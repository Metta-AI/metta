from experiments.recipes.in_context_learning.in_context_learning import (
    TaskGenerator,
    TaskGeneratorConfig,
)
import subprocess
import time
from experiments.recipes.in_context_learning.in_context_learning import (
    RESOURCE_TYPES,
    CONVERTER_TYPES,
    ASSEMBLER_TYPES,
)
from experiments.recipes.in_context_learning.assemblers.foraging import (
    ForagingTaskGenerator,
)
from experiments.recipes.in_context_learning.assemblers.assembly_lines import (
    AssemblyLinesTaskGenerator,
)
import random
from typing import Optional
from experiments.recipes.in_context_learning.assemblers import foraging
from experiments.recipes.in_context_learning.assemblers import assembly_lines
from experiments.recipes.in_context_learning.in_context_learning import (
    MettaGridConfig,
    num_agents_to_positions,
    play_icl,
    train_icl,
)
from metta.tools.train import TrainTool
from experiments.evals.in_context_learning.assemblers.all_assemblers import (
    make_assembler_eval_suite,
)
from metta.tools.play import PlayTool

foraging_curriculum_args = {
    "num_agents": [1, 4, 8, 12, 24],
    "num_altars": list(range(5, 20, 5)),
    "num_generators": [0, 1, 4],
    "room_sizes": ["small", "medium", "large"],
    "positions": num_agents_to_positions[1]
    + num_agents_to_positions[2]
    + num_agents_to_positions[3],
    "max_recipe_inputs": [1, 2, 3],
    "num_chests": [0],
}

assembly_lines_curriculum_args = {
    "num_agents": [1, 2, 4],
    "chain_lengths": [2, 3, 4, 5],
    "num_sinks": [0, 1, 2],
    "room_sizes": ["small", "medium", "large"],
    "positions": [["Any"], ["Any", "Any"], ["Any", "Any", "Any"]],
    "num_chests": [0],
}


class AssemblerTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["AssemblerTaskGenerator"]):
        pass

    def __init__(self, config: "TaskGeneratorConfig"):
        super().__init__(config)
        self.config = config
        self.resource_types = RESOURCE_TYPES.copy()
        self.converter_types = CONVERTER_TYPES.copy()
        self.assembler_types = ASSEMBLER_TYPES.copy()

        self.task_generators = {
            "foraging": ForagingTaskGenerator(
                foraging.make_task_generator_cfg(**foraging_curriculum_args)
            ),
            "assembly_lines": AssemblyLinesTaskGenerator(
                assembly_lines.make_task_generator_cfg(**assembly_lines_curriculum_args)
            ),
        }

    def _generate_task(
        self, task_id: int, rng: random.Random, num_instances: Optional[int] = None
    ) -> MettaGridConfig:
        # choose uniformly
        task_generator = self.task_generators[
            rng.choice(list(self.task_generators.keys()))
        ]
        return task_generator.generate_task(task_id, rng, num_instances)


def play() -> PlayTool:
    task_generator = AssemblerTaskGenerator(TaskGeneratorConfig())
    return play_icl(task_generator)


def train() -> TrainTool:
    task_generator_cfg = AssemblerTaskGenerator.Config()

    return train_icl(task_generator_cfg, make_assembler_eval_suite)


def experiment():
    subprocess.run(
        [
            "./devops/skypilot/launch.py",
            "experiments.recipes.in_context_learning.assemblers.all_assemblers.train",
            f"run=in_context.all_assemblers.{time.strftime('%Y-%m-%d')}",
        ]
    )
