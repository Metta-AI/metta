from experiments.recipes.cogs_v_clips import facilities, generalized_terrain, foraging
from experiments.recipes.cogs_v_clips.config import (
    generalized_terrain_curriculum_args,
    foraging_curriculum_args,
)
import random
from typing import Optional
from metta.agent.policies.vit_reset import ViTResetConfig
from mettagrid.config.mettagrid_config import MettaGridConfig
from metta.tools.train import TrainTool
from metta.rl.trainer_config import TrainerConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.rl.loss import LossConfig
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from experiments.evals.cogs_v_clips import make_cogs_v_clips_evals
from metta.cogworks.curriculum.task_generator import TaskGenerator, TaskGeneratorConfig


class CogsVClipsTaskGenerator(TaskGenerator):
    class Config(TaskGeneratorConfig["CogsVClipsTaskGenerator"]):
        pass

    def __init__(
        self,
        config: "TaskGeneratorConfig",
        terrain_curriculum="multi_agent_triplets",
        foraging_curriculum="all",
    ):
        super().__init__(config)

        self.task_generators = {
            "generalized_terrain": generalized_terrain.GeneralizedTerrainTaskGenerator(
                **generalized_terrain_curriculum_args[terrain_curriculum]
            ),
            "facilities": facilities.make_task_generator().create(),
            "foraging": foraging.ForagingTaskGenerator(
                **foraging_curriculum_args[foraging_curriculum]
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


def train() -> TrainTool:
    task_generator_cfg = CogsVClipsTaskGenerator.Config()
    algorithm_config = LearningProgressConfig(
        num_active_tasks=1000,
    )
    policy_config = ViTResetConfig()

    return TrainTool(
        trainer=TrainerConfig(losses=LossConfig()),
        training_env=TrainingEnvironmentConfig(
            curriculum=CurriculumConfig(
                task_generator=task_generator_cfg, algorithm_config=algorithm_config
            )
        ),
        policy_architecture=policy_config,
        evaluator=EvaluatorConfig(
            simulations=make_cogs_v_clips_evals(),
        ),
        stats_server_uri="https://api.observatory.softmax-research.net",
    )


def experiment():
    import subprocess
    import time

    subprocess.run(
        [
            "./devops/skypilot/launch.py",
            "experiments.recipes.cogs_v_clips.combined.train",
            f"run=cogs_v_clips.combined.{random.randint(0, 10000)}.{time.strftime('%Y-%m-%d')}",
            "--gpus=4",
            "--heartbeat-timeout=3600",
            "--skip-git-check",
        ]
    )


if __name__ == "__main__":
    experiment()
