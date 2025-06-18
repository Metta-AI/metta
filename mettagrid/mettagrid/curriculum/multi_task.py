import logging
from typing import Dict

from omegaconf import DictConfig

from mettagrid.curriculum.curriculum import Curriculum

logger = logging.getLogger(__name__)


class MultiTaskCurriculum(Curriculum):
    """Base class for curricula with multiple tasks."""

    def __init__(self, curricula: Dict[str, float], env_overrides: DictConfig):
        self._curriculums = curricula
        num_agents = None
        for task_id, curriculum in self._curriculums.items():
            cfg_num_agents = curriculum.get_task().env_cfg().game.num_agents
            if num_agents is None:
                num_agents = cfg_num_agents
            else:
                assert cfg_num_agents == num_agents, (
                    f"Task {task_id} has num_agents {cfg_num_agents}, expected {num_agents}"
                )
        self.task_completions = {}

    def complete_task(self, id: str, score: float):
        if id not in self.task_completions:
            self.task_completions[id] = 0
        self.task_completions[id] += 1
        super().complete_task(id, score)
