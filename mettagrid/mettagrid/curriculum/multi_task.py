import logging
from typing import Dict

from omegaconf import DictConfig

from mettagrid.curriculum.curriculum import Curriculum

logger = logging.getLogger(__name__)


class MultiTaskCurriculum(Curriculum):
    """Base class for curricula with multiple tasks."""

    def __init__(self, curricula: Dict[str, float], env_overrides: DictConfig, moving_avg_window: int = 500):
        self._curricula = curricula
        num_agents = None
        for task_id, curriculum in self._curricula.items():
            cfg_num_agents = curriculum.get_task().env_cfg().game.num_agents
            if num_agents is None:
                num_agents = cfg_num_agents
            else:
                assert cfg_num_agents == num_agents, (
                    f"Task {task_id} has num_agents {cfg_num_agents}, expected {num_agents}"
                )
        self.moving_avg_window = moving_avg_window
        self.completed_tasks = []

    def complete_task(self, id: str, score: float):
        if len(self.completed_tasks) > self.moving_avg_window:
            self.completed_tasks.pop(0)
        self.completed_tasks.append(id)
        super().complete_task(id, score)
