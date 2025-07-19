import logging
from typing import Dict

from metta.mettagrid.curriculum.core import Curriculum

logger = logging.getLogger(__name__)


class MultiTaskCurriculum(Curriculum):
    """Base class for curricula with multiple tasks."""

    def __init__(self, curricula: Dict[str, Curriculum], completion_moving_avg_window: int = 500):
        super().__init__()
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
        self._completion_moving_avg_window = completion_moving_avg_window
        self._completed_tasks = []

    def complete_task(self, id: str, score: float):
        if len(self._completed_tasks) > self._completion_moving_avg_window:
            self._completed_tasks.pop(0)
        self._completed_tasks.append(id)
        super().complete_task(id, score)

    def stats(self) -> dict:
        stats = super().stats()
        stats.update({f"task_completions/{task_id}": 0.0 for task_id in self._curricula})
        completed_tasks = self._completed_tasks
        num_completed_tasks = len(completed_tasks)
        if num_completed_tasks != 0:
            for task in completed_tasks:
                stats[f"task_completions/{task}"] += 1
            stats = {k: v / num_completed_tasks for k, v in stats.items()}

        # Sometimes tasks start with '/' and we don't want task_probs//task/...
        stats = {k.replace("//", "/"): v for k, v in stats.items()}

        for curriculum in self._curricula.values():
            stats.update(curriculum.stats())

        return stats
