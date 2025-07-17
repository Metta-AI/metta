import logging
from typing import Dict, List

from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum

logger = logging.getLogger(__name__)


class MultiTaskCurriculum(Curriculum):
    """Base class for curricula with multiple tasks."""

    def __init__(self, curricula: Dict[str, Curriculum], completion_moving_avg_window: int = 500):
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
        for task_id in self._curricula:
            logger.info(f"Curriculum Task: {task_id}")

    def complete_task(self, id: str, score: float):
        if len(self._completed_tasks) > self._completion_moving_avg_window:
            self._completed_tasks.pop(0)
        self._completed_tasks.append(id)
        super().complete_task(id, score)

    def completed_tasks(self) -> List[str]:
        return self._completed_tasks

    def get_completion_rates(self):
        completion_rates = {f"task_completions/{task_id}": 0.0 for task_id in self._curricula}
        completed_tasks = self.completed_tasks()
        num_completed_tasks = len(completed_tasks)
        if num_completed_tasks != 0:
            for task in completed_tasks:
                completion_rates[f"task_completions/{task}"] += 1
            completion_rates = {k: v / num_completed_tasks for k, v in completion_rates.items()}
        return completion_rates

    def get_task_probs(self) -> dict[str, float]:
        """Return the current task probabilities for logging purposes."""
        total = sum(self._task_weights.values())
        if total == 0:
            # Avoid division by zero, assign uniform probability
            n = len(self._task_weights)
            return {k: 1.0 / n for k in self._task_weights}
        return {k: v / total for k, v in self._task_weights.items()}

    def get_env_cfg_by_bucket(self) -> dict[str, DictConfig]:
        configs = {}
        for task_id, curriculum in self._curricula.items():
            # Get configs from child curriculums without creating tasks
            child_configs = curriculum.get_env_cfg_by_bucket()
            # Use task_id as key if child returns a single config
            if len(child_configs) == 1:
                configs[task_id] = list(child_configs.values())[0]
            else:
                # Prefix multiple configs with task_id
                for sub_id, cfg in child_configs.items():
                    configs[f"{task_id}/{sub_id}"] = cfg
        return configs
