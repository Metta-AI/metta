import logging
from typing import Dict

from omegaconf import DictConfig

from .core import Curriculum
from .util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class MultiTaskCurriculum(Curriculum):
    """Base class for curricula with multiple tasks."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig):
        self._curriculums = {t: curriculum_from_config_path(t, env_overrides) for t in tasks.keys()}
        self._task_weights = tasks

        num_agents = None
        for task_id, curriculum in self._curriculums.items():
            cfg_num_agents = curriculum.get_task().env_cfg().game.num_agents
            if num_agents is None:
                num_agents = cfg_num_agents
            else:
                assert cfg_num_agents == num_agents, (
                    f"Task {task_id} has num_agents {cfg_num_agents}, expected {num_agents}"
                )

    def get_task_probs(self) -> dict[str, float]:
        """Return the current task probabilities for logging purposes."""
        total = sum(self._task_weights.values())
        if total == 0:
            # Avoid division by zero, assign uniform probability
            n = len(self._task_weights)
            return {k: 1.0 / n for k in self._task_weights}
        return {k: v / total for k, v in self._task_weights.items()}
