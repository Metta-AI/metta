import logging
from typing import Dict, List

<<<<<<< HEAD:mettagrid/mettagrid/curriculum/multi_task.py
from mettagrid.curriculum.curriculum import Curriculum
=======
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.util import curriculum_from_config_path
>>>>>>> 785b3ad4173cc4604c075c3dfa7b1541f0d07e16:mettagrid/src/metta/mettagrid/curriculum/multi_task.py

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

    def complete_task(self, id: str, score: float):
        if len(self._completed_tasks) > self._completion_moving_avg_window:
            self._completed_tasks.pop(0)
        self._completed_tasks.append(id)
        super().complete_task(id, score)

    def completed_tasks(self) -> List[str]:
        return self._completed_tasks
