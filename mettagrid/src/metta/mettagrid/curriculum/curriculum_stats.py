from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from metta.mettagrid.curriculum.curriculum import Curriculum


class CurriculumStats:
    def __init__(self, curriculum: "Curriculum", num_tasks: int):
        self._curriculum = curriculum
        self._completed_tasks = np.zeros(num_tasks, dtype=np.int32)
        self._sampled_tasks = np.zeros(num_tasks, dtype=np.int32)
        self._total_completed_tasks = 0
        self._total_sampled_tasks = 0
        self._num_tasks = num_tasks

    def record_sample(self, task_idx: int):
        self._sampled_tasks[task_idx] += 1
        self._total_sampled_tasks += 1

    def record_completion(self, task_idx: int):
        self._completed_tasks[task_idx] += 1
        self._total_completed_tasks += 1

    def as_dict(self) -> dict[str, float]:
        result = {}
        result["completed_tasks"] = self._total_completed_tasks
        result["sampled_tasks"] = self._total_sampled_tasks
        result["completion_rate"] = self._total_completed_tasks / self._total_sampled_tasks

        # Flatten all stats into a single dict
        stats_sources = [
            ("probabilities_marginal", self.get_task_probabilities(relative_to_root=False)),
            ("probabilities_absolute", self.get_task_probabilities(relative_to_root=True)),
            ("completion_rates", self.get_total_completions()),
            ("sample_rates", self.get_sample_rates()),
            ("sample_counts", self.get_sample_counts()),
            ("algorithm_stats", self.get_algorithm_stats()),
        ]
        for prefix, d in stats_sources:
            for k, v in d.items():
                result[f"{prefix}/{k}"] = v
        return result

    def _collect_task_stats(self, value_fn, recurse_fn=None, **kwargs) -> dict:
        result = dict()
        tasks = self._curriculum.tasks()
        for task_idx in range(self._num_tasks):
            task = tasks[task_idx]
            result[task.full_name()] = value_fn(task_idx, task, **kwargs)
            if not task.is_leaf() and recurse_fn is not None:
                result.update(recurse_fn(task.stats(), **kwargs))
        return result

    def get_sample_rates(self) -> dict[str, float]:
        return self._collect_task_stats(
            lambda idx, task: float(self._sampled_tasks[idx]) / float(self._total_sampled_tasks)
            if self._total_sampled_tasks > 0
            else 0.0,
            recurse_fn=lambda stats, **_: stats.get_sample_rates(),
        )

    def get_sample_counts(self) -> dict[str, int]:
        return self._collect_task_stats(
            lambda idx, task: int(self._sampled_tasks[idx]),
            recurse_fn=lambda stats, **_: stats.get_sample_counts(),
        )

    def get_total_completions(self) -> dict[str, float]:
        return self._collect_task_stats(
            lambda idx, task: self._completed_tasks[idx],
            recurse_fn=lambda stats, **_: stats.get_total_completions(),
        )

    def get_task_probabilities(self, relative_to_root: bool = False) -> dict[str, float]:
        return self._collect_task_stats(
            lambda idx, task, relative_to_root=False: self._curriculum.algorithm().probabilities[idx],
            recurse_fn=lambda stats, **kwargs: stats.get_task_probabilities(**kwargs),
            relative_to_root=relative_to_root,
        )

    def get_algorithm_stats(self) -> dict[str, float]:
        # TODO: make recursive and include stats from children
        return self._curriculum.algorithm().stats()
