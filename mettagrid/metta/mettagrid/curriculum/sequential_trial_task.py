from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from metta.common.util.config import copy_omegaconf_config
from metta.mettagrid.curriculum.core import Curriculum, Task
from metta.mettagrid.util.hydra import config_from_path


class SequentialTrialTask(Task):
    def __init__(
        self,
        *,
        id: str,
        curriculum: Curriculum,
        env_cfg_template_path: str,
        # We override the environment config for each trial. This is applied after the env_overrides.
        trial_overrides: List[DictConfig],
        # We override the environment config for the task. This is applied before the trial overrides.
        env_overrides: DictConfig | None = None,
        # If zero, we always advance
        advancement_threshold: float = 0.0,
        # If zero, we never regress
        regression_threshold: float = 0.0,
        # We terminate the task when we've completed this many trials, or when we advance beyond the last trial.
        # This only makes sense in the context of a non-zero advancement_threshold, where you might get stuck running
        # the same trial over and over.
        max_trials: int | None = None,
    ):
        """A task that has a single trial, but the trial is repeated multiple times."""
        super().__init__(id, curriculum)
        base_cfg = config_from_path(env_cfg_template_path, env_overrides)
        OmegaConf.merge(base_cfg, env_overrides)
        # Copy to reset the config's root, so other overrides work correctly. This may be unnecessary voodoo.
        self._base_cfg = copy_omegaconf_config(base_cfg)
        self._scores = []
        self._current_trial_idx = 0
        self._max_trials = max_trials
        self._trial_overrides = trial_overrides
        self._advancement_threshold = advancement_threshold
        self._regression_threshold = regression_threshold

    def complete_trial(self, score: float):
        assert not self._is_complete, "Task is already complete"
        self._scores.append(score)
        if score >= self._advancement_threshold:
            self._current_trial_idx += 1
        elif self._current_trial_idx > 0 and score < self._regression_threshold:
            self._current_trial_idx -= 1
        if self._current_trial_idx >= len(self._trial_overrides) or (
            self._max_trials is not None and len(self._scores) >= self._max_trials
        ):
            # It's over!
            for curriculum, id in self._curricula:
                curriculum.complete_task(id, sum(self._scores))
            self._is_complete = True

    def env_cfg(self) -> DictConfig:
        cfg = OmegaConf.merge(self._base_cfg, self._trial_overrides[self._current_trial_idx])
        return hydra.utils.instantiate(cfg)
