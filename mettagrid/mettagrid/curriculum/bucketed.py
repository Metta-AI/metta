from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mettagrid.curriculum.curriculum import Task
from mettagrid.curriculum.util import config_from_path, curriculum_from_config_path

from .low_reward import LowRewardCurriculum  # adjust path if needed

log = logging.getLogger(__name__)


def _build_options(spec: Dict[str, Any], default_bins: int) -> List[Any]:
    """Return a list of bucket values from a `values:` list *or* a {range:,bins:} spec."""
    if "values" in spec:
        return list(spec["values"])

    lo, hi = spec["range"]
    n = int(spec.get("bins", default_bins))

    # equally-spaced, then cast to int if endpoints were ints
    pts = np.linspace(lo, hi, num=n)
    if isinstance(lo, int) and isinstance(hi, int):
        pts = np.round(pts).astype(int)
    return pts.tolist()


class BucketedCurriculum(LowRewardCurriculum):
    """
    One-env curriculum that samples **one** bin per bucket-dimension on every episode
    and re-weights those bins by low reward.
    """

    def __init__(
        self,
        env_cfg_template: str,
        buckets: Dict[str, Dict[str, Any]],
        env_overrides: DictConfig,
        alpha: float = 0.01,
        default_bins: int = 5,
    ):
        # 1️⃣ Flatten bucket specs and build priority arrays
        self._bucket_specs: Dict[str, List[Any]] = {}
        self._prio: Dict[str, np.ndarray] = {}
        self._visits: Dict[str, np.ndarray] = {}
        self._clean2path: Dict[str, str] = {}

        for path, spec in buckets.items():
            opts = _build_options(spec, default_bins)
            self._bucket_specs[path] = opts
            self._prio[path] = np.ones(len(opts), dtype=np.float32)
            self._visits[path] = np.zeros(len(opts), dtype=np.int64)

            # loss-free label for parent_id strings
            clean = path.replace(".", "_").replace("[", "_").replace("]", "").replace('"', "").replace("'", "")
            self._clean2path[clean] = path

        # 2️⃣ single-env priority (kept for symmetry)
        self._env_priority = 1.0
        self._env_visits = 0

        self._template = env_cfg_template
        self._env_overrides = env_overrides
        self._alpha = alpha
        self._eps = 1e-6

        # 3️⃣ Call parent with the real env key so Hydra finds the config
        super().__init__(tasks={env_cfg_template: 1.0}, env_overrides=env_overrides)

        # wipe parent’s fixed-key bookkeeping; we manage our own IDs
        self._reward_averages, self._reward_maxes = {}, {}
        self._task_weights = {}  # keep attribute but make empty

    # --------------------------------------------------------------------- #
    #                              Sampling                                 #
    # --------------------------------------------------------------------- #
    def get_task(self) -> Task:
        override: Dict[str, Any] = {}
        id_parts: List[str] = []

        # sample one bin per bucket-dimension
        for path, opts in self._bucket_specs.items():
            p_arr = self._prio[path]
            idx = int(np.random.choice(len(opts), p=p_arr / p_arr.sum()))
            self._visits[path][idx] += 1
            value = opts[idx]

            clean = next(k for k, v in self._clean2path.items() if v == path)
            id_parts.append(f"{clean}={idx}")
            override[path] = value

        parent_id = "|".join(id_parts)

        # realise env config with overrides
        base_dc = config_from_path(self._template, self._env_overrides)
        raw_dict = OmegaConf.to_container(base_dc, resolve=False)
        cfg = OmegaConf.create(raw_dict)

        for p, v in override.items():
            # cast to template’s original type
            original = OmegaConf.select(cfg, p)
            if isinstance(original, int):
                v = int(round(v))
            OmegaConf.update(cfg, p, v, merge=False)

        final_conf = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        sub = curriculum_from_config_path(self._template, final_conf)

        task = sub.get_task()
        task.add_parent(self, parent_id)
        return task

    # --------------------------------------------------------------------- #
    #                           Priority update                             #
    # --------------------------------------------------------------------- #
    def complete_task(self, parent_id: str, score: float):
        err = abs(-score) + self._eps
        prio = err**self._alpha
        self._env_priority = prio

        # update each chosen bucket’s priority
        for label in parent_id.split("|"):
            clean, idx_str = label.split("=")
            path = self._clean2path[clean]
            self._prio[path][int(idx_str)] = prio

        # rolling stats for this specific combination (for inspection/logging)
        avg = self._reward_averages.get(parent_id, 0.0)
        avg = (1 - self._alpha) * avg + self._alpha * score
        self._reward_averages[parent_id] = avg
        self._reward_maxes[parent_id] = max(self._reward_maxes.get(parent_id, -np.inf), score)

        log.debug(f"[Bucketed] {parent_id}: reward={score:.3f}, avg={avg:.3f}, max={self._reward_maxes[parent_id]:.3f}")
