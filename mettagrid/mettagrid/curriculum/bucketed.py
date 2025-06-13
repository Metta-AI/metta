from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mettagrid.curriculum.sampling import SampledTaskCurriculum
from mettagrid.curriculum.util import config_from_path

from .low_reward import LowRewardCurriculum

logger = logging.getLogger(__name__)


class BucketedCurriculum(LowRewardCurriculum):
    """
    Build a dedicated sub-curriculum for **every** bucket combination once,
    then let LowRewardCurriculum handle sampling / reweighting.
    """

    def __init__(
        self,
        env_cfg_template: str,
        buckets: Dict[str, Dict[str, Any]],
        env_overrides: DictConfig,
        *,
        default_bins: int = 1,
        alpha: float = 0.01,
    ):
        bucket_parameters, bucket_values = _expand_buckets(buckets, default_bins)

        # here, tasks map directly to curricula
        tasks = {}
        base = config_from_path(env_cfg_template, env_overrides)
        env_cfg_template = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
        logger.info("Generating bucketed tasks")
        for task_id, parameter_values in tqdm(enumerate(product(*bucket_values))):
            curriculum = SampledTaskCurriculum(task_id, env_cfg_template, bucket_parameters, parameter_values)
            tasks[task_id] = curriculum
        super().__init__(tasks=tasks, env_overrides=env_overrides, alpha=alpha)

    def set_curricula(self, tasks, env_overrides=None):
        self._curriculums = tasks
        self._task_weights = {t: 1.0 for t in tasks}  # uniform task weights


def _expand_buckets(buckets: Dict[str, Dict[str, Any]], default_bins: int = 1) -> Tuple[List[str], List[List[Any]]]:
    buckets_unpacked = {}
    for parameter, bucket_spec in buckets.items():
        if "values" in bucket_spec:
            buckets_unpacked[parameter] = bucket_spec["values"]
        elif "range" in bucket_spec:
            lo, hi = bucket_spec["range"]
            n = int(bucket_spec.get("bins", default_bins))
            step = (hi - lo) / n
            want_int = isinstance(lo, int) and isinstance(hi, int)

            binned_ranges = []
            for i in range(n):
                lo_i, hi_i = lo + i * step, lo + (i + 1) * step
                binned_ranges.append({"range": (lo_i, hi_i), "want_int": want_int})

            buckets_unpacked[parameter] = binned_ranges
        else:
            raise ValueError(f"Invalid bucket spec: {bucket_spec}")
    return list(buckets_unpacked.keys()), list(buckets_unpacked.values())
