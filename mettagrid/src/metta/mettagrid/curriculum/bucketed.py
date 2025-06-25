from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum.sampling import SampledTaskCurriculum
from metta.mettagrid.curriculum.util import config_from_path

from .low_reward import LowRewardCurriculum

logger = logging.getLogger(__name__)


class BucketedCurriculum(LowRewardCurriculum):
    def __init__(
        self,
        env_cfg_template: str,
        buckets: Dict[str, Dict[str, Any]],
        env_overrides: Optional[DictConfig] = None,
        default_bins: int = 1,
        moving_avg_decay_rate: float = 0.01,
    ):
        expanded_buckets = _expand_buckets(buckets, default_bins)

        self._id_to_curriculum = {}
        base_cfg = config_from_path(env_cfg_template, env_overrides)
        env_cfg_template = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))

        logger.info("Generating bucketed tasks")
        for parameter_values in tqdm(product(*expanded_buckets.values())):
            curriculum_id = get_id(expanded_buckets.keys(), parameter_values)
            self._id_to_curriculum[curriculum_id] = SampledTaskCurriculum(
                curriculum_id, env_cfg_template, expanded_buckets.keys(), parameter_values
            )
        tasks = {t: 1.0 for t in self._id_to_curriculum.keys()}
        super().__init__(tasks=tasks, env_overrides=env_overrides, moving_avg_decay_rate=moving_avg_decay_rate)

    def _curriculum_from_id(self, id: str) -> Curriculum:
        return self._id_to_curriculum[id]


def get_id(parameters, values):
    curriculum_id = ""
    for k, v in zip(parameters, values, strict=False):
        if isinstance(v, dict):
            v = v.get("range", "values")
        curriculum_id += f"{'.'.join(k.split('.')[-3:])}={v};"
    return curriculum_id


def _expand_buckets(buckets: Dict[str, Dict[str, Any]], default_bins: int = 1) -> Tuple[List[str], List[List[Any]]]:
    """
    buckets: specified in the config, values or ranges for each parameter
    returns: unpacked configurations for each parameter given the number of bins
    """
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
    return buckets_unpacked
