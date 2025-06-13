from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from mettagrid.curriculum.util import config_from_path, curriculum_from_config

from .low_reward import LowRewardCurriculum

logger = logging.getLogger(__name__)


class LazyDict(dict):
    def __init__(self, *args, **kwargs):
        self.lamba_dict = dict(*args, **kwargs)
        self.resolved_dict = {}

    def __getitem__(self, key):
        if key not in self.resolved_dict:
            self.resolved_dict[key] = self.lamba_dict[key]()
        return self.resolved_dict[key]

    def __setitem__(self, key, value):
        raise NotImplementedError("LazyDict is immutable")

    def __delitem__(self, key):
        raise NotImplementedError("LazyDict is immutable")

    def __contains__(self, key):
        return key in self.lamba_dict

    def __len__(self):
        return len(self.lamba_dict)

    def __iter__(self):
        return iter(self.lamba_dict)

    def keys(self):
        return self.lamba_dict.keys()

    def values(self):
        raise NotImplementedError("values() not implemented for LazyDict")

    def items(self):
        raise NotImplementedError("items() not implemented for LazyDict")

    def get(self, key, default=None):
        raise NotImplementedError("get() not implemented for LazyDict")

    def clear(self):
        raise NotImplementedError("LazyDict is immutable")

    def copy(self):
        raise NotImplementedError("copy() not implemented for LazyDict")

    def update(self, other=None, **kwargs):
        raise NotImplementedError("LazyDict is immutable")

    def pop(self, key, default=None):
        raise NotImplementedError("LazyDict is immutable")

    def popitem(self):
        raise NotImplementedError("LazyDict is immutable")

    def setdefault(self, key, default=None):
        raise NotImplementedError("setdefault() not implemented for LazyDict")


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
        default_bins: int = 5,
        alpha: float = 0.01,
    ):
        buckets_unpacked: Dict[str, List[Any]] = {
            path: _buckets_from_spec(spec, default_bins) for path, spec in buckets.items()
        }

        # here, tasks map directly to env configs
        # do we want our id to be more descriptive than an enumerated integer?
        env_cfgs_items = {}

        base = config_from_path(env_cfg_template, env_overrides)
        cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))

        for task_id, parameter_values in enumerate(product(*buckets_unpacked.values())):
            print(f"task_id: {task_id}")
            print(f"parameter_values: {parameter_values}")

            def make_cfg():
                cfg_copy = cfg.copy()
                override = dict(zip(buckets_unpacked.keys(), parameter_values, strict=False))
                for k, v in override.items():
                    OmegaConf.update(cfg_copy, k, v, merge=False)
                env_cfg = OmegaConf.create(OmegaConf.to_container(cfg_copy, resolve=True))
                return curriculum_from_config(env_cfg, env_overrides)

            env_cfgs_items[task_id] = make_cfg

        tasks = LazyDict(env_cfgs_items)

        # call parent WITH the full id→weight mapping
        super().__init__(tasks=tasks, env_overrides=env_overrides, alpha=alpha)

    def set_curricula(self, tasks, env_overrides):
        # here we use the configs directly rather than the paths
        self._curriculums = tasks
        self._task_weights = {t: 1.0 for t in tasks}  # uniform task weights


def _buckets_from_spec(spec: Dict[str, Any], default_bins: int) -> List[Any]:
    """Return a list of bucket values from a `values:` list *or* a {range:,bins:} spec."""
    if "values" in spec:
        return list(spec["values"])

    lo, hi = spec["range"]
    n = int(spec.get("bins", default_bins))

    # Create n ranges from [lo,hi]
    step = (hi - lo) / n
    binned_ranges = [{"range": (lo + i * step, lo + (i + 1) * step)} for i in range(n)]
    return binned_ranges
