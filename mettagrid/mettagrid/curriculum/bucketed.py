from __future__ import annotations

import logging
from itertools import product
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from mettagrid.curriculum.util import config_from_path, curriculum_from_config_path

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
        default_bins: int = 5,
        alpha: float = 0.01,
    ):
        opts: Dict[str, List[Any]] = {path: _buckets_from_spec(spec, default_bins) for path, spec in buckets.items()}

        bucket_tasks: Dict[str, float] = {}  # id → initial weight (1.0)
        self._environments: Dict[str, DictConfig] = {}  # id → resolved DictConfig

        for combo in product(*opts.values()):
            override = dict(zip(opts.keys(), combo, strict=False))

            # build a unique stable id for the environments key
            parts = [f"{k.replace('.', '_')}={i}" for k, i in zip(opts, combo, strict=False)]
            bucket_id = "|".join(parts)

            # create resolved config for this bucket
            base = config_from_path(env_cfg_template, env_overrides)
            cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
            for k, v in override.items():
                OmegaConf.update(cfg, k, v, merge=False)
            env_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

            bucket_tasks[bucket_id] = 1.0  # uniform initial weight
            self._environments[bucket_id] = env_cfg

        # 3️⃣ call parent WITH the full id→weight mapping
        super().__init__(tasks=bucket_tasks, env_overrides=DictConfig({}), alpha=alpha)

        # 4️⃣ replace each entry in self._curriculums with a concrete curriculum
        for bid, cfg in self._environments.items():
            self._curriculums[bid] = curriculum_from_config_path(env_cfg_template, cfg)


def _buckets_from_spec(spec: Dict[str, Any], default_bins: int) -> List[Any]:
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


# class BucketedCurriculum(LowRewardCurriculum):
#     """
#     One-env curriculum that samples **one** bin per bucket-dimension on every episode
#     and re-weights those bins by low reward.
#     """

#     def __init__(
#         self,
#         env_cfg_template: str,
#         buckets: Dict[str, Dict[str, Any]],
#         env_overrides: DictConfig,
#         alpha: float = 0.01,
#         default_bins: int = 1,
#     ):
#         """
#         Args:
#             buckets: A dictionary of bucket specifications.
#             env_overrides: A dictionary of environment overrides.
#             alpha: Smoothing factor on the running average reward for each sampled bucket-combination.
#             Larger alpha makes the curriculum more sensitive to recent rewards, smaller alpha stabilizes it.
#             default_bins: The number of bins to use for each bucket, default is a single bin.
#         """
#         # Flatten bucket specs and build priority arrays
#         self._bucket_specs: Dict[str, List[Any]] = {}
#         self._priorities: Dict[str, np.ndarray] = {}
#         self._visits: Dict[str, np.ndarray] = {}
#         self._clean2path: Dict[str, str] = {}

#         for path, spec in buckets.items():
#             opts = _buckets_from_spec(spec, default_bins)
#             self._bucket_specs[path] = opts
#             self._priorities[path] = np.ones(len(opts), dtype=np.float32)
#             self._visits[path] = np.zeros(len(opts), dtype=np.int32)

#             # loss-free label for parent_id strings
#             clean_label = path.replace(".", "_").replace("[", "_").replace("]", "").replace('"', "").replace("'", "")
#             self._clean2path[clean_label] = path

#         # single-env priority (kept for symmetry)
#         self._env_visits = 0

#         self._template = env_cfg_template
#         self._env_overrides = env_overrides
#         self._eps = 1e-6

#         # Call parent with the real env key so Hydra finds the config
#         super().__init__(tasks={env_cfg_template: 1.0}, env_overrides=env_overrides, alpha=alpha)

#         # wipe parent’s fixed-key bookkeeping; we manage our own IDs
#         self._reward_averages, self._reward_maxes = {}, {}
#         self._task_weights = {}  # keep attribute but make empty

#     # --------------------------------------------------------------------- #
#     #                              Sampling                                 #
#     # --------------------------------------------------------------------- #
#     def get_task(self) -> Task:
#         override: Dict[str, Any] = {}
#         id_parts: List[str] = []

#         # sample one bin per bucket-dimension
#         for path, opts in self._bucket_specs.items():
#             p_arr = self._priorities[path]
#             idx = int(np.random.choice(len(opts), p=p_arr / p_arr.sum()))
#             self._visits[path][idx] += 1
#             value = opts[idx]

#             clean = next(k for k, v in self._clean2path.items() if v == path)
#             id_parts.append(f"{clean}={idx}")
#             override[path] = value

#         parent_id = "|".join(id_parts)

#         # realise env config with overrides
#         base_dc = config_from_path(self._template, self._env_overrides)
#         raw_dict = OmegaConf.to_container(base_dc, resolve=False)
#         cfg = OmegaConf.create(raw_dict)

#         for p, v in override.items():
#             # cast to template’s original type
#             original = OmegaConf.select(cfg, p)
#             if isinstance(original, int):
#                 v = int(round(v))
#             OmegaConf.update(cfg, p, v, merge=True)

#         final_conf = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
#         sub = curriculum_from_config_path(self._template, final_conf)

#         task = sub.get_task()
#         task.add_parent(self, parent_id)
#         return task

#     # --------------------------------------------------------------------- #
#     #                           Priority update                             #
#     # --------------------------------------------------------------------- #
#     def complete_task(self, parent_id: str, score: float):
#         err = abs(-score) + self._eps
#         priority = err**self._alpha

#         # update each chosen bucket’s priority
#         for label in parent_id.split("|"):
#             clean, idx_str = label.split("=")
#             path = self._clean2path[clean]
#             self._priorities[path][int(idx_str)] = priority

#         # rolling stats for this specific combination (for inspection/logging)
#         avg = self._reward_averages.get(parent_id, 0.0)
#         avg = (1 - self._alpha) * avg + self._alpha * score
#         self._reward_averages[parent_id] = avg
#         self._reward_maxes[parent_id] = max(self._reward_maxes.get(parent_id, -np.inf), score)

#         logger.debug(
#             f"[Bucketed] {parent_id}: reward={score:.3f}, avg={avg:.3f}, max={self._reward_maxes[parent_id]:.3f}"
#         )
