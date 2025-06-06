from __future__ import annotations
import itertools
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .random import LowRewardCurriculum  # inherit from LowRewardCurriculum
from mettagrid.curriculum.curriculum import Task
from mettagrid.curriculum.util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class BucketedCurriculum(LowRewardCurriculum):
    """
    A “bucketed” low-reward curriculum.

    Inherits from LowRewardCurriculum so that:
      • We automatically get RandomCurriculum’s weighted-sampling machinery.
      • We automatically get LowRewardCurriculum’s moving-average + max tracking and reweighting.

    This subclass discovers all `${sampling:lo,hi,ctr}` macros in each env-config,
    discretizes them into `num_bins` bins each, then instantiates one sub-curriculum
    for every combination of bucket-values.
    """

    def __init__(
        self,
        env_overrides: DictConfig,
        num_bins: int = 7,
        alpha: float = 0.01,
        initial_weight: float = 1.0,
    ):
        """
        Args:
            env_overrides: DictConfig that must contain `envs: List[str]`,
                           where each entry is a path to a base OmegaConf config.
                           Within those configs, any field set to "${sampling:lo,hi,ctr}"
                           will be discovered and bucketed.
            num_bins: how many discrete buckets per sampled parameter.
            alpha: smoothing factor for moving average (overrides LowRewardCurriculum’s default).
            initial_weight: the initial (uniform) weight assigned to each bucket-combination.
        """
        # 1) Read raw list of env-paths from env_overrides.envs (unresolved)
        raw_env_paths: List[str] = OmegaConf.to_container(env_overrides.envs, resolve=False)
        self._num_bins = num_bins

        # 2) For each env index, find all sampling-macros and their bin-lists
        self._param_options: List[Dict[str, List[Any]]] = []
        for env_path in raw_env_paths:
            # Load the base OmegaConf without resolving, so we still see literal "${sampling:…}"
            base_curr = curriculum_from_config_path(env_path, env_overrides)
            raw_cfg = OmegaConf.to_container(base_curr.env_cfg(), resolve=False)
            opts: Dict[str, List[Any]] = {}
            self._collect_and_bin(raw_cfg, prefix_keys=[], out=opts)
            self._param_options.append(opts)

        # 3) Build every bucket-combination as its own “task_id”
        tasks: Dict[str, float] = {}
        self._subcfg_overrides: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        #   maps task_id → (env_index, { path_expr → chosen_bin_value })

        for e_idx, opts in enumerate(self._param_options):
            if not opts:
                # No sampling macros → single “no-sampling” bucket
                task_id = f"env{e_idx}__no_sampling"
                tasks[task_id] = initial_weight
                self._subcfg_overrides[task_id] = (e_idx, {})
                continue

            # items: list of (path_expr, bins_list)
            items = list(opts.items())
            param_keys = [item[0] for item in items]
            bins_lists = [item[1] for item in items]

            for choice in itertools.product(*bins_lists):
                override_map: Dict[str, Any] = {}
                suffix_parts: List[str] = [f"env{e_idx}"]
                for key, bin_val in zip(param_keys, choice):
                    override_map[key] = bin_val
                    # sanitize float → string (replace '.' with 'p') for readability
                    if isinstance(bin_val, float):
                        sanitized = str(bin_val).replace(".", "p")
                    else:
                        sanitized = str(bin_val)
                    suffix_parts.append(f"{key.replace('.', '_')}={sanitized}")

                task_id = "__".join(suffix_parts)
                tasks[task_id] = initial_weight
                self._subcfg_overrides[task_id] = (e_idx, override_map)

        # 4) Store env_overrides for later (so we can reload each env config on-the-fly)
        self._env_overrides = env_overrides

        # 5) Initialize LowRewardCurriculum with our new tasks dictionary.
        #    LowRewardCurriculum → RandomCurriculum → MultiTaskCurriculum will set up:
        #      - self._curriculums  (not actually used directly; we override get_task)
        #      - self._task_weights = tasks (initial)
        #      - self._reward_averages, self._reward_maxes, etc.
        super().__init__(tasks=tasks, env_overrides=env_overrides)

        # Override alpha (smoothing factor) if provided
        self._alpha = alpha

        # Reinitialize reward buffers in case the parent used a different set
        self._reward_averages = {t: 0.0 for t in tasks.keys()}
        self._reward_maxes = {t: 0.0 for t in tasks.keys()}

    def _collect_and_bin(
        self,
        node: Any,
        prefix_keys: List[str],
        out: Dict[str, List[Any]],
    ) -> None:
        """
        Recursively scan a raw Python container (dict/list/primitive). Whenever we see a string
        of the form "${sampling:lo,hi,ctr}", we:
          1) parse lo, hi, ctr
          2) discretize into self._num_bins equally spaced buckets between lo and hi (inclusive),
             rounding to int if lo/hi/ctr are integer-valued
          3) build a bracket-notation path_expr (e.g. 'objects["mine.red"].cooldown')
          4) store out[path_expr] = list_of_bin_values
        """
        if isinstance(node, dict):
            for k, v in node.items():
                self._collect_and_bin(v, prefix_keys + [k], out)

        elif isinstance(node, list):
            for idx, v in enumerate(node):
                self._collect_and_bin(v, prefix_keys + [str(idx)], out)

        elif isinstance(node, str) and node.startswith("${sampling:"):
            inner = node[len("${sampling:") : -1]
            lo, hi, ctr = map(float, inner.split(","))

            # Discretize into self._num_bins equally spaced values
            bins = np.linspace(lo, hi, num=self._num_bins)
            # If lo, hi, ctr are all integer-valued, cast bins to ints
            if lo.is_integer() and hi.is_integer() and ctr.is_integer():
                bins = np.round(bins).astype(int)
            bins_list = [
                int(b) if isinstance(b, (np.integer, int)) else float(b)
                for b in bins
            ]

            # Build bracket-notation path_expr from prefix_keys
            def make_path_expr(keys: List[str]) -> str:
                expr = keys[0]
                for seg in keys[1:]:
                    if seg.isdigit():
                        expr += f"[{seg}]"
                    elif "." in seg:
                        expr += f'["{seg}"]'
                    else:
                        expr += f".{seg}"
                return expr

            path_expr = make_path_expr(prefix_keys)
            out[path_expr] = bins_list

        else:
            # Primitive or irrelevant → do nothing
            return

    def get_task(self) -> Task:
        """
        Overrides RandomCurriculum.get_task() to:
          1) Pick one bucketed task_id according to current self._task_weights.
          2) Look up (env_idx, override_map) = self._subcfg_overrides[task_id].
          3) Load that env’s base OmegaConf, apply all overrides in override_map, resolve macros,
             then build a new sub-curriculum via curriculum_from_config_path().
          4) Draw a Task from the sub-curriculum, call task.add_parent(self, task_id),
             and return it.
        """
        # 1) Weighted choice on task_id
        all_ids = list(self._task_weights.keys())
        all_w = np.array(list(self._task_weights.values()), dtype=np.float64)
        probs = all_w / all_w.sum()
        chosen_idx = np.random.choice(len(all_ids), p=probs)
        task_id = all_ids[chosen_idx]

        # 2) Look up which env index & override map
        env_idx, override_map = self._subcfg_overrides[task_id]
        env_path = OmegaConf.to_container(self._env_overrides.envs, resolve=False)[env_idx]

        # 3) Reload base OmegaConf for this env, apply overrides, resolve
        base_curr = curriculum_from_config_path(env_path, self._env_overrides)
        raw_cfg = OmegaConf.to_container(base_curr.env_cfg(), resolve=False)
        tmp_conf = OmegaConf.create(raw_cfg)
        for path_expr, val in override_map.items():
            OmegaConf.update(tmp_conf, path_expr, val, merge=False)
        resolved = OmegaConf.to_container(tmp_conf, resolve=True)
        final_conf = OmegaConf.create(resolved)

        # 4) Build the actual sub-curriculum and draw a Task
        sub_curr = curriculum_from_config_path(env_path, final_conf)
        task: Task = sub_curr.get_task()
        task.add_parent(self, task_id)
        logger.debug(f"[BucketedCurriculum] Chose bucket {task_id}, sub-task {task.name()}")
        return task

    def complete_task(self, id: str, score: float):
        """
        Override LowRewardCurriculum.complete_task. Since LowRewardCurriculum expects `id` to be
        one of its task-keys, and because we called task.add_parent(self, task_id) above,
        `id` here is exactly our bucketed task_id string (e.g. "env0__paramA=0p0__paramB=1p2").

        We update the moving average / max for that bucket, recompute all bucket weights,
        then call super() so that LowRewardCurriculum does any additional bookkeeping.
        """
        old_avg = self._reward_averages[id]
        self._reward_averages[id] = (1 - self._alpha) * old_avg + self._alpha * score
        self._reward_maxes[id] = max(self._reward_maxes[id], score)

        eps = 1e-6
        self._task_weights = {
            t: eps + (self._reward_maxes[t] / (self._reward_averages[t] + eps))
            for t in self._task_weights.keys()
        }

        super().complete_task(id, score)
