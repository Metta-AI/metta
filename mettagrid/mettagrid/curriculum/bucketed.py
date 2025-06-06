from __future__ import annotations
import itertools
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .low_reward import LowRewardCurriculum  # <-- inherit from LowRewardCurriculum
from mettagrid.curriculum.curriculum import Task
from mettagrid.curriculum.util import curriculum_from_config_path

logger = logging.getLogger(__name__)


class BucketedCurriculum(LowRewardCurriculum):
    """
    A “bucketed” low-reward curriculum that takes a `tasks: Dict[str, float]` mapping
    each environment-config path to a total initial weight.  We discover every
    `${sampling:lo,hi,ctr}` macro in each env-config, discretize into `num_bins`
    bins, enumerate all bucket combinations, and assign each bucket a weight
    equal to (env_weight / number_of_combinations_in_that_env).

    Inheriting from LowRewardCurriculum gives us:
      • RandomCurriculum’s weighted sampling machinery
      • LowRewardCurriculum’s moving-average + max tracking (complete_task reweights)
    """

    def __init__(
        self,
        tasks: Dict[str, float],
        env_overrides: DictConfig,
        num_bins: int = 7,
        alpha: float = 0.01,
    ):
        """
        Args:
            tasks: Dict mapping each env-config path (string) → total weight (float).
                   E.g. {"cfgs/env_easy.yaml": 1.0, "cfgs/env_hard.yaml": 2.0}
            env_overrides: DictConfig (the same override container you’d pass to
                           curriculum_from_config_path).
            num_bins: how many discrete buckets per sampled parameter.
            alpha: smoothing factor for LowRewardCurriculum’s moving average.
        """
        # 1) Interpret `tasks.keys()` as the list of env-paths (unresolved).
        self._env_paths: List[str] = list(tasks.keys())
        self._num_bins = num_bins
        self._env_total_weights = tasks  # store for later distribution

        # 2) For each env-path, scan its raw OmegaConf (unresolved) to discover all {sampling:lo,hi,ctr}.
        #    Build a list of { path_expr → [bin_values] } for each env.
        raw_env_paths: List[str] = self._env_paths
        self._param_options: List[Dict[str, List[Any]]] = []
        for env_path in raw_env_paths:
            # Load the curriculum ONLY to grab its env_cfg (unresolved)
            base_curr = curriculum_from_config_path(env_path, env_overrides)
            raw_cfg = OmegaConf.to_container(base_curr.env_cfg(), resolve=False)
            opts: Dict[str, List[Any]] = {}
            self._collect_and_bin(raw_cfg, prefix_keys=[], out=opts)
            self._param_options.append(opts)

        # 3) Build a new dict `bucketed_tasks: Dict[str, float]` where:
        #      - each key is a unique bucket ID (e.g. "env0__paramA=0p0__paramB=1p2")
        #      - each value = (env_weight / number_of_combinations_for_that_env)
        bucketed_tasks: Dict[str, float] = {}
        self._subcfg_overrides: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        #     maps bucket_id → (env_index, { path_expr → chosen_bin_value })

        for e_idx, env_path in enumerate(raw_env_paths):
            opts = self._param_options[e_idx]
            env_weight = self._env_total_weights[env_path]

            if not opts:
                # No sampling macros in this env → exactly one “no-sampling” bucket
                bucket_id = f"env{e_idx}__no_sampling"
                bucketed_tasks[bucket_id] = env_weight
                self._subcfg_overrides[bucket_id] = (e_idx, {})
                continue

            # items = list of (param_path_expr, bins_list)
            items = list(opts.items())
            param_keys = [item[0] for item in items]
            bins_lists = [item[1] for item in items]

            # Count how many total bucket-combinations: product of lengths
            num_combinations = 1
            for b in bins_lists:
                num_combinations *= len(b)
            if num_combinations == 0:
                # (shouldn’t happen, but guard anyway)
                num_combinations = 1

            # Each bucket gets equal share = (env_weight / num_combinations)
            per_bucket_weight = env_weight / float(num_combinations)

            # Enumerate the cartesian product:
            for choice in itertools.product(*bins_lists):
                override_map: Dict[str, Any] = {}
                suffix_parts: List[str] = [f"env{e_idx}"]
                for key, bin_val in zip(param_keys, choice):
                    override_map[key] = bin_val
                    # sanitize float → string (replace '.' with 'p')
                    if isinstance(bin_val, float):
                        sanitized = str(bin_val).replace(".", "p")
                    else:
                        sanitized = str(bin_val)
                    suffix_parts.append(f"{key.replace('.', '_')}={sanitized}")

                bucket_id = "__".join(suffix_parts)
                bucketed_tasks[bucket_id] = per_bucket_weight
                self._subcfg_overrides[bucket_id] = (e_idx, override_map)

        # 4) Keep env_overrides around so we can resolve each bucket’s sub-conf on-the-fly.
        self._env_overrides = env_overrides

        # 5) Finally call LowRewardCurriculum with our new bucketed_tasks dictionary.
        #    LowRewardCurriculum→RandomCurriculum→MultiTaskCurriculum will:
        #      • set self._task_weights = bucketed_tasks
        #      • initialize self._reward_averages and self._reward_maxes to zeros for each bucket_id
        super().__init__(tasks=bucketed_tasks, env_overrides=env_overrides)

        # Override alpha if given
        self._alpha = alpha

        # Reset reward arrays in case parent used a different initial set
        self._reward_averages = {t: 0.0 for t in bucketed_tasks.keys()}
        self._reward_maxes = {t: 0.0 for t in bucketed_tasks.keys()}


    def _collect_and_bin(
        self,
        node: Any,
        prefix_keys: List[str],
        out: Dict[str, List[Any]],
    ) -> None:
        """
        Recursively scan a raw Python container (dict/list/primitive). Whenever we see a string
        of the form "${sampling:lo,hi,ctr}", we:
          1) parse lo, hi, ctr (floats)
          2) create `self._num_bins` equally spaced values between lo and hi, rounding to int
             if all of (lo,hi,ctr) are integer-valued
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

            # Discretize into self._num_bins equally spaced bins
            bins = np.linspace(lo, hi, num=self._num_bins)
            # If lo, hi, ctr are all integer-valued, cast bins to ints
            if lo.is_integer() and hi.is_integer() and ctr.is_integer():
                bins = np.round(bins).astype(int)

            bins_list: List[Any] = [
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
            return  # primitive or no-op


    def get_task(self) -> Task:
        """
        Override RandomCurriculum.get_task() to:
          1) Weighted‐choice on self._task_weights to pick a bucket_id.
          2) Look up (env_idx, override_map) = self._subcfg_overrides[bucket_id].
          3) Reload that env’s base OmegaConf, apply override_map via OmegaConf.update(...),
             resolve macros, build a fresh sub-curriculum, and call sub_curr.get_task().
          4) Add parent(self, bucket_id) to the returned Task so complete_task knows which bucket.
        """
        # 1) Weighted sampling of bucket_id
        all_ids = list(self._task_weights.keys())
        all_w = np.array(list(self._task_weights.values()), dtype=np.float64)
        probs = all_w / all_w.sum()
        chosen_idx = np.random.choice(len(all_ids), p=probs)
        bucket_id = all_ids[chosen_idx]

        # 2) Look up env_idx & override_map
        env_idx, override_map = self._subcfg_overrides[bucket_id]
        env_path = self._env_paths[env_idx]

        # 3) Reload the base OmegaConf for this env, apply overrides, resolve
        base_curr = curriculum_from_config_path(env_path, self._env_overrides)
        raw_cfg = OmegaConf.to_container(base_curr.env_cfg(), resolve=False)
        tmp_conf = OmegaConf.create(raw_cfg)
        for path_expr, val in override_map.items():
            OmegaConf.update(tmp_conf, path_expr, val, merge=False)
        resolved = OmegaConf.to_container(tmp_conf, resolve=True)
        final_conf = OmegaConf.create(resolved)

        # 4) Build the actual sub-curriculum, draw a Task, attach parent
        sub_curr = curriculum_from_config_path(env_path, final_conf)
        task: Task = sub_curr.get_task()
        task.add_parent(self, bucket_id)
        logger.debug(f"[BucketedCurriculum] Chose bucket {bucket_id}, sub-task {task.name()}")
        return task


    def complete_task(self, id: str, score: float):
        """
        Override LowRewardCurriculum.complete_task.  Here, `id` is exactly the bucket_id
        string (because we did task.add_parent(self, bucket_id) in get_task).
        We update that bucket’s moving average and moving max, recompute all weights:
          w_t = ε + (max_t / (avg_t + ε))
        Then call super() so any additional parent logic runs.
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
