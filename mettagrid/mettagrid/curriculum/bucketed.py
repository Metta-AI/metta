from __future__ import annotations
import itertools
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .low_reward import LowRewardCurriculum     # <— same place where you already define LowRewardCurriculum
from mettagrid.curriculum.curriculum import Task
from mettagrid.curriculum.util import curriculum_from_config_path, config_from_path

logger = logging.getLogger(__name__)


class BucketedCurriculum(LowRewardCurriculum):
    """
    A “bucketed” low-reward curriculum that takes a `tasks: Dict[str,float>` mapping
    each Hydra‐style env‐key (e.g. "/env/mettagrid/navigation/training/terrain_from_numpy")
    to a total weight.  We:
      1. Call config_from_path(raw_path, env_overrides) to obtain the *unresolved* DictConfig.
      2. Convert that to a raw Python container (resolve=False) so we can see literal
         "${sampling:lo,hi,ctr}" tokens.
      3. For each such token, discretize into `num_bins`.
      4. Enumerate every Cartesian‐product bucket combination; each bucket becomes one
         “sub‐task” whose initial weight = (env_total_weight / number_of_combinations).
      5. Call LowRewardCurriculum.__init__(bucketed_tasks, env_overrides), which sets up
         the normal moving‐average + max‐reward logic over “bucket ids.”

    When get_task() is called, we:
      • Pick a “bucket id” by weighted sampling on self._task_weights.
      • Look up which (env_idx, override_map) corresponds to that bucket.
      • Re‐load the original Hydra config via config_from_path(raw_path, env_overrides),
        apply the few overrides for that bucket, resolve all macros, and then call
        curriculum_from_config_path(raw_path, resolved_conf) to get a real Curriculum.
      • Draw a Task from that sub‐curriculum, add_parent(self, bucket_id), and return it.

    In this way, you reuse exactly the same pattern as `SamplingPrioritizedEnvSet` for
    reading + binning each env‐config, but wrap the result in a LowRewardCurriculum.
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
          tasks: Dict mapping each *Hydra‐style* env‐key to a total weight.
                 e.g.
                   {
                     "/env/mettagrid/navigation/training/terrain_from_numpy": 1.0,
                     "/env/mettagrid/navigation/training/cylinder_world":     1.0,
                     "/env/mettagrid/navigation/training/varied_terrain_sparse": 1.0,
                   }
          env_overrides: any global Hydra/OmegaConf overrides you want passed on
                         to curriculum_from_config_path when we finally build each sub‐curriculum.
          num_bins: how many discrete buckets to cut each `${sampling:...}` into.
          alpha: smoothing factor for LowRewardCurriculum’s moving average.
        """
        # ————————————
        # 1) Store the Hydra‐style env‐keys in a list
        # ————————————
        self._env_keys = list(tasks.keys())  # e.g. ["/env/mettagrid/navigation/training/terrain_from_numpy", …]
        self._num_bins = num_bins
        self._env_total_weights = tasks     # store each env’s total weight

        # ——————————————
        # 2) For each env_key, grab the *unresolved* DictConfig via config_from_path()
        #    then convert to a Python container (resolve=False) so we see literal "${sampling:...}".
        # ——————————————
        self._param_options: List[Dict[str, List[Any]]] = []
        for raw_path in self._env_keys:
            # This is exactly what SamplingPrioritizedEnvSet did:
            #    raw_cfg = OmegaConf.to_container(config_from_path(raw_path, env_overrides), resolve=False)
            dc = config_from_path(raw_path, env_overrides)  # returns a Hydra/OmegaConf DictConfig
            raw_cfg = OmegaConf.to_container(dc, resolve=False)  # a plain dict/list/primitive

            # Now scan raw_cfg for any "${sampling:lo,hi,ctr}" and build bins
            opts: Dict[str, List[Any]] = {}
            self._collect_and_bin(raw_cfg, prefix_keys=[], out=opts)
            self._param_options.append(opts)

        # ————————————————————————————
        # 3) Build bucketed_tasks: Dict[bucket_id, float]
        #    and record for each bucket_id which (env_idx, override_map) it corresponds to.
        # ————————————————————————————
        bucketed_tasks: Dict[str, float] = {}
        self._subcfg_overrides: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        #    maps bucket_id -> (env_index, { path_expr -> chosen_bin_value })

        for e_idx, raw_path in enumerate(self._env_keys):
            opts = self._param_options[e_idx]
            total_weight = self._env_total_weights[raw_path]

            if not opts:
                # If this env has NO sampling macros, create one “no_sampling” bucket
                bucket_id = f"env{e_idx}__no_sampling"
                bucketed_tasks[bucket_id] = total_weight
                self._subcfg_overrides[bucket_id] = (e_idx, {})
                continue

            # Otherwise, we have items = [(path_expr1, bins_list1), (path_expr2, bins_list2), …]
            items = list(opts.items())
            param_keys = [item[0] for item in items]
            bins_lists = [item[1] for item in items]

            # How many total combinations?  ∏ len(bins_list)
            num_combinations = 1
            for b in bins_lists:
                num_combinations *= len(b)
            if num_combinations == 0:
                num_combinations = 1

            per_bucket_weight = total_weight / float(num_combinations)

            for choice in itertools.product(*bins_lists):
                override_map: Dict[str, Any] = {}
                suffix_parts: List[str] = [f"env{e_idx}"]

                for key, bin_val in zip(param_keys, choice):
                    override_map[key] = bin_val
                    # sanitize floats → “0.333” → “0p333” for readability
                    if isinstance(bin_val, float):
                        sanitized = str(bin_val).replace(".", "p")
                    else:
                        sanitized = str(bin_val)
                    suffix_parts.append(f"{key.replace('.', '_')}={sanitized}")

                bucket_id = "__".join(suffix_parts)
                bucketed_tasks[bucket_id] = per_bucket_weight
                self._subcfg_overrides[bucket_id] = (e_idx, override_map)

        # ————————————————
        # 4) Remember env_overrides for later, then call LowRewardCurriculum.__init__
        # ————————————————
        self._env_overrides = env_overrides
        super().__init__(tasks=bucketed_tasks, env_overrides=env_overrides)

        # Override the default alpha if requested
        self._alpha = alpha

        # Force‐zero out reward arrays (in case parent set them differently)
        self._reward_averages = {t: 0.0 for t in bucketed_tasks.keys()}
        self._reward_maxes = {t: 0.0 for t in bucketed_tasks.keys()}


    def _collect_and_bin(
        self,
        node: Any,
        prefix_keys: List[str],
        out: Dict[str, List[Any]],
    ) -> None:
        """
        Exactly the same “discover & discretize” logic as in your SamplingPrioritizedEnvSet:
        Recursively scan a raw Python container (dict / list / primitive). Whenever we see a
        string of the form "${sampling:lo,hi,ctr}", we:
          1) parse lo,hi,ctr  (floats)
          2) call np.linspace(lo, hi, num=self._num_bins) to get N bins
          3) if lo/hi/ctr were integer‐valued, round to int
          4) build a bracket‐notation path_expr (e.g. 'objects["mine.red"].cooldown')
          5) store out[path_expr] = [bin_1, bin_2, …]
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

            # 1) Build equally spaced bins
            bins = np.linspace(lo, hi, num=self._num_bins)
            # 2) If lo,hi,ctr are integer‐valued, round to int
            if lo.is_integer() and hi.is_integer() and ctr.is_integer():
                bins = np.round(bins).astype(int)

            bins_list = [
                int(b) if isinstance(b, (np.integer, int)) else float(b) for b in bins
            ]

            # 3) Build bracket‐notation path_expr from prefix_keys
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
            # primitive or normal string → ignore
            return


    def get_task(self) -> Task:
        """
        Override LowRewardCurriculum.get_task() (which normally calls RandomCurriculum.get_task()):
          1) We do a weighted‐choice on self._task_weights to pick one bucket_id.
          2) Look up (env_idx, override_map) = self._subcfg_overrides[bucket_id].
          3) Reload the original Hydra config via config_from_path(raw_path, env_overrides),
             apply `override_map` via OmegaConf.update(..., merge=False), resolve macros,
             then call curriculum_from_config_path(raw_path, resolved_conf) to obtain a real sub‐Curriculum.
          4) Call sub_curr.get_task(), do task.add_parent(self, bucket_id), and return it.
        """
        # 1) Weighted pick
        all_ids = list(self._task_weights.keys())
        all_w  = np.array(list(self._task_weights.values()), dtype=np.float64)
        probs  = all_w / all_w.sum()
        chosen_idx = np.random.choice(len(all_ids), p=probs)
        bucket_id   = all_ids[chosen_idx]

        # 2) Look up which env & overrides
        env_idx, override_map = self._subcfg_overrides[bucket_id]
        raw_path = self._env_keys[env_idx]

        # 3) Reload the raw Hydra config, apply the per‐bucket overrides, resolve
        dc = config_from_path(raw_path, self._env_overrides)  # unresolved DictConfig
        raw_container = OmegaConf.to_container(dc, resolve=False)
        tmp_conf = OmegaConf.create(raw_container)
        for path_expr, val in override_map.items():
            OmegaConf.update(tmp_conf, path_expr, val, merge=False)
        resolved = OmegaConf.to_container(tmp_conf, resolve=True)
        final_conf = OmegaConf.create(resolved)

        # 4) Instantiate a real sub‐curriculum with the final, resolved config
        sub_curr: LowRewardCurriculum = curriculum_from_config_path(raw_path, final_conf)
        task: Task = sub_curr.get_task()
        task.add_parent(self, bucket_id)
        logger.debug(f"[BucketedCurriculum] Selected bucket {bucket_id} → sub-task {task.name()}")
        return task


    def complete_task(self, id: str, score: float):
        """
        Override LowRewardCurriculum.complete_task.  Because we did `task.add_parent(self, bucket_id)` above,
        `id` is exactly the bucket_id string.  We update that bucket’s moving average and max, recompute
        all weights via:
            w_t = ε + (max_reward[t] / (avg_reward[t] + ε))
        then call super().complete_task(id, score).
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
