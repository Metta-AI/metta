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
    each environment-config file path to a total initial weight.  We discover every
    `${sampling:lo,hi,ctr}` macro inside each env-config, discretize into `num_bins`
    bins, enumerate all bucket combinations, and finally assign each bucket a weight
    = (env_weight / number_of_combinations_for_that_env).

    Inheriting from LowRewardCurriculum means we get:
      • RandomCurriculum’s weighted-sampling machinery
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
            tasks: Dict[str, float] mapping each env-config path → total weight.
                   e.g. {"cfgs/nav_easy.yaml": 1.0, "cfgs/nav_hard.yaml": 2.0}
            env_overrides: DictConfig (Hydra/OmegaConf overrides) that you’d normally
                           pass to curriculum_from_config_path.
            num_bins: how many discrete buckets each `${sampling:...}` should be split into.
            alpha: smoothing factor for LowRewardCurriculum’s moving average.
        """
        # 1) We'll treat each key of `tasks` as a file‐path to an OmegaConf YAML.
        self._env_paths = list(tasks.keys())
        self._num_bins = num_bins
        self._env_total_weights = tasks  # store each env's total weight

        # 2) For each env‐path, load the raw YAML directly from disk, then collect
        #    all `${sampling:lo,hi,ctr}` occurrences into bin-lists.
        self._param_options: List[Dict[str, List[Any]]] = []
        for env_path in self._env_paths:
            # Instead of instantiating a curriculum and calling `.env_cfg()`,
            # we directly load the YAML file. This avoids the AttributeError.
            raw_conf = OmegaConf.load(env_path)

            # If you *do* need to apply env_overrides to see how sampling might
            # already have been overridden, you could merge:
            # merged = OmegaConf.merge(raw_conf, env_overrides)
            # raw_container = OmegaConf.to_container(merged, resolve=False)
            #
            # But for discovering the literal "${sampling:...}" tokens, we want the file's
            # original (un‐resolved) text. So we do:
            raw_container = OmegaConf.to_container(raw_conf, resolve=False)

            opts: Dict[str, List[Any]] = {}
            self._collect_and_bin(raw_container, prefix_keys=[], out=opts)
            self._param_options.append(opts)

        # 3) Build a new dict `bucketed_tasks: Dict[str, float]`.  For each env,
        #    we enumerate all combinations of bins.  If an env has N total combos,
        #    each bucket gets weight = (env_weight / N).
        bucketed_tasks: Dict[str, float] = {}
        self._subcfg_overrides: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        #   maps bucket_id → (env_index, { path_expr → chosen_bin_value })

        for e_idx, env_path in enumerate(self._env_paths):
            opts = self._param_options[e_idx]
            env_weight = self._env_total_weights[env_path]

            if not opts:
                # If there are no `${sampling:...}` macros in this YAML, create a single
                # “no-sampling” bucket that carries the full env_weight.
                bucket_id = f"env{e_idx}__no_sampling"
                bucketed_tasks[bucket_id] = env_weight
                self._subcfg_overrides[bucket_id] = (e_idx, {})
                continue

            # items = [(path_expr1, bins_list1), (path_expr2, bins_list2), ...]
            items = list(opts.items())
            param_keys = [item[0] for item in items]
            bins_lists = [item[1] for item in items]

            # Count how many total bucket‐combinations: ∏(len(bins_list))
            num_combinations = 1
            for b in bins_lists:
                num_combinations *= len(b)
            if num_combinations == 0:
                num_combinations = 1

            per_bucket_weight = env_weight / float(num_combinations)

            # Enumerate the Cartesian product of all bins
            for choice in itertools.product(*bins_lists):
                override_map: Dict[str, Any] = {}
                suffix_parts: List[str] = [f"env{e_idx}"]

                for key, bin_val in zip(param_keys, choice):
                    override_map[key] = bin_val
                    # sanitize float → string ("0.333" → "0p333")
                    if isinstance(bin_val, float):
                        sanitized = str(bin_val).replace(".", "p")
                    else:
                        sanitized = str(bin_val)
                    suffix_parts.append(f"{key.replace('.', '_')}={sanitized}")

                bucket_id = "__".join(suffix_parts)
                bucketed_tasks[bucket_id] = per_bucket_weight
                self._subcfg_overrides[bucket_id] = (e_idx, override_map)

        # 4) Store env_overrides for later, so that when we actually build a sub-curriculum,
        #    we can still pass any global Hydra overrides if needed.
        self._env_overrides = env_overrides

        # 5) Now call LowRewardCurriculum.__init__(tasks=bucketed_tasks,...).  That parent
        #    constructor will run MultiTaskCurriculum → RandomCurriculum → LowRewardCurriculum,
        #    set up self._task_weights = bucketed_tasks, and allocate self._reward_averages
        #    and self._reward_maxes to zero for each bucket_id.
        super().__init__(tasks=bucketed_tasks, env_overrides=env_overrides)

        # Override alpha if the user asked for a different smoothing factor
        self._alpha = alpha

        # Re‐zero out reward arrays in case the parent did something unexpected
        self._reward_averages = {t: 0.0 for t in bucketed_tasks.keys()}
        self._reward_maxes = {t: 0.0 for t in bucketed_tasks.keys()}


    def _collect_and_bin(
        self,
        node: Any,
        prefix_keys: List[str],
        out: Dict[str, List[Any]]
    ) -> None:
        """
        Recursively scan a raw Python container (dict / list / primitive). Whenever we see
        a string of the form "${sampling:lo,hi,ctr}", we:
          1) parse lo, hi, ctr (as floats)
          2) create `self._num_bins` equally spaced values between lo and hi (inclusive),
             rounding to int if all of (lo, hi, ctr) are integer-valued
          3) build a bracket-notation `path_expr` (e.g. 'objects["mine.red"].cooldown')
          4) store out[path_expr] = [bin_value_1, bin_value_2, …]
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

            # 1) Discretize into self._num_bins equally spaced bins
            bins = np.linspace(lo, hi, num=self._num_bins)
            # 2) If lo, hi, ctr are all integer-valued, cast to ints
            if lo.is_integer() and hi.is_integer() and ctr.is_integer():
                bins = np.round(bins).astype(int)

            bins_list: List[Any] = [
                int(b) if isinstance(b, (np.integer, int)) else float(b)
                for b in bins
            ]

            # 3) Build bracket‐notation path expression from prefix_keys
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
            # Primitive (int/float/bool) or a normal string → no-op
            return


    def get_task(self) -> Task:
        """
        Override RandomCurriculum.get_task() to:
          1) Pick one bucket_id via weighted‐sampling on self._task_weights.
          2) Look up (env_idx, override_map) = self._subcfg_overrides[bucket_id].
          3) Reload that env’s YAML, apply override_map via OmegaConf.update(...), resolve macros,
             then call curriculum_from_config_path(...) to build a sub-curriculum.
          4) Draw a Task from that sub-curriculum, do task.add_parent(self, bucket_id), return it.
        """
        # 1) Weighted‐choice of bucket_id
        all_ids = list(self._task_weights.keys())
        all_w = np.array(list(self._task_weights.values()), dtype=np.float64)
        probs = all_w / all_w.sum()
        chosen_idx = np.random.choice(len(all_ids), p=probs)
        bucket_id = all_ids[chosen_idx]

        # 2) Find which env & which param‐overrides to apply
        env_idx, override_map = self._subcfg_overrides[bucket_id]
        env_path = self._env_paths[env_idx]

        # 3) Reload the base YAML from disk, apply all override_map → bin_values, then resolve
        raw_conf = OmegaConf.load(env_path)
        tmp_conf = OmegaConf.create(OmegaConf.to_container(raw_conf, resolve=False))
        for path_expr, val in override_map.items():
            OmegaConf.update(tmp_conf, path_expr, val, merge=False)
        resolved = OmegaConf.to_container(tmp_conf, resolve=True)
        final_conf = OmegaConf.create(resolved)

        # 4) Build the actual sub-curriculum (with that resolved conf), draw a Task, and attach parent
        sub_curr = curriculum_from_config_path(env_path, final_conf)
        task: Task = sub_curr.get_task()
        task.add_parent(self, bucket_id)
        logger.debug(f"[BucketedCurriculum] Chose bucket {bucket_id} → sub-task {task.name()}")
        return task


    def complete_task(self, id: str, score: float):
        """
        Override LowRewardCurriculum.complete_task.  Here, `id` is exactly the bucket_id
        string (because we called task.add_parent(self, bucket_id) in get_task).  We update
        that bucket’s moving average and max, recompute all weights via:
          w_t = ε + (max_reward[t] / (avg_reward[t] + ε))
        Then call super() so the parent can do any extra work (logging, etc.).
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
