from __future__ import annotations
import itertools
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .low_reward import LowRewardCurriculum     # Adjust if your file name/path differs
from mettagrid.curriculum.curriculum import Task
from mettagrid.curriculum.util import curriculum_from_config_path, config_from_path

logger = logging.getLogger(__name__)


class BucketedCurriculum(LowRewardCurriculum):
    """
    A “bucketed” low‐reward curriculum with per‐parameter prioritized sampling (no full Cartesian product).

    - We take `tasks: Dict[str,float]`, where each key is a Hydra‐style env‐key
      (e.g. "/env/mettagrid/navigation/training/terrain_from_numpy") and each
      value is a “base weight” (float). Hydra resolves that key via `config_from_path(...)`.

    - For each environment (index e), we build:
        • env_priority[e]: a single float = 1.0  (initial priority)
        • env_visit[e]: int = 0                    (visit counter)
        • param_options[e]: Dict[path_expr→List[bin_values]]  (all bins for each `${sampling:…}` in that env)
        • param_priorities[e][path_expr]: np.ndarray (ones, length = #bins)
        • param_visits[e][path_expr]: np.ndarray (zeros, length = #bins)

    - get_task():
        1) Sample e ∼ env_priorities[e]/∑env_priorities
        2) For each parameter path_expr in param_options[e], sample a bin index i from
           param_priorities[e][path_expr]/sum(...)
        3) Build override_map { path_expr: chosen_bin_value }
        4) Build parent_id string = "env{e}|<path1>=<i1>|<path2>=<i2>|…"
        5) Load the fully merged DictConfig via config_from_path(raw_key, env_overrides),
           apply override_map via OmegaConf.update(…, merge=False), resolve, then call
           curriculum_from_config_path(raw_key, resolved_conf) to get a real sub-curriculum.
        6) Return one Task from that sub-curriculum, after doing task.add_parent(self, parent_id).

    - complete_task(parent_id, score):
        1) Parse out e and each chosen bin index from the parent_id string
        2) Compute “priority” = (abs(-score) + ε)^alpha
        3) Update env_priorities[e] = priority,
           Update param_priorities[e][path_expr][chosen_bin] = priority for each param
        4) Maintain own reward buffers self._reward_averages[parent_id], self._reward_maxes[parent_id]
           so you can inspect each sampled combination later if you wish.
        5) (Do **not** call super().complete_task(parent_id,score) because parent expects keys = real env‐keys)

    Inheriting from LowRewardCurriculum only to get its `__init__` shape and have the fields `_reward_averages`
    and `_reward_maxes` in place—otherwise, we override all sampling logic.
    """

    def __init__(
        self,
        tasks: Dict[str, float],
        env_overrides: DictConfig,
        num_bins: int = 3,
        alpha: float = 0.01,
    ):
        """
        Args:
          tasks: Dict mapping Hydra‐style env‐key → base_weight (float).
                 e.g.
                   {
                     "/env/mettagrid/navigation/training/terrain_from_numpy": 1.0,
                     "/env/mettagrid/navigation/training/cylinder_world":     1.0,
                     "/env/mettagrid/navigation/training/varied_terrain_sparse": 1.0,
                   }
          env_overrides: Hydra/OmegaConf overrides (passed later into curriculum_from_config_path).
          num_bins: number of equally‐spaced bins for each `${sampling:…}` parameter.
          alpha: smoothing factor for LowRewardCurriculum’s moving‐average (used only in complete_task).
        """
        # 1) Save the list of env‐keys, the number of bins, and base weights
        self._env_keys = list(tasks.keys())
        self._num_bins = num_bins
        self._env_base_weights = tasks
        self._alpha = alpha

        # 2) Initialize per-env data structures
        num_envs = len(self._env_keys)

        #  2a) One priority per env
        self._env_priorities = np.ones(num_envs, dtype=np.float32)
        self._env_visits = np.zeros(num_envs, dtype=np.int64)

        #  2b) For each env index e, we will collect:
        #      - param_options[e]:    Dict[str→List[Any]]    (path_expr → list of bin‐values)
        #      - param_priorities[e]: Dict[str→np.ndarray]   (path_expr → priority array over bins)
        #      - param_visits[e]:     Dict[str→np.ndarray]   (path_expr → visit counts over bins)
        self._param_options: List[Dict[str, List[Any]]] = []
        self._param_priorities: List[Dict[str, np.ndarray]] = []
        self._param_visits: List[Dict[str, np.ndarray]] = []

        #  2c) For each env_key, extract its unresolved DictConfig, convert to container (resolve=False),
        #      find all "${sampling:lo,hi,ctr}" macros, discretize each into num_bins, and store them.
        for raw_key in self._env_keys:
            # Load the unresolved DictConfig (Hydra will merge defaults for us here, but resolve=False
            # means we still see literal "${sampling:…}" in the container).
            dc: DictConfig = config_from_path(raw_key, env_overrides)
            raw_container = OmegaConf.to_container(dc, resolve=False)

            # Find all sampling macros and build bins
            opts: Dict[str, List[Any]] = {}
            self._collect_and_bin(raw_container, prefix_keys=[], out=opts)
            self._param_options.append(opts)

            # For each path_expr, initialize a priority array of ones and visits array of zeros
            prio_map: Dict[str, np.ndarray] = {}
            visit_map: Dict[str, np.ndarray] = {}
            for path_expr, bins in opts.items():
                prio_map[path_expr] = np.ones(len(bins), dtype=np.float32)
                visit_map[path_expr] = np.zeros(len(bins), dtype=np.int64)
            self._param_priorities.append(prio_map)
            self._param_visits.append(visit_map)

        # 3) Keep env_overrides so get_task() can pass them when building sub‐curriculum
        self._env_overrides = env_overrides

        # 4) Call LowRewardCurriculum.__init__ **with the real env keys** so Hydra finds them:
        #    The parent will call `curriculum_from_config_path(key, env_overrides)` for each key,
        #    so those must actually exist in your Hydra env= group.
        parent_task_map = { raw_key: float(self._env_base_weights[raw_key]) for raw_key in self._env_keys }
        super().__init__(tasks=parent_task_map, env_overrides=env_overrides)

        # 5) Overwrite parent’s reward buffers (we’ll track per‐“bucket string” below instead).
        self._reward_averages = {}
        self._reward_maxes = {}

        # And remove parent’s `_task_weights` (we do our own sampling on `env_priorities`)
        del self._task_weights

        # Track the number of episodes so far
        self._episode_count = 0
        self._last_episode_reward = 0.0


    def _collect_and_bin(
        self,
        node: Any,
        prefix_keys: List[str],
        out: Dict[str, List[Any]],
    ) -> None:
        """
        Recursively scan a raw Python container (dict/list/primitive). Whenever we see
        a string of the form "${sampling:lo,hi,ctr}", we:
          1) parse lo, hi, ctr  (as floats)
          2) call np.linspace(lo, hi, num=self._num_bins) → an array of length `num_bins`
          3) if lo,hi,ctr are integer‐valued, round bins to int
          4) build a bracket‐notation path_expr (e.g. 'objects["mine.red"].cooldown')
          5) store out[path_expr] = [bin1, bin2, …]
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

            # 2) Discretize into num_bins equally spaced
            bins = np.linspace(lo, hi, num=self._num_bins)
            # 3) If all of (lo, hi, ctr) are integer‐valued, round to int
            if lo.is_integer() and hi.is_integer() and ctr.is_integer():
                bins = np.round(bins).astype(int)

            bins_list = [
                int(b) if isinstance(b, (np.integer, int)) else float(b)
                for b in bins
            ]

            # 4) Build bracket‐notation path expression
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
            # primitive or normal string: do nothing
            return


    def get_task(self) -> Task:
        """
        Override LowRewardCurriculum.get_task(). We:
          1) Sample env index e ∼ env_priorities[e]/sum(env_priorities)
          2) For each path_expr in param_options[e], sample one bin index i ∼
             param_priorities[e][path_expr] / sum(param_priorities[e][path_expr])
          3) Build override_map = { path_expr: bins_list[i] for each parameter }
          4) Build parent_id string = "env{e}|<cleaned_path1>={i1}|<cleaned_path2>={i2}|…"
          5) Load fully‐merged config via base_dc = config_from_path(raw_key, self._env_overrides),
             convert to container (resolve=False), `OmegaConf.update(...)` for each override,
             resolve everything, then `curriculum_from_config_path(raw_key, final_conf)` to get
             a real Curriculum. Call sub_curr.get_task(), do task.add_parent(self, parent_id), return.
        """
        # 1) Sample environment
        env_probs = self._env_priorities / self._env_priorities.sum()
        e = int(np.random.choice(len(self._env_keys), p=env_probs))
        self._env_visits[e] += 1

        raw_key = self._env_keys[e]
        param_opts = self._param_options[e]
        param_prios = self._param_priorities[e]
        # Build both override_map (for values) and suffix parts (for parent_id)
        override_map: Dict[str, Any] = {}
        suffix_parts: List[str] = [f"env{e}"]

        # 2) For each sampling‐parameter in this env, sample one bin
        for path_expr, bins_list in param_opts.items():
            prio_array = param_prios[path_expr]
            probs = prio_array / prio_array.sum()
            i_bin = int(np.random.choice(len(bins_list), p=probs))
            chosen_value = bins_list[i_bin]
            override_map[path_expr] = chosen_value
            self._param_visits[e][path_expr][i_bin] += 1

            # Clean up path_expr into a valid label, e.g. 'objects["mine.red"].cooldown'
            #   →  'objects_mine_red_cooldown'
            clean_label = (
                path_expr.replace(".", "_")
                          .replace("[", "_")
                          .replace("]", "")
                          .replace('"', "")
                          .replace("'", "")
            )
            suffix_parts.append(f"{clean_label}={i_bin}")

        parent_id = "|".join(suffix_parts)

        # 3) Load the fully merged Hydra config, apply override_map, resolve, build sub‐cur
        base_dc: DictConfig = config_from_path(raw_key, self._env_overrides)
        raw_container = OmegaConf.to_container(base_dc, resolve=False)
        tmp_conf = OmegaConf.create(raw_container)
        for path_expr, val in override_map.items():
            OmegaConf.update(tmp_conf, path_expr, val, merge=False)
        resolved = OmegaConf.to_container(tmp_conf, resolve=True)
        final_conf = OmegaConf.create(resolved)

        sub_curr = curriculum_from_config_path(raw_key, final_conf)
        task: Task = sub_curr.get_task()
        task.add_parent(self, parent_id)
        logger.debug(f"[BucketedCurriculum] Sampled {parent_id} → sub‐task {task.name()}")
        return task


    def complete_task(self, id: str, score: float):
        """
        After an episode, Trainer calls complete_task(parent_id, score).
        `id` looks like "env{e}|label1={i1}|label2={i2}|…".
        We:
          1) Parse out e and each bin index i_j.
          2) Compute low‐reward priority = (| -score | + ε)^alpha.
          3) env_priorities[e] = priority
             param_priorities[e][path_expr][i_j] = priority  for each parameter
          4) Maintain self._reward_averages[id] and self._reward_maxes[id].
          5) (Do not call super().complete_task, because parent expects id to be an env‐key.)
        """
        # 1) Parse e and the bin indices
        pieces = id.split("|")
        e_str = pieces[0]  # e.g. "env2"
        e = int(e_str[len("env") :])

        # Reconstruct the list of path_exprs in the same order we sampled them:
        # We know param_opts = list(self._param_options[e].keys()) in insertion order.
        param_keys = list(self._param_options[e].keys())

        # 2) Compute low‐reward “priority”
        err = abs(-score) + 1e-6
        prio = err**self._alpha

        # 3) Update env priority
        self._env_priorities[e] = prio

        # 4) Update each param_priority for the chosen bin
        #    pieces[1:] look like ["objects_mine_red_cooldown=3", "agent_speed=1", …]
        for idx, piece in enumerate(pieces[1:]):
            # piece: "<clean_label>=<i_bin>"
            _, bin_idx_str = piece.split("=")
            i_bin = int(bin_idx_str)
            path_expr = param_keys[idx]
            self._param_priorities[e][path_expr][i_bin] = prio

        # 5) Maintain our own LowReward-like buffers _reward_averages/_reward_maxes
        if id not in self._reward_averages:
            self._reward_averages[id] = 0.0
            self._reward_maxes[id] = 0.0

        old_avg = self._reward_averages[id]
        new_avg = (1 - self._alpha) * old_avg + self._alpha * score
        self._reward_averages[id] = new_avg
        self._reward_maxes[id] = max(self._reward_maxes[id], score)

        # (We DO NOT call super().complete_task here, because its `id` must be a real env‐key,
        #  and `id` is a bucket string. If you want parent to log something, you can override
        #  LowRewardCurriculum’s complete_task separately, but it is not required.)

        logger.debug(f"[BucketedCurriculum] Updated priorities for {id}: avg={new_avg:.3f}, max={self._reward_maxes[id]:.3f}, env_prio={prio:.3f}")
