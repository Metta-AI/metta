from pdb import set_trace as T

import numpy as np
from collections import defaultdict

import pufferlib
from gymnasium.spaces import Discrete

from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path


class BidirectionalLearningProgess:
    def __init__(self, search_space, ema_alpha = 0.001, p_theta = 0.05, num_active_tasks = 16, rand_task_rate = 0.25,
                 sample_threshold = 10, memory = 25):
        if isinstance(search_space, int):
            search_space = Discrete(search_space)
        assert isinstance(search_space, Discrete), f"search_space must be a Discrete space or int, got {type(search_space)}"
        self.search_space = search_space
        self.num_tasks = max_num_levels = search_space.n
        self.ema_alpha = ema_alpha
        self.p_theta = p_theta
        self.n = int(num_active_tasks)
        self.rand_task_rate = rand_task_rate
        self.sample_threshold = sample_threshold
        self.memory = int(memory)
        self.outcomes = {}
        for i in range(max_num_levels):
            self.outcomes[i] = []
        self.ema_tsr = None
        self._p_fast = None
        self._p_slow = None
        self._p_true = None
        self.random_baseline = None
        self.task_success_rate = np.zeros(max_num_levels)
        self.task_sampled_tracker = max_num_levels * [0]
        self.mean_samples_per_eval = []
        self.num_nans = []
        self.collecting = True
        self.update_mask = np.ones(max_num_levels).astype(bool)
        self.sample_levels = np.arange(max_num_levels).astype(np.int32)
        self.counter = {i: 0 for i in self.sample_levels}

    def add_stats(self, info):
        info['lp/num_active_tasks'] = len(self.sample_levels)
        info['lp/mean_sample_prob'] = np.mean(self.task_dist)
        info['lp/num_zeros_lp_dist'] = np.sum(self.task_dist == 0)
        info['lp/task_1_success_rate'] = self.task_success_rate[0]
        info[f'lp/task_{self.num_tasks // 2}_success_rate'] = self.task_success_rate[self.num_tasks // 2]
        info['lp/last_task_success_rate'] = self.task_success_rate[-1]
        info['lp/task_success_rate'] = np.mean(self.task_success_rate)
        info['lp/mean_evals_per_task'] = self.mean_samples_per_eval[-1]
        info['lp/num_nan_tasks'] = self.num_nans[-1]

    def _update(self):
        task_success_rates = np.array([np.mean(self.outcomes[i]) for i in range(self.num_tasks)])
        update_mask = self.update_mask

        if self.random_baseline is None:
            self.random_baseline = np.minimum(task_success_rates, 0.75)
            self.task_rates = task_success_rates

        normalized_task_success_rates = np.maximum(
            task_success_rates[update_mask] - self.random_baseline[update_mask],
            np.zeros(task_success_rates[update_mask].shape)) / (1.0 - self.random_baseline[update_mask])

        if self._p_fast is None:
            self._p_fast = normalized_task_success_rates[update_mask]
            self._p_slow = normalized_task_success_rates[update_mask]
            self._p_true = task_success_rates[update_mask]
        else:
            self._p_fast[update_mask] = (normalized_task_success_rates * self.ema_alpha) + (self._p_fast[update_mask] * (1.0 - self.ema_alpha))
            self._p_slow[update_mask] = (self._p_fast[update_mask] * self.ema_alpha) + (self._p_slow[update_mask] * (1.0 - self.ema_alpha))
            self._p_true[update_mask] = (task_success_rates[update_mask] * self.ema_alpha) + (self._p_true[update_mask] * (1.0 - self.ema_alpha))

        self.task_rates[update_mask] = task_success_rates[update_mask]
        self._stale_dist = True
        self.task_dist = None

        return task_success_rates

    def collect_data(self, infos):
        if not bool(infos):
            return

        for k, v in infos.items():
            if 'tasks' in k:
                task_id = int(k.split('/')[1])
                for res in v:
                    self.outcomes[task_id].append(res)
                    if task_id in self.sample_levels:
                        self.counter[task_id] += 1

    def continue_collecting(self):
        return self.collecting

    def _learning_progress(self, reweight: bool = True) -> float:
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow
        return abs(fast - slow)

    def _reweight(self, p: np.ndarray) -> float:
        numerator = p * (1.0 - self.p_theta)
        denominator = p + self.p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        task_dist = np.ones(self.num_tasks) / self.num_tasks
        learning_progress = self._learning_progress()
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0
        subprobs = learning_progress[posidxs] if any_progress else learning_progress
        std = np.std(subprobs)
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1)
        subprobs = self._sigmoid(subprobs)
        subprobs = subprobs / np.sum(subprobs)
        if any_progress:
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            task_dist = subprobs
        self.task_dist = task_dist.astype(np.float32)
        self._stale_dist = False
        out_vec = [np.mean(self.outcomes[i]) for i in range(self.num_tasks)]
        self.num_nans.append(sum(np.isnan(out_vec)))
        self.task_success_rate = np.nan_to_num(out_vec)
        self.mean_samples_per_eval.append(np.mean([len(self.outcomes[i]) for i in range(self.num_tasks)]))
        for i in range(self.num_tasks):
            self.outcomes[i] = self.outcomes[i][-self.memory:]
        self.collecting = True
        return self.task_dist

    def _sample_tasks(self):
        sample_levels = []
        self.update_mask = np.zeros(self.num_tasks).astype(bool)
        for i in range(self.n):
            if np.random.rand() < self.rand_task_rate:
                level = np.random.choice(range(self.num_tasks))
            else:
                level = np.random.choice(range(self.num_tasks), p=self.task_dist)
            sample_levels.append(level)
            self.update_mask[level] = True
        self.sample_levels = np.array(sample_levels).astype(np.int32)
        self.counter = {i: 0 for i in self.sample_levels}
        return self.sample_levels

    def calculate_dist(self):
        if all([v < self.sample_threshold for k, v in self.counter.items()]) and self.random_baseline is not None:
            return self.task_dist, self.sample_levels
        self.task_success_rate = self._update()
        dist = self._sample_distribution()
        tasks = self._sample_tasks()
        return dist, tasks

    def reset_outcomes(self):
        self.prev_outcomes = self.outcomes
        self.outcomes = {}
        for i in range(self.num_tasks):
            self.outcomes[i] = []


class MettaGridEnvLPSet:
    """
    This is a wrapper around MettaGridEnv that allows for multiple environments to be used for training
    with learning progress.
    Uses composition to avoid circular import issues.
    """
    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        ema_alpha: float = 0.001,
        p_theta: float = 0.05,
        num_active_tasks: int = 16,
        rand_task_rate: float = 0.25,
        sample_threshold: int = 10,
        memory: int = 25,
        lp_metric: str = 'episode/reward.mean',
        **kwargs,
    ):
        from mettagrid.mettagrid_env import MettaGridEnv
        self._env_cfgs = env_cfg.envs
        self._num_agents_global = env_cfg.num_agents
        self._num_envs = len(self._env_cfgs)
        self._episode_count = 0
        self._ema_alpha = ema_alpha
        self._p_theta = p_theta
        self._num_active_tasks = num_active_tasks
        self._rand_task_rate = rand_task_rate
        self._sample_threshold = sample_threshold
        self._memory = memory
        self._lp_metric = lp_metric
        self._env_cfg = self._get_new_env_cfg()
        self.all_levels = np.arange(self._num_envs)
        self.lp_levels = np.arange(self._num_envs)
        self.sampling_dist = np.ones(self._num_envs) / self._num_envs
        self._current_env_idx = np.random.choice(self.all_levels)
        self.lp = BidirectionalLearningProgess(
            search_space=self._num_envs,
            num_active_tasks=self._num_active_tasks,
            rand_task_rate=self._rand_task_rate,
            sample_threshold=self._sample_threshold,
            memory=self._memory
        )
        self.send_lp_metrics = False
        # Compose the actual environment
        self.env = MettaGridEnv(self._env_cfg, render_mode, buf=buf, env_map=None, **kwargs)
        self._cfg_template = None

    def _update_learning_progress(self, env_idx: int, performance: float):
        self.lp.collect_data({f'tasks/{env_idx}': [performance]})

    def _get_env_probabilities(self):
        dist, _ = self.lp.calculate_dist()
        return dist

    def _get_new_env_cfg(self):
        self._env_map = None
        _, self.lp_levels = self.lp.calculate_dist()
        env_idx = np.random.choice(self.lp_levels)
        self._current_env_idx = env_idx
        selected_env = self._env_cfgs[env_idx]
        env_cfg = config_from_path(selected_env)
        if self._num_agents_global != env_cfg.game.num_agents:
            raise ValueError(
                "For MettaGridEnvSet, the number of agents must be the same for all environments. "
                f"Global: {self._num_agents_global}, Env: {env_cfg.game.num_agents}"
            )
        env_cfg = OmegaConf.create(env_cfg)
        OmegaConf.resolve(env_cfg)
        return env_cfg

    def reset(self, seed=None, options=None):
        self._episode_count += 1
        if self._episode_count > self._num_envs:
            self.send_lp_metrics = True
        obs, infos = self.env.reset(seed, options)
        return obs, infos

    def step(self, actions):
        observations, rewards, terminals, truncations, infos = self.env.step(actions)
        if (terminals.all() or truncations.all()) and "episode/reward.mean" in infos:
            metric = self._lp_metric
            self._update_learning_progress(self._current_env_idx, infos[metric])
            if self.send_lp_metrics:
                infos[f'{self._current_env_idx}/{metric}'] = infos[metric]
                self.lp.add_stats(infos)
            self._last_episode_reward = infos[metric]
        return observations, rewards, terminals, truncations, infos

    def get_env_stats(self):
        infos = {}
        self.lp.add_stats(infos)
        return infos

    def __getattr__(self, name):
        # Delegate attribute access to the composed environment
        return getattr(self.env, name)
