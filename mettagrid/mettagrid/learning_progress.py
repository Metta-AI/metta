from pdb import set_trace as T

import numpy as np
from collections import defaultdict

import pufferlib
from gymnasium.spaces import Discrete

import numpy as np
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path
from mettagrid.mettagrid_env import MettaGridEnv


class BidirectionalLearningProgess:
    def __init__(self, search_space, ema_alpha = 0.001, p_theta = 0.05, num_active_tasks = 16, rand_task_rate = 0.25,
                 sample_threshold = 10, memory = 25):
        # try reducing ema_alpha more? do tuning sweep over that
        # also do the sweep on p_theta
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

        # should we continue collecting
        #  or if we have enough data to update the learning progress
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
            # Assume that any perfect success rate is actually 75% due to evaluation precision.
            # Prevents NaN probabilities and prevents task from being completely ignored.
            self.random_baseline = np.minimum(task_success_rates, 0.75)
            self.task_rates = task_success_rates

        # Update task scores
        normalized_task_success_rates = np.maximum(
            task_success_rates[update_mask] - self.random_baseline[update_mask],
            np.zeros(task_success_rates[update_mask].shape)) / (1.0 - self.random_baseline[update_mask])

        if self._p_fast is None:
            # Initial values
            self._p_fast = normalized_task_success_rates[update_mask]
            self._p_slow = normalized_task_success_rates[update_mask]
            self._p_true = task_success_rates[update_mask]
        else:
            # Exponential moving average
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
                    # we could be finishing up a rollout from last time
                    # don't count that one below
                    self.outcomes[task_id].append(res)
                    if task_id in self.sample_levels:
                        self.counter[task_id] += 1

    def continue_collecting(self):
        return self.collecting

    def _learning_progress(self, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow

        return abs(fast - slow)

    def _reweight(self, p: np.ndarray, p_theta: float = 0.1) -> float:
        """
        Reweight the given success rate using the reweighting function from the paper.
        """
        numerator = p * (1.0 - p_theta)
        denominator = p + p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        """ Sigmoid function for reweighting the learning progress."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        """ Return sampling distribution over the task space based on the learning progress."""
        task_dist = np.ones(self.num_tasks) / self.num_tasks

        learning_progress = self._learning_progress()
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0

        subprobs = learning_progress[posidxs] if any_progress else learning_progress
        std = np.std(subprobs)
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1)  # z-score
        subprobs = self._sigmoid(subprobs)  # sigmoid
        subprobs = subprobs / np.sum(subprobs)  # normalize
        if any_progress:
            # If some tasks have nonzero progress, zero out the rest
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            # If all tasks have 0 progress, return uniform distribution
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
            # collect more data on the current batch of tasks
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


class MettaGridEnvLPSet(MettaGridEnv):
    """
    This is a wrapper around MettaGridEnv that allows for multiple environments to be used for training
    with learning progress.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        ema_alpha: float = 0.001, # Exponential moving average alpha (controls how quickly the learning progress is updated) 
        p_theta: float = 0.05, # Reweighting parameter (controls how much the learning progress is reweighted towards unsolved tasks)
        num_active_tasks: int = 16, # Number of tasks to sample from the task space
        rand_task_rate: float = 0.25, # Probability of sampling a random task 
        sample_threshold: int = 10, # Minimum number of samples required to update the learning progress
        memory: int = 25, # Number of samples to keep in memory for each task
        lp_metric: str = 'episode/reward.mean', # Metric to use for learning progress (e.g., episode reward)
        **kwargs,
    ):
        self._env_cfgs = env_cfg.envs
        self._num_agents_global = env_cfg.num_agents
        self._num_envs = len(self._env_cfgs)
        self._episode_count = 0

        # Learning progress parameters
        self._ema_alpha = ema_alpha
        self._p_theta = p_theta
        self._num_active_tasks = num_active_tasks
        self._rand_task_rate = rand_task_rate
        self._sample_threshold = sample_threshold
        self._memory = memory
        self._lp_metric = lp_metric

        # Get initial environment config
        self._env_cfg = self._get_new_env_cfg()
        self.all_levels = np.arange(self._num_envs)
        self.lp_levels = np.arange(self._num_envs)
        self.sampling_dist = np.ones(self._num_envs) / self._num_envs
        self._current_env_idx = np.random.choice(self.all_levels)

        self.lp = BidirectionalLearningProgess(
                        search_space=self._num_envs, 
                        num_active_tasks=self.num_active_tasks,
                        rand_task_rate=self.rand_task_rate,
                        sample_threshold=self.sample_threshold,
                        memory=self.memory
                    )
        self.send_lp_metrics = False

        super().__init__(env_cfg, render_mode, buf=buf, env_map=None, **kwargs)
        self._cfg_template = None  # we don't use this with multiple envs, so we clear it to emphasize that fact

    def _update_learning_progress(self, env_idx: int, performance: float):
        """
        Update the priority for a specific environment based on agent performance.

        Args:
            env_idx: Index of the environment
            performance: Performance metric (e.g., negative reward or TD error)
        """
        # Update performance tracking
        self.lp.collect_data({f'tasks/{env_idx}': [performance]})

    def _get_env_probabilities(self):
        """
        Calculate probabilities for environment selection based on lp.
        """
        dist, _ = self.lp.calculate_dist()
        return dist

    def _get_new_env_cfg(self):
        """
        Select an environment based on lp probabilities.
        """
        self._env_map = None
        # Updates the LP, if enough data has been collected
        # Get levels selected by the lp as best to learn on
        #  this will internally handle exploration
        _, self.lp_levels = self.lp.calculate_dist()

        # Select environment from the lp curated tasks (with exploration)
        env_idx = np.random.choice(self.lp_levels)
        self._current_env_idx = env_idx

        # Get the environment configuration
        selected_env = self._env_cfgs[env_idx]
        env_cfg = config_from_path(selected_env)

        # Check consistency in number of agents
        if self._num_agents_global != env_cfg.game.num_agents:
            raise ValueError(
                "For MettaGridEnvSet, the number of agents must be the same for all environments. "
                f"Global: {self._num_agents_global}, Env: {env_cfg.game.num_agents}"
            )

        env_cfg = OmegaConf.create(env_cfg)
        OmegaConf.resolve(env_cfg)
        return env_cfg

    def reset(self, seed=None, options=None):
        """
        Reset the environment and select a new environment based on lp
        """
        # Increment episode counter
        self._episode_count += 1
        if self._episode_count > self._num_envs:
            self.send_lp_metrics = True
        # Standard reset procedure which
        #  will call the overloaded _get_new_env_cfg method 
        #  to select a new environment according to lp
        obs, infos = super().reset(seed, options)
        return obs, infos

    def step(self, actions):
        """
        Step the environment and track necessary information for lp.
        """
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # Store information needed for lp
        if (terminals.all() or truncations.all()) and "episode/reward.mean" in infos:
            metric = self.lp_metric
            self._update_learning_progress(self._current_env_idx, infos[metric])
            if self.send_lp_metrics:
                infos[f'{self._env_cfg_idx}/{metric}'] = infos[metric]
                self.lp.add_stats(infos)
            self._last_episode_reward = infos[metric]

        return observations, rewards, terminals, truncations, infos

    def get_env_stats(self):
        """
        Return statistics about environment selection for logging/debugging.
        """
        infos = {}
        self.lp.add_stats(infos)
        return infos

