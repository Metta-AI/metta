from pdb import set_trace as T

import numpy as np
from collections import defaultdict

import pufferlib
from gymnasium.spaces import Discrete

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
            # high_success_idxs = np.where(task_success_rates > 0.75)
            # high_success_rates = task_success_rates[high_success_idxs]
            #  warnings.warn(
            #     f"Tasks {high_success_idxs} had very high success rates {high_success_rates} for random baseline. Consider removing them from the training set of tasks.")
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
            try:
                self._p_fast[update_mask] = (normalized_task_success_rates * self.ema_alpha) + (self._p_fast[update_mask] * (1.0 - self.ema_alpha))
                self._p_slow[update_mask] = (self._p_fast[update_mask] * self.ema_alpha) + (self._p_slow[update_mask] * (1.0 - self.ema_alpha))
                self._p_true[update_mask] = (task_success_rates[update_mask] * self.ema_alpha) + (self._p_true[update_mask] * (1.0 - self.ema_alpha))
            except IndexError:
                T()

        self.task_rates[update_mask] = task_success_rates[update_mask]    # Logging only
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

        # self.task_sampled_tracker = [int(bool(o)) for k, o in self.outcomes.items()]
        # print(f'data collected on {sum(self.task_sampled_tracker)} / {self.num_tasks} tasks')
        # if sum(self.task_sampled_tracker) == self.num_tasks:
            # T()
        # self.collecting = False
        # self.task_sampled_tracker = self.num_tasks * [0]

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
        # clear the outcome dict
        # go through the outcomes and for each task, keep the last 25
        # plot number of nans in the outcomes
        # print(f'number of nans in outcomes: {sum(np.isnan([np.mean(self.outcomes[i]) for i in range(self.num_tasks)]))}')
        out_vec = [np.mean(self.outcomes[i]) for i in range(self.num_tasks)]
        self.num_nans.append(sum(np.isnan(out_vec)))
        # sort sample rates and plot to see how big the tail is
        self.task_success_rate = np.nan_to_num(out_vec)
        self.mean_samples_per_eval.append(np.mean([len(self.outcomes[i]) for i in range(self.num_tasks)]))
        for i in range(self.num_tasks):
            self.outcomes[i] = self.outcomes[i][-self.memory:]
        self.collecting = True
        sample_levels = []
        self.update_mask = np.zeros(self.num_tasks).astype(bool)
        for i in range(self.n):
            if np.random.rand() < self.rand_task_rate:
                level = np.random.choice(range(self.num_tasks))
            else:
                level = np.random.choice(range(self.num_tasks), p=task_dist)
            sample_levels.append(level)
            self.update_mask[level] = True
        self.sample_levels = np.array(sample_levels).astype(np.int32)
        self.counter = {i: 0 for i in self.sample_levels}
        return self.task_dist, self.sample_levels

    def calculate_dist(self):
        if all([v < self.sample_threshold for k, v in self.counter.items()]) and self.random_baseline is not None:
            # collect more data on the current batch of tasks
            return self.task_dist, self.sample_levels
        self.task_success_rate = self._update()
        return self._sample_distribution()

    def reset_outcomes(self):
        self.prev_outcomes = self.outcomes
        self.outcomes = {}
        for i in range(self.num_tasks):
            self.outcomes[i] = [] 


class LPEnvWrapper:
    """Note, this wrapper probably needs some other helper methods that
    simply pass along the call to the env"""
    def __init__(self, env, num_tasks, ema_alpha = 0.001, p_theta = 0.05, num_active_tasks = 16, 
                 rand_task_rate = 0.25, sample_threshold = 10, memory = 25, 
                 use_lp = True, lp_metric='episode/reward.mean'):
        self.env = env
        self.raw_env = env.env
        self.n = num_tasks
        self.use_lp = use_lp
        self.ema_alpha = ema_alpha
        self.p_theta = p_theta
        self.num_active_tasks = num_active_tasks
        self.rand_task_rate = rand_task_rate
        self.sample_threshold = sample_threshold
        self.memory = memory
        self.lp_metric = lp_metric

        self.cfgs = [self.raw_env._get_new_env_cfg() for _ in range(self.n)]
        self.all_levels = np.arange(self.n)
        self.lp_levels = np.arange(self.n)
        self.sampling_dist = np.ones(self.n) / self.n
        self._env_cfg_idx = np.random.choice(self.all_levels)
        self.lp_metric = lp_metric

        self.lp = BidirectionalLearningProgess(search_space=self.n, num_active_tasks=self.num_active_tasks,
                                                rand_task_rate=self.rand_task_rate,
                                                sample_threshold=self.sample_threshold,
                                                memory=self.memory)
        self.send_lp_metrics = False
    
    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)

        if all(term) or all(trunc):
            # alternate possability, send agent/heart.get
            metric = self.lp_metric
            self.lp.collect_data({f'tasks/{self._env_cfg_idx}': [info[metric]]})
            if self.send_lp_metrics:
                info[f'{self._env_cfg_idx}/{metric}'] = info[metric]
                self.lp.add_stats(info)
            self.reset()
            self.env.should_reset = True
            if 'agent_raw' in info:
                del info['agent_raw']
            if 'episode_rewards' in info:
                info['score'] = info['episode_rewards']
        else:
            info = []

        return obs, rew, term, trunc, [info]
    
    def reset(self, seed=None):
        self._env_cfg_idx = self.get_next_task_id()
        self.raw_env._env_cfg = self.cfgs[self._env_cfg_idx]
        self.raw_env._reset_env()

        self.raw_env._c_env.set_buffers(
            self.raw_env.observations,
            self.raw_env.terminals,
            self.raw_env.truncations,
            self.raw_env.rewards)

        obs, infos = self.raw_env._c_env.reset()
        self.raw_env.should_reset = False
        self.tick = 0
        return obs, infos

    def notify(self):
        self.sampling_dist, self.lp_levels = self.lp.calculate_dist()
        self.lp_dist = self.sampling_dist
        self.send_lp_metrics = True

    def get_next_task_id(self):
        return np.random.choice(self.lp_levels)
    
    def get_lp_dist(self):
        return self.sampling_dist

    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
