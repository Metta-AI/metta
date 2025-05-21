import numpy as np
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path
from mettagrid.mettagrid_env import MettaGridEnv


class MettaGridEnvSet(MettaGridEnv):
    """
    This is a wrapper around MettaGridEnv that allows for multiple environments to be used for training
    with prioritized environment replay.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        alpha: float = 0.6,  # Priority exponent (controls how much prioritization is used)
        beta: float = 0.4,  # Initial importance sampling weight (increases to 1 over time)
        beta_annealing_steps: int = 100000,  # Number of episodes to anneal beta to 1.0
        epsilon: float = 0.01,  # Small constant to avoid zero priority
        **kwargs,
    ):
        self._env_cfgs = env_cfg.envs
        self._num_agents_global = env_cfg.num_agents
        self._num_envs = len(self._env_cfgs)

        # Prioritized replay parameters
        self._alpha = alpha
        self._beta = beta
        self._beta_annealing_steps = beta_annealing_steps
        self._beta_increment = (1.0 - beta) / beta_annealing_steps if beta_annealing_steps > 0 else 0
        self._epsilon = epsilon

        # Initialize tracking variables
        self._episode_count = 0
        self._current_env_idx = None
        self._env_priorities = np.ones(self._num_envs, dtype=np.float32)  # Initial priorities are uniform
        self._env_performance = np.zeros(self._num_envs, dtype=np.float32)  # Track performance for each env
        self._env_visits = np.zeros(self._num_envs, dtype=np.int32)  # Count how many times each env is visited

        super().__init__(env_cfg, render_mode, buf=buf, env_map=None, **kwargs)
        self._cfg_template = None  # we don't use this with multiple envs, so we clear it to emphasize that fact

    def _update_priorities(self, env_idx: int, performance: float):
        """
        Update the priority for a specific environment based on agent performance.

        Args:
            env_idx: Index of the environment
            performance: Performance metric (e.g., negative reward or TD error)
        """
        # Update performance tracking
        self._env_performance[env_idx] = performance

        # Update priority with absolute error (raised to power of alpha)
        self._env_priorities[env_idx] = (abs(performance) + self._epsilon) ** self._alpha

    def _get_env_probabilities(self):
        """
        Calculate probabilities for environment selection based on priorities.
        """
        # Normalize priorities to get probabilities
        total_priority = np.sum(self._env_priorities)
        return self._env_priorities / total_priority if total_priority > 0 else np.ones(self._num_envs) / self._num_envs

    def _get_importance_sampling_weight(self, env_idx: int):
        """
        Calculate importance sampling weight for the selected environment.
        """
        # Calculate probability of selecting this environment
        probs = self._get_env_probabilities()

        # Calculate importance sampling weight (to correct for bias)
        weight = (1.0 / (self._num_envs * probs[env_idx])) ** self._beta

        # Normalize weight by max weight to keep values reasonable
        max_weight = (1.0 / (self._num_envs * np.min(probs))) ** self._beta
        return weight / max_weight

    def _update_beta(self):
        """Anneal beta parameter towards 1.0 over time."""
        self._beta = min(1.0, self._beta + self._beta_increment)

    def _get_new_env_cfg(self):
        """
        Select an environment based on prioritized replay probabilities.
        """
        self._env_map = None
        # Get probabilities based on priorities
        probs = self._get_env_probabilities()

        # Select environment based on priorities
        env_idx = np.random.choice(self._num_envs, p=probs)
        self._current_env_idx = env_idx
        self._env_visits[env_idx] += 1

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
        Reset the environment and update priorities based on performance in the previous episode.
        """
        # Update priorities based on last episode's performance if applicable
        if self._current_env_idx is not None:
            # Use a performance metric from the previous episode (e.g., negative reward)
            # The current implementation assumes we want to prioritize environments where
            # agents are struggling (lower rewards = higher priority)
            if hasattr(self, "_last_episode_reward"):
                # Use negative reward as error (higher error = higher priority)
                performance = -self._last_episode_reward
                self._update_priorities(self._current_env_idx, performance)

        # Increment episode counter and update beta
        self._episode_count += 1
        self._update_beta()

        # Standard reset procedure
        obs, infos = super().reset(seed, options)
        return obs, infos

    def step(self, actions):
        """
        Step the environment and track necessary information for prioritization.
        """
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # compute weight
        is_weight = self._get_importance_sampling_weight(self._current_env_idx)
        # put it in infos for each agent/step
        infos["importance_weight"] = np.array([is_weight] * len(rewards))
        # Store information needed for prioritization
        if (terminals.all() or truncations.all()) and "episode/reward.mean" in infos:
            self._last_episode_reward = infos["episode/reward.mean"]

        return observations, rewards, terminals, truncations, infos

    def get_env_stats(self):
        """
        Return statistics about environment selection for logging/debugging.
        """
        return {
            "env_priorities": self._env_priorities.copy(),
            "env_probabilities": self._get_env_probabilities(),
            "env_visits": self._env_visits.copy(),
            "env_performance": self._env_performance.copy(),
            "beta": self._beta,
            "episode_count": self._episode_count,
        }


class SamplingPrioritizedEnvSet(MettaGridEnv):
    """
    A wrapper around MettaGridEnv that performs prioritized environment replay
    across multiple environment configs and continuous sampling parameters
    (via discretized bins), preserving integer types when appropriate.

    This version correctly handles keys containing dots by using
    bracket-notation paths when updating the OmegaConf.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing_steps: int = 100_000,
        epsilon: float = 1e-3,
        num_bins: int = 7,
        **kwargs,
    ):
        # Base configs
        self._env_cfgs = env_cfg.envs
        self._num_envs = len(self._env_cfgs)

        # Prioritized replay parameters
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = (1.0 - beta) / beta_annealing_steps
        self._epsilon = epsilon

        # Env-level trackers
        self._env_priorities = np.ones(self._num_envs, dtype=np.float32)
        self._env_visits = np.zeros(self._num_envs, dtype=np.int32)

        # Sampling bins
        self._num_bins = num_bins

        # Discover sampling macros and bin them
        self._param_options = []  # env_idx -> {path_expr: [bin_values]}
        self._param_priorities = []  # env_idx -> {path_expr: np.array}
        self._param_visits = []  # env_idx -> {path_expr: np.array}

        def collect_and_bin(node, prefix_keys, out):
            if isinstance(node, dict):
                for k, v in node.items():
                    collect_and_bin(v, prefix_keys + [k], out)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    collect_and_bin(v, prefix_keys + [str(i)], out)
            elif isinstance(node, str) and node.startswith("${sampling:"):
                # build bracket-notation path: e.g. ['objects', 'mine.red', 'cooldown'] -> 'objects["mine.red"].cooldown'
                def make_path_expr(keys):
                    expr = keys[0]
                    for seg in keys[1:]:
                        if seg.isdigit():
                            expr += f"[{seg}]"
                        elif "." in seg:
                            expr += f'["{seg}"]'
                        else:
                            expr += f".{seg}"
                    return expr

                args = node[len("${sampling:") : -1].split(",")
                lo, hi, ctr = map(float, args)
                bins = np.linspace(lo, hi, self._num_bins)
                if lo.is_integer() and hi.is_integer() and ctr.is_integer():
                    bins = np.round(bins).astype(int)
                bins = [int(b) if isinstance(b, (np.integer, int)) else float(b) for b in bins]

                path_expr = make_path_expr(prefix_keys)
                out[path_expr] = bins

        for raw_path in self._env_cfgs:
            # unresolved raw container
            raw = OmegaConf.to_container(config_from_path(raw_path), resolve=False)
            opts = {}
            collect_and_bin(raw, [], opts)

            # init priorities & visits
            prio_map = {p: np.ones(len(bins), dtype=np.float32) for p, bins in opts.items()}
            visit_map = {p: np.zeros(len(bins), dtype=np.int32) for p, bins in opts.items()}
            self._param_options.append(opts)
            self._param_priorities.append(prio_map)
            self._param_visits.append(visit_map)

        # current choice tracking
        self._current_choice = {"env": None, "params": {}}

        # first config
        self._env_cfg = self._get_new_env_cfg()
        super().__init__(env_cfg, render_mode, buf=buf, env_map=None, **kwargs)
        self._cfg_template = None

    def _get_new_env_cfg(self):
        # 1) env sampling
        env_p = self._env_priorities / self._env_priorities.sum()
        e = np.random.choice(self._num_envs, p=env_p)
        self._env_visits[e] += 1
        raw = OmegaConf.create(config_from_path(self._env_cfgs[e]))
        self._current_choice.update({"env": e, "params": {}})

        # 2) per-param sampling
        for path_expr, bins in self._param_options[e].items():
            prio = self._param_priorities[e][path_expr]
            probs = prio / prio.sum()
            i = np.random.choice(len(bins), p=probs)

            # record & visit
            self._current_choice["params"][path_expr] = i
            self._param_visits[e][path_expr][i] += 1

            # inject via bracket-path
            OmegaConf.update(raw, path_expr, bins[i], merge=False)

        # resolve macros
        resolved = OmegaConf.to_container(raw, resolve=True)
        return OmegaConf.create(resolved)

    def reset(self, *args, **kwargs):
        if hasattr(self, "_last_episode_reward"):
            err = abs(-self._last_episode_reward) + self._epsilon
            prio = err**self._alpha
            e = self._current_choice["env"]
            # update env
            self._env_priorities[e] = prio
            # update params
            for path_expr, idx in self._current_choice["params"].items():
                self._param_priorities[e][path_expr][idx] = prio

        self._beta = min(1.0, self._beta + self._beta_increment)
        self._episode_count = getattr(self, "_episode_count", 0) + 1
        return super().reset(*args, **kwargs)

    def step(self, actions):
        o, r, t, tr, info = super().step(actions)
        if (t.all() or tr.all()) and "episode/reward.mean" in info:
            self._last_episode_reward = info["episode/reward.mean"]
        return o, r, t, tr, info

    def get_env_stats(self):
        return {
            "env_priorities": self._env_priorities.copy(),
            "env_visits": self._env_visits.copy(),
            "param_priorities": self._param_priorities,
            "param_visits": self._param_visits,
            "beta": self._beta,
            "episode_count": getattr(self, "_episode_count", 0),
        }
