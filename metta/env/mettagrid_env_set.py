from typing import Optional

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

        # Get initial environment config
        self._env_cfg = self._get_new_env_cfg()

        super().__init__(env_cfg, render_mode, buf=buf, level=None, **kwargs)
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
            "episode_count": getattr(self, "_episode_count", 0),
        }


class ProgressiveEnvSet(MettaGridEnv):
    """
    A curriculum-based environment manager that progressively activates and
    samples from environment ensembles using normalized performance for
    prioritization.

    This class implements a curriculum learning approach where different
    ensembles of environments become active at specified episode counts. Within
    each active ensemble, environments are sampled using a prioritized selection
    mechanism. Prioritization is based on the agent's performance in an
    environment relative to its own historical performance in that same
    environment (using z-score normalization of rewards).

    Key Features:
        - Progressive curriculum: Ensembles activate based on episode count
          thresholds.
        - Normalized prioritized sampling: Environments are sampled based on
          z-score of rewards relative to their own history.
        - Performance tracking: Maintains running statistics (mean, std_dev) of
          rewards for each environment.

    Parameters:
        - env_cfg (DictConfig): Configuration containing ensemble definitions.
        - render_mode (str): Rendering mode for the environment.
        - buf: Optional buffer for environment state.
        - alpha (float): Priority exponent for sampling (default: 0.6).
        - epsilon (float): Small constant added to priorities (default: 0.01).
        - min_samples_for_norm_stats (int): Minimum episodes before using
          z-score normalization (default: 10).
        - max_z_score_magnitude (float): Used to shift z-scores for priority
          calculation (default: 3.0). This value also serves as the high
          priority base for environments with insufficient stats.
    """

    MIN_SAMPLES_FOR_NORM_STATS = 10  # Default, can be overridden by env_cfg
    MAX_Z_SCORE_MAGNITUDE = 3.0  # Default, can be overridden by env_cfg
    # Small constant to prevent division by zero in std_dev and ensure var is non-negative
    NUMERICAL_STABILITY_EPSILON = 1e-6

    def __init__(
        self,
        env_cfg: DictConfig,
        render_mode: str,
        buf=None,
        alpha: float = 0.6,
        epsilon: float = 0.01,
        min_samples_for_norm_stats: Optional[int] = None,  # Allow override from config
        max_z_score_magnitude: Optional[float] = None,  # Allow override from config
        **kwargs,
    ):
        # List to store parsed ensemble configurations
        self._parsed_ensembles = []
        # Store the global number of agents from config
        self._num_agents_global = env_cfg.num_agents

        # Set minimum samples required for normalization statistics
        # Use provided value or fall back to default class constant
        self._min_samples_for_norm_stats = (
            min_samples_for_norm_stats if min_samples_for_norm_stats is not None else self.MIN_SAMPLES_FOR_NORM_STATS
        )
        # Set maximum z-score magnitude for priority calculations
        # Use provided value or fall back to default class constant
        self._max_z_score_magnitude = (
            max_z_score_magnitude if max_z_score_magnitude is not None else self.MAX_Z_SCORE_MAGNITUDE
        )

        # Iterate over each ensemble configuration
        for i, ensemble_conf_item in enumerate(env_cfg.ensembles):
            # Check if the ensemble configuration is a string
            if isinstance(ensemble_conf_item, str):
                raise ValueError(
                    f"Ensemble definition at index {i} is a string. Expected a dictionary with 'name', 'episode_activation', and 'envs'."
                )

            # Extract the name of the ensemble
            name = OmegaConf.select(ensemble_conf_item, "name", default=f"Ensemble_{i}")
            # Extract the episode activation threshold for the ensemble
            activation_episode = OmegaConf.select(ensemble_conf_item, "episode_activation", default=0)

            # Extract the list of environment paths for the ensemble
            env_paths_conf = OmegaConf.select(ensemble_conf_item, "envs")
            # Check if the environment paths configuration is None
            if env_paths_conf is None:
                raise ValueError(f"Ensemble '{name}' (index {i}) is missing the 'envs' list.")

            # Convert the environment paths configuration to a list
            env_paths = list(env_paths_conf)

            # Check if the environment paths list is empty
            if not env_paths:
                raise ValueError(f"Ensemble '{name}' (index {i}) has no environments defined in 'envs' list.")

            # Calculate the number of environments in the ensemble
            num_envs_in_ensemble = len(env_paths)

            # Append the parsed ensemble configuration to the list
            self._parsed_ensembles.append(
                {
                    "name": name,
                    "activation_episode": activation_episode,
                    "env_paths": env_paths,
                    "num_envs_in_ensemble": num_envs_in_ensemble,
                    # Environment-level stats
                    "priorities": np.ones(num_envs_in_ensemble, dtype=np.float32),
                    "last_raw_rewards": np.zeros(num_envs_in_ensemble, dtype=np.float32),
                    "visits": np.zeros(num_envs_in_ensemble, dtype=np.int32),
                    "reward_sum": np.zeros(num_envs_in_ensemble, dtype=np.float64),
                    "reward_sum_sq": np.zeros(num_envs_in_ensemble, dtype=np.float64),
                    "reward_count": np.zeros(num_envs_in_ensemble, dtype=np.int32),
                    # Ensemble-level stats for prioritization
                    "ensemble_priority": 1.0,  # Initial priority for the ensemble
                    "ensemble_priority_base_sum": 0.0,  # Sum of env_priority_base from this ensemble
                    "ensemble_episodes_completed_count": 0,  # Episodes completed in this ensemble
                }
            )

        # Check if no ensembles are defined in the configuration
        if not self._parsed_ensembles:
            raise ValueError("No ensembles defined in the configuration.")

        # Initialize priority parameters
        self._alpha = alpha
        self._epsilon = epsilon

        # Initialize episode count and active ensemble indices
        self._episode_count = 0
        self._active_ensemble_indices = []
        # Initialize current ensemble and environment indices
        self._current_ensemble_idx = None
        self._current_env_idx_in_ensemble = None

        # Initialize the base environment with a dummy configuration
        dummy_initial_cfg_for_super = OmegaConf.create({"game": {"num_agents": self._num_agents_global}})
        super().__init__(dummy_initial_cfg_for_super, render_mode, buf=buf, env_map=None, **kwargs)
        self._cfg_template = None

    def _update_active_ensembles(self):
        """
        Updates the list of active ensembles based on the current episode count
        and activation thresholds.

        This method manages the progressive curriculum by activating ensembles
        when their activation episode threshold is reached. It implements
        several fallback mechanisms to ensure at least one ensemble is active:

        1. Activates ensembles whose activation_episode threshold has been
           reached
        2. If no ensembles are active and the first ensemble has
           activation_episode=0, activates it
        3. If still no active ensembles, activates the ensemble with the lowest
           activation_episode that has been reached
        4. Raises a RuntimeError if no ensembles can be activated

        The method maintains the self._active_ensemble_indices list which tracks
        currently active ensembles. This list is used by other methods to select
        environments for training.

        Raises:
            RuntimeError: If no ensembles can be activated at the current
            episode count.
        """
        # List to track if any ensembles were newly activated
        newly_activated = False
        # Iterate over each ensemble
        for i, ensemble_data in enumerate(self._parsed_ensembles):
            # Check if the ensemble is not already active and the activation episode threshold has been reached
            if i not in self._active_ensemble_indices and self._episode_count >= ensemble_data["activation_episode"]:
                # Activate the ensemble
                self._active_ensemble_indices.append(i)
                # Set the flag to True to indicate that an ensemble was newly activated
                newly_activated = True

        # Check if no active ensembles
        if not self._active_ensemble_indices:
            # This case should ideally not be hit if at least one ensemble has activation_episode=0
            # Or if the logic ensures at least one is active.
            # For safety, activate the first one if none are active yet (e.g. if
            # all activations are > 0 and episode_count is low)
            # However, the design implies at least one ensemble is active from episode 0.
            if self._parsed_ensembles and self._parsed_ensembles[0]["activation_episode"] == 0:
                self._active_ensemble_indices.append(0)
            else:
                # Fallback: if no ensemble is configured for episode 0, activate
                # the one with the lowest activation_episode.
                if self._parsed_ensembles:
                    # Initialize the minimum activation episode and index
                    min_act_ep = float("inf")
                    min_idx = -1
                    # Iterate over each ensemble
                    for idx, ens in enumerate(self._parsed_ensembles):
                        # Check if the activation episode is less than the current minimum
                        if ens["activation_episode"] < min_act_ep:
                            # Update the minimum activation episode and index
                            min_act_ep = ens["activation_episode"]
                            min_idx = idx
                    # Check if the minimum index is not -1 and the activation
                    # episode is greater than or equal to the current episode
                    # count
                    if min_idx != -1 and self._episode_count >= self._parsed_ensembles[min_idx]["activation_episode"]:
                        self._active_ensemble_indices.append(min_idx)

        # Check if no active ensembles and there are parsed ensembles
        if not self._active_ensemble_indices and self._parsed_ensembles:
            # If still no active ensembles, it means all configured activation_episodes are
            # greater than the current _episode_count. This is a valid state if curriculum starts later.
            # However, the game cannot start. This should be handled by ensuring at least one ensemble
            # has activation_episode: 0.
            # For now, let's raise an error if no ensemble can be activated.
            raise RuntimeError(
                f"No ensembles are active at episode {self._episode_count}. "
                "Ensure at least one ensemble has 'episode_activation: 0' or an activation "
                "threshold less than or equal to the current episode count."
            )

    def _get_new_env_cfg(self):
        """
        Selects and returns a new environment configuration based on the
        progressive curriculum system.

        This method implements a two-stage selection process:
            1. First selects an active ensemble based on ensemble priorities
            2. Then selects a specific environment from that ensemble based on
               environment priorities

        The selection uses weighted random sampling where weights are determined
        by priorities. If all priorities are zero, falls back to uniform random
        selection.

        Returns:
            OmegaConf: A resolved environment configuration object

        Raises:
            RuntimeError: If no ensembles are active ValueError: If there's a
            mismatch in the number of agents between the selected environment
            and the global configuration
        """
        # Reset the environment map
        self._env_map = None
        # Update the list of active ensembles
        self._update_active_ensembles()

        # Check if no active ensembles
        if not self._active_ensemble_indices:
            raise RuntimeError("Cannot select an environment: No active ensembles.")

        # Step 1: Select an active ensemble using prioritization
        active_ensemble_priorities = np.array(
            [self._parsed_ensembles[idx]["ensemble_priority"] for idx in self._active_ensemble_indices],
            dtype=np.float32,
        )
        # Calculate the total priority for all active ensembles
        total_ensemble_priority = np.sum(active_ensemble_priorities)

        # Check if the total priority is greater than 0
        if total_ensemble_priority > 0:
            # Calculate the probabilities for each active ensemble
            ensemble_probs = active_ensemble_priorities / total_ensemble_priority
        else:  # Fallback to uniform if all priorities are zero (e.g. at the very start)
            # Fallback to uniform if all priorities are zero (e.g. at the very start)
            ensemble_probs = np.ones(len(self._active_ensemble_indices)) / len(self._active_ensemble_indices)

        # np.random.choice requires probabilities to sum to 1. Small numerical
        # errors can violate this. Thus, we normalize the probabilities.
        ensemble_probs /= np.sum(ensemble_probs)

        # Select an active ensemble using the calculated probabilities
        selected_active_idx = np.random.choice(len(self._active_ensemble_indices), p=ensemble_probs)
        # Set the current ensemble index to the selected active ensemble index
        self._current_ensemble_idx = self._active_ensemble_indices[selected_active_idx]

        # Get the data for the current ensemble
        current_ensemble_data = self._parsed_ensembles[self._current_ensemble_idx]

        # Step 2: Select an environment from the chosen ensemble based on priorities
        # Get the priorities for the current ensemble
        ensemble_env_priorities = current_ensemble_data["priorities"]
        # Calculate the total priority for the current ensemble
        total_env_priority = np.sum(ensemble_env_priorities)
        # Check if the total priority is greater than 0
        env_probs = (
            ensemble_env_priorities / total_env_priority
            if total_env_priority > 0
            else np.ones(current_ensemble_data["num_envs_in_ensemble"]) / current_ensemble_data["num_envs_in_ensemble"]
        )
        # Ensure probabilities sum to 1 for np.random.choice
        env_probs /= np.sum(env_probs)

        # Select an environment from the current ensemble using the calculated
        # probabilities
        self._current_env_idx_in_ensemble = np.random.choice(current_ensemble_data["num_envs_in_ensemble"], p=env_probs)

        # Get the path for the selected environment
        selected_env_path = current_ensemble_data["env_paths"][self._current_env_idx_in_ensemble]
        # Load the environment configuration from the selected environment path
        env_cfg_loaded = config_from_path(selected_env_path)

        # Check if the number of agents in the selected environment matches the global number of agents
        if self._num_agents_global != env_cfg_loaded.game.num_agents:
            raise ValueError(
                f"Mismatch in num_agents for ensemble '{current_ensemble_data['name']}', env '{selected_env_path}'. "
                f"Global: {self._num_agents_global}, Env: {env_cfg_loaded.game.num_agents}"
            )

        # Create a resolved environment configuration object
        env_cfg_resolved = OmegaConf.create(env_cfg_loaded)
        # Resolve the environment configuration
        OmegaConf.resolve(env_cfg_resolved)
        return env_cfg_resolved

    def _update_priority_for_env(
        self, ensemble_data: dict, env_idx_in_ensemble: int, last_episode_reward: float
    ) -> float:
        """
        Updates the priority of an environment within its ensemble based on its
        performance.

        This method calculates a priority value for an environment using a
        z-score based approach:
            1. For environments with insufficient samples (<
               min_samples_for_norm_stats), assigns maximum priority
            2. For environments with sufficient samples:
                - Calculates z-score based on reward history
                - Converts z-score to priority using max_z_score_magnitude
            3. Applies epsilon-greedy exploration and alpha power scaling to
               final priority

        Args:
            ensemble_data (dict): Dictionary containing ensemble statistics and
            data env_idx_in_ensemble (int): Index of the environment within its
            ensemble last_episode_reward (float): Reward received in the last
            episode

        Returns:
            float: The priority_base value used for ensemble-level priority
            calculations
        """
        # Store the last episode reward
        ensemble_data["last_raw_rewards"][env_idx_in_ensemble] = last_episode_reward

        # Get the current reward count for the environment
        current_reward_count = ensemble_data["reward_count"][env_idx_in_ensemble]

        # Initialize the priority base
        priority_base = 0.0

        # Check if the current reward count is less than the minimum samples for
        # normalization statistics
        if current_reward_count < self._min_samples_for_norm_stats:
            # If the current reward count is less than the minimum samples for
            # normalization statistics, assign the maximum priority
            priority_base = self._max_z_score_magnitude
        else:
            # Calculate the mean reward for the environment
            mean_reward = ensemble_data["reward_sum"][env_idx_in_ensemble] / current_reward_count
            # Calculate the variance of the reward history
            variance = (ensemble_data["reward_sum_sq"][env_idx_in_ensemble] / current_reward_count) - (mean_reward**2)
            # Calculate the standard deviation of the reward history
            std_dev = np.sqrt(max(variance, 0)) + self.NUMERICAL_STABILITY_EPSILON

            # Calculate the z-score of the last episode reward
            z_score = (
                (last_episode_reward - mean_reward) / std_dev if std_dev > self.NUMERICAL_STABILITY_EPSILON else 0.0
            )
            # Calculate the error signal
            error_signal = -z_score
            # Calculate the priority base
            priority_base = error_signal + self._max_z_score_magnitude

        # Calculate the final priority value
        final_priority_value = (max(0, priority_base) + self._epsilon) ** self._alpha
        # Update the priority for the environment
        ensemble_data["priorities"][env_idx_in_ensemble] = final_priority_value
        # Return the base for ensemble-level calculation
        return priority_base

    def reset(self, seed=None, options=None):
        """
        Reset the environment and update curriculum statistics.

        This method handles the reset operation for the environment while
        maintaining curriculum progression statistics. It updates reward
        statistics, priorities, and episode counts for both individual
        environments and their ensembles. The method also populates the info
        dictionary with curriculum-related metrics.

        Args:
            seed (int, optional): Random seed for environment reset. Defaults to
            None. options (dict, optional): Additional reset options. Defaults
            to None.

        Returns:
            tuple: A tuple containing:
                - obs: The initial observation after reset
                - infos (dict): Dictionary containing curriculum statistics
                  including:
                    - curriculum/ensemble_name: Name of current ensemble
                    - curriculum/ensemble_idx: Index of current ensemble
                    - curriculum/env_idx_in_ensemble: Index of environment in
                      ensemble
                    - curriculum/env_path: Path to current environment
                    - curriculum/env_priority: Priority of current environment
                    - curriculum/episode_count: Total number of episodes
                      completed
        """
        # Check if the last episode reward is set and the current ensemble and
        # environment indices are not None
        if (
            hasattr(self, "_last_episode_reward")
            and self._current_ensemble_idx is not None
            and self._current_env_idx_in_ensemble is not None
        ):
            # Get the ensemble data for the current ensemble
            ensemble_data = self._parsed_ensembles[self._current_ensemble_idx]
            # Get the index of the current environment in the ensemble
            env_idx = self._current_env_idx_in_ensemble

            # Update the reward statistics for the current environment
            ensemble_data["reward_sum"][env_idx] += self._last_episode_reward
            ensemble_data["reward_sum_sq"][env_idx] += self._last_episode_reward**2
            ensemble_data["reward_count"][env_idx] += 1
            ensemble_data["visits"][env_idx] += 1

            # Update environment-level priority and get its priority_base
            env_priority_base = self._update_priority_for_env(ensemble_data, env_idx, self._last_episode_reward)

            # Update ensemble-level statistics for its priority calculation
            ensemble_data["ensemble_priority_base_sum"] += env_priority_base
            ensemble_data["ensemble_episodes_completed_count"] += 1

            # Check if the ensemble has completed any episodes
            if ensemble_data["ensemble_episodes_completed_count"] > 0:
                # Calculate the average priority base for the ensemble
                avg_ensemble_priority_base = (
                    ensemble_data["ensemble_priority_base_sum"] / ensemble_data["ensemble_episodes_completed_count"]
                )
                # Calculate the ensemble priority
                ensemble_data["ensemble_priority"] = (max(0, avg_ensemble_priority_base) + self._epsilon) ** self._alpha
            else:  # Should not happen if count is incremented before, but as a safeguard
                # Set the ensemble priority to 1.0
                ensemble_data["ensemble_priority"] = 1.0

            # Delete the last episode reward
            del self._last_episode_reward

        # Increment the episode count
        self._episode_count += 1

        # Reset the environment
        obs, infos = super().reset(seed, options)

        # Check if the current ensemble index is not None
        if self._current_ensemble_idx is not None:
            # Get the data for the current ensemble
            current_ensemble_data = self._parsed_ensembles[self._current_ensemble_idx]
            # Set the ensemble name in the info dictionary
            infos["curriculum/ensemble_name"] = current_ensemble_data["name"]
            infos["curriculum/ensemble_idx"] = self._current_ensemble_idx
            infos["curriculum/env_idx_in_ensemble"] = self._current_env_idx_in_ensemble
            infos["curriculum/env_path"] = current_ensemble_data["env_paths"][self._current_env_idx_in_ensemble]
            infos["curriculum/env_priority"] = current_ensemble_data["priorities"][self._current_env_idx_in_ensemble]

        # Set the episode count in the info dictionary
        infos["curriculum/episode_count"] = self._episode_count

        return obs, infos

    def step(self, actions):
        """
        Step the environment and track necessary information for prioritization.
        """
        observations, rewards, terminals, truncations, infos = super().step(actions)

        # Store information needed for prioritization
        if (terminals.all() or truncations.all()) and "episode/reward.mean" in infos:
            self._last_episode_reward = infos["episode/reward.mean"]

        return observations, rewards, terminals, truncations, infos

    def get_env_stats(self):
        """
        Get comprehensive statistics about the curriculum progression and environment selection.

        Returns:
            dict: A dictionary containing:
                - global_episode_count (int): Total number of episodes completed
                - active_ensemble_indices (list): Indices of currently active ensembles
                - active_ensemble_names (list): Names of currently active ensembles
                - active_ensemble_sample_probabilities (list): Sampling probabilities for all ensembles
                - current_ensemble_idx_selected (int): Index of currently selected ensemble
                - current_env_idx_in_ensemble_selected (int): Index of currently selected environment
                - config_min_samples_for_norm_stats (int): Minimum samples required for normalization
                - config_max_z_score_magnitude (float): Maximum z-score magnitude for normalization
                - ensembles_details (list): List of dictionaries containing per-ensemble statistics:
                    - name (str): Ensemble name
                    - activation_episode (int): Episode when ensemble was activated
                    - is_active (bool): Whether ensemble is currently active
                    - ensemble_priority (float): Current priority of ensemble
                    - ensemble_avg_priority_base (float): Average priority base of ensemble
                    - ensemble_episodes_completed (int): Number of completed episodes
                    - num_envs (int): Number of environments in ensemble
                    - env_paths (list): Paths to environments in ensemble
                    - env_priorities (list): Priorities of environments in ensemble
                    - last_raw_rewards (list): Last raw rewards for each environment
                    - visits (list): Number of visits to each environment
                    - reward_counts (list): Number of reward samples per environment
                    - reward_means (list): Mean rewards per environment
                    - reward_stddevs (list): Standard deviations of rewards per environment
                    - env_probabilities_in_ensemble (list): Sampling probabilities within ensemble
        """
        # List to store per-ensemble statistics
        ensemble_stats = []
        # Iterate over each ensemble
        for i, ens_data in enumerate(self._parsed_ensembles):
            # Initialize arrays for mean and std rewards
            mean_rewards = np.zeros_like(ens_data["reward_sum"])
            std_rewards = np.zeros_like(ens_data["reward_sum"])
            # Create a mask for environments with valid statistics
            valid_stats_mask = ens_data["reward_count"] > 0

            # Check if there are any valid statistics
            if np.any(valid_stats_mask):
                # Calculate the mean reward for the environment
                mean_rewards[valid_stats_mask] = (
                    ens_data["reward_sum"][valid_stats_mask] / ens_data["reward_count"][valid_stats_mask]
                )
                # Calculate the variance of the reward history
                variance = (
                    ens_data["reward_sum_sq"][valid_stats_mask] / ens_data["reward_count"][valid_stats_mask]
                ) - (mean_rewards[valid_stats_mask] ** 2)
                # Calculate the standard deviation of the reward history
                std_rewards[valid_stats_mask] = np.sqrt(np.maximum(variance, 0))

            # Initialize the average priority base
            avg_ensemble_priority_base = 0.0
            # Check if the ensemble has completed any episodes
            if ens_data["ensemble_episodes_completed_count"] > 0:
                # Calculate the average priority base for the ensemble
                avg_ensemble_priority_base = (
                    ens_data["ensemble_priority_base_sum"] / ens_data["ensemble_episodes_completed_count"]
                )
            # Create a dictionary for the ensemble statistics
            stats = {
                "name": ens_data["name"],
                "activation_episode": ens_data["activation_episode"],
                "is_active": i in self._active_ensemble_indices,
                "ensemble_priority": ens_data["ensemble_priority"],
                "ensemble_avg_priority_base": avg_ensemble_priority_base,
                "ensemble_episodes_completed": ens_data["ensemble_episodes_completed_count"],
                "num_envs": ens_data["num_envs_in_ensemble"],
                "env_paths": ens_data["env_paths"],
                "env_priorities": ens_data["priorities"].copy(),
                "last_raw_rewards": ens_data["last_raw_rewards"].copy(),
                "visits": ens_data["visits"].copy(),
                "reward_counts": ens_data["reward_count"].copy(),
                "reward_means": mean_rewards,
                "reward_stddevs": std_rewards,
            }
            # Check if the ensemble has any environments
            if ens_data["num_envs_in_ensemble"] > 0:
                # Calculate the total priority for the environments in the ensemble
                total_env_priority_in_ensemble = np.sum(ens_data["priorities"])
                # Calculate the probabilities for the environments in the ensemble
                env_probs_in_ensemble = (
                    ens_data["priorities"] / total_env_priority_in_ensemble
                    if total_env_priority_in_ensemble > 0
                    else np.ones(ens_data["num_envs_in_ensemble"]) / ens_data["num_envs_in_ensemble"]
                )
                stats["env_probabilities_in_ensemble"] = env_probs_in_ensemble
            # Append the ensemble statistics to the list
            ensemble_stats.append(stats)

        # Get the names of the active ensembles
        active_ensemble_names = [
            self._parsed_ensembles[i]["name"] for i in self._active_ensemble_indices if i < len(self._parsed_ensembles)
        ]
        # Get the priorities of the active ensembles
        active_ensemble_priorities_list = [
            self._parsed_ensembles[idx]["ensemble_priority"] for idx in self._active_ensemble_indices
        ]
        # Calculate the total priority for the active ensembles
        total_active_ensemble_priority = np.sum(active_ensemble_priorities_list)
        # Initialize the probabilities for the active ensembles
        active_ensemble_probs = np.zeros(len(self._parsed_ensembles))  # Full list, but only active will be non-zero
        # Check if the total priority is greater than 0 and there are active ensembles
        if total_active_ensemble_priority > 0 and self._active_ensemble_indices:
            # Calculate the probabilities for the active ensembles
            probs_for_active = np.array(active_ensemble_priorities_list) / total_active_ensemble_priority
            # Iterate over the active ensembles
            for k_idx, original_ensemble_idx in enumerate(self._active_ensemble_indices):
                # Set the probability for the active ensemble
                active_ensemble_probs[original_ensemble_idx] = probs_for_active[k_idx]

        return {
            "global_episode_count": self._episode_count,
            "active_ensemble_indices": list(self._active_ensemble_indices),
            "active_ensemble_names": active_ensemble_names,
            "active_ensemble_sample_probabilities": active_ensemble_probs.tolist(),  # Probabilities for ALL ensembles, zero if not active
            "current_ensemble_idx_selected": self._current_ensemble_idx,
            "current_env_idx_in_ensemble_selected": self._current_env_idx_in_ensemble,
            "config_min_samples_for_norm_stats": self._min_samples_for_norm_stats,
            "config_max_z_score_magnitude": self._max_z_score_magnitude,
            "ensembles_details": ensemble_stats,
        }
