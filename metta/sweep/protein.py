import logging
import math
import random
import time
from copy import deepcopy

import numpy as np
import torch
from pyro.contrib import gp as gp

from mettagrid.util.dict_utils import unroll_nested_dict

logger = logging.getLogger(__name__)


class Space:
    def __init__(self, min, max, scale, mean, is_integer=False):
        self.min = min
        self.max = max
        self.scale = scale
        self.mean = mean  # TODO: awkward to have just this normalized
        self.norm_min = self.normalize(min)
        self.norm_max = self.normalize(max)
        self.norm_mean = self.normalize(mean)
        self.is_integer = is_integer

    def normalize(self, value):
        raise NotImplementedError

    def unnormalize(self, value):
        raise NotImplementedError


class Linear(Space):
    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        zero_one = (value - self.min) / (self.max - self.min)
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        value = zero_one * (self.max - self.min) + self.min
        if self.is_integer:
            value = round(value)
        return value


class Pow2(Space):
    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        zero_one = (math.log(value, 2) - math.log(self.min, 2)) / (math.log(self.max, 2) - math.log(self.min, 2))
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (math.log(self.max, 2) - math.log(self.min, 2)) + math.log(self.min, 2)
        rounded = round(log_spaced)
        return 2**rounded


class Log(Space):
    base: int = 10

    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == "time":
            scale = 1 / (np.log2(max) - np.log2(min))
        elif scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        zero_one = (math.log(value, self.base) - math.log(self.min, self.base)) / (
            math.log(self.max, self.base) - math.log(self.min, self.base)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (math.log(self.max, self.base) - math.log(self.min, self.base)) + math.log(
            self.min, self.base
        )
        value = self.base**log_spaced
        if self.is_integer:
            value = round(value)
        return value


class Logit(Space):
    base: int = 10

    def __init__(self, min, max, scale, mean, is_integer=False):
        if scale == "auto":
            scale = 0.5
        super().__init__(min, max, scale, mean, is_integer)

    def normalize(self, value):
        zero_one = (math.log(1 - value, self.base) - math.log(1 - self.min, self.base)) / (
            math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base)
        )
        return 2 * zero_one - 1

    def unnormalize(self, value):
        zero_one = (value + 1) / 2
        log_spaced = zero_one * (math.log(1 - self.max, self.base) - math.log(1 - self.min, self.base)) + math.log(
            1 - self.min, self.base
        )
        return 1 - self.base**log_spaced


def _params_from_puffer_sweep(sweep_config):
    param_spaces = {}
    for name, param in sweep_config.items():
        if name in ("method", "metric", "goal"):
            continue
        assert isinstance(param, dict)
        if any(isinstance(param[k], dict) for k in param):
            param_spaces[name] = _params_from_puffer_sweep(param)
            continue
        assert "distribution" in param
        distribution = param["distribution"]
        search_center = param["mean"]
        kwargs = dict(
            min=param["min"],
            max=param["max"],
            scale=param["scale"],
            mean=search_center,
        )
        if distribution == "uniform":
            space = Linear(**kwargs)
        elif distribution == "int_uniform":
            space = Linear(**kwargs, is_integer=True)
        elif distribution == "uniform_pow2":
            space = Pow2(**kwargs, is_integer=True)
        elif distribution == "log_normal":
            space = Log(**kwargs)
        elif distribution == "logit_normal":
            space = Logit(**kwargs)
        else:
            raise ValueError(f"Invalid distribution: {distribution}")
        param_spaces[name] = space
    return param_spaces


class Hyperparameters:
    def __init__(self, config, verbose=True):
        self.spaces = _params_from_puffer_sweep(config)
        self.flat_spaces = dict(unroll_nested_dict(self.spaces))
        self.num = len(self.flat_spaces)
        self.metric = config["metric"]
        goal = config["goal"]
        assert goal in ("maximize", "minimize")
        self.optimize_direction = 1 if goal == "maximize" else -1
        self.search_centers = np.array([e.norm_mean for e in self.flat_spaces.values()])
        self.min_bounds = np.array([e.norm_min for e in self.flat_spaces.values()])
        self.max_bounds = np.array([e.norm_max for e in self.flat_spaces.values()])
        self.search_scales = np.array([e.scale for e in self.flat_spaces.values()])

    def sample(self, n, mu=None, scale=1):
        if mu is None:
            mu = self.search_centers
        if len(mu.shape) == 1:
            mu = mu[None, :]
        n_input, n_dim = mu.shape
        scale = scale * self.search_scales
        mu_idxs = np.random.randint(0, n_input, n)
        samples = scale * (2 * np.random.rand(n, n_dim) - 1) + mu[mu_idxs]
        return np.clip(samples, self.min_bounds, self.max_bounds)

    def from_dict(self, params):
        flat_params = dict(unroll_nested_dict(params))
        values = []
        for key, space in self.flat_spaces.items():
            assert key in flat_params, f"Missing hyperparameter {key}"
            val = flat_params[key]
            normed = space.normalize(val)
            values.append(normed)
        return np.array(values)

    def to_dict(self, sample, fill=None):
        params = deepcopy(self.spaces) if fill is None else fill
        self._fill(params, self.spaces, sample)
        return params

    def _fill(self, params, spaces, flat_sample, idx=0):
        for name, space in spaces.items():
            if isinstance(space, dict):
                # Create nested dict if it doesn't exist
                if name not in params:
                    params[name] = {}
                idx = self._fill(params[name], spaces[name], flat_sample, idx=idx)
            else:
                params[name] = spaces[name].unnormalize(flat_sample[idx])
                idx += 1
        return idx


def pareto_points(observations, eps=1e-6):
    # Backwards-compatible wrapper defaults to maximize
    return pareto_points_oriented(observations, direction=1, eps=eps)


def pareto_points_oriented(observations, direction=1, eps=1e-6):
    """Compute Pareto front on (score, cost) with goal encoded by `direction`.
    direction = +1 for maximize, -1 for minimize."""
    scores = np.array([direction * e["output"] for e in observations])
    costs = np.array([e["cost"] for e in observations])
    pareto, idxs = [], []
    for idx in range(len(observations)):
        higher_score = scores + eps > scores[idx]
        lower_cost = costs - eps < costs[idx]
        better = higher_score & lower_cost
        better[idx] = False
        if not better.any():
            pareto.append(observations[idx])
            idxs.append(idx)
    return pareto, idxs


def create_gp(x_dim, scale_length=1.0):
    X = torch.zeros((1, x_dim))
    y = torch.zeros((1,))
    matern_kernel = gp.kernels.Matern32(input_dim=x_dim, lengthscale=scale_length * torch.ones(x_dim))
    linear_kernel = gp.kernels.Polynomial(input_dim=x_dim, degree=1)
    kernel = gp.kernels.Sum(linear_kernel, matern_kernel)
    model = gp.models.GPRegression(X, y, kernel=kernel, jitter=1.0e-3)  # Increased jitter for stability
    # Keep noise as a positive tensor (simpler & numerically stable)
    model.noise = torch.tensor(1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer


class Protein:
    def __init__(
        self,
        sweep_config,
        max_suggestion_cost=3600,
        resample_frequency=0,
        num_random_samples=50,
        global_search_scale=1,
        random_suggestions=1024,
        suggestions_per_pareto=256,
        seed_with_search_center=True,
        expansion_rate=0.25,
        randomize_acquisition=False,
    ):
        self.hyperparameters = Hyperparameters(sweep_config)
        self.num_random_samples = num_random_samples
        self.global_search_scale = global_search_scale
        self.random_suggestions = random_suggestions
        self.suggestions_per_pareto = suggestions_per_pareto
        self.seed_with_search_center = seed_with_search_center
        self.resample_frequency = resample_frequency
        self.max_suggestion_cost = max_suggestion_cost
        self.expansion_rate = expansion_rate
        self.randomize_acquisition = randomize_acquisition
        self.success_observations = []
        self.failure_observations = []
        self.suggestion_idx = 0
        self.gp_score, self.score_opt = create_gp(self.hyperparameters.num)
        self.gp_cost, self.cost_opt = create_gp(self.hyperparameters.num)

    def suggest(self, n_suggestions=1, fill=None):
        info = {}
        self.suggestion_idx += 1

        # Set random seed for diversity in parallel runs
        if self.randomize_acquisition:
            # Use current time plus suggestion index for unique seed
            seed = int((time.time() * 1000000 + self.suggestion_idx) % 2**32)
            np.random.seed(seed)
            random.seed(seed)
            info["random_seed"] = seed

        if len(self.success_observations) == 0 and self.seed_with_search_center:
            best = self.hyperparameters.search_centers
            best_dict = self.hyperparameters.to_dict(best, fill)

            if n_suggestions == 1:
                return best_dict, info
            else:
                # Generate additional random suggestions
                results = [(best_dict, info)]
                additional_suggestions = self.hyperparameters.sample(n_suggestions - 1, scale=self.global_search_scale)
                for i in range(n_suggestions - 1):
                    suggestion_dict = self.hyperparameters.to_dict(additional_suggestions[i], fill)
                    results.append((suggestion_dict, info.copy()))
                return results

        elif len(self.success_observations) < self.num_random_samples:
            # Generate enough random suggestions
            suggestions = self.hyperparameters.sample(
                max(n_suggestions, self.random_suggestions), scale=self.global_search_scale
            )

            if n_suggestions == 1:
                self.suggestion = random.choice(suggestions[: self.random_suggestions])
                return self.hyperparameters.to_dict(self.suggestion, fill), info
            else:
                # Return multiple random suggestions
                results = []
                selected_indices = random.sample(range(len(suggestions)), min(n_suggestions, len(suggestions)))
                for idx in selected_indices:
                    suggestion_dict = self.hyperparameters.to_dict(suggestions[idx], fill)
                    results.append((suggestion_dict, info.copy()))
                return results

        elif self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points_oriented(self.success_observations, self.hyperparameters.optimize_direction)
            pareto_suggestions = np.stack([e["input"] for e in candidates])

            if n_suggestions == 1:
                best_idx = np.random.randint(0, len(candidates))
                best = pareto_suggestions[best_idx]
                return self.hyperparameters.to_dict(best, fill), info
            else:
                # Sample from Pareto points and generate variations
                results = []
                for i in range(n_suggestions):
                    if i < len(pareto_suggestions):
                        # Use actual Pareto points first
                        idx = i % len(pareto_suggestions)
                        suggestion_dict = self.hyperparameters.to_dict(pareto_suggestions[idx], fill)
                    else:
                        # Generate variations around Pareto points
                        center_idx = np.random.randint(0, len(pareto_suggestions))
                        variation = self.hyperparameters.sample(
                            1, mu=pareto_suggestions[center_idx], scale=self.global_search_scale
                        )[0]
                        suggestion_dict = self.hyperparameters.to_dict(variation, fill)
                    results.append((suggestion_dict, info.copy()))
                return results

        # Rest of the method remains the same (GP-based path)...

        # === Train score GP on standardized outputs with progressive fallback ===
        """
        Progressive Fallback Strategy for Robust GP Training

        Gaussian Process training can fail due to numerical instabilities, particularly
        when the kernel matrix becomes ill-conditioned (near-singular). This commonly
        happens when:
        - Observations are very similar (duplicate or near-duplicate parameters)
        - Observation noise is very low
        - The kernel lengthscales are inappropriate for the data scale

        Our progressive fallback strategy ensures the optimizer remains functional even
        with problematic data by progressively reducing the dataset size until training
        succeeds.
        """
        params = np.array([e["input"] for e in self.success_observations])
        y = np.array([e["output"] for e in self.success_observations])

        # Progressive fallback: try with N, N/2, N/4, ... observations
        # This binary reduction ensures we quickly find a trainable subset while
        # keeping the most informative observations (best performers)
        n_obs = len(self.success_observations)
        subset_size = n_obs
        subset_indices = list(range(n_obs))  # Initialize with all indices
        gp_trained = False

        while subset_size >= 10 and not gp_trained:
            try:
                # Select subset of observations
                # When reducing dataset size, keep the best performers to maintain
                # information about promising regions of the parameter space
                if subset_size == n_obs:
                    # First attempt: use all observations
                    subset_indices = list(range(n_obs))
                else:
                    # Subsequent attempts: keep only the best performers
                    # This biases the GP toward successful regions but ensures stability
                    if self.hyperparameters.optimize_direction == 1:
                        # Maximization: keep observations with highest scores
                        subset_indices = np.argsort(y)[-subset_size:]
                    else:
                        # Minimization: keep observations with lowest scores
                        subset_indices = np.argsort(y)[:subset_size]

                # Get subset data
                params_subset = params[subset_indices]
                y_subset = y[subset_indices]

                # Standardize outputs for numerical stability
                # Z-score normalization ensures the GP operates on a consistent scale
                # regardless of the actual score magnitudes
                y_mean = float(np.mean(y_subset))
                y_std = float(np.std(y_subset) + 1e-12)  # Add epsilon to prevent division by zero
                y_z = (y_subset - y_mean) / y_std

                # Convert to tensors
                params_t = torch.from_numpy(params_subset).float()

                # Try to train GP
                # This involves:
                # 1. Setting the training data
                # 2. Optimizing kernel hyperparameters via marginal likelihood
                # 3. Computing the Cholesky decomposition of the kernel matrix
                self.gp_score.set_data(params_t, torch.from_numpy(y_z).float())
                self.gp_score.train()
                gp.util.train(self.gp_score, self.score_opt)  # MLL optimization
                self.gp_score.eval()

                gp_trained = True
                if subset_size < n_obs:
                    logger.debug(f"GP trained with {subset_size}/{n_obs} best observations")

            except (torch._C._LinAlgError, RuntimeError):
                # Cholesky decomposition failed - kernel matrix is ill-conditioned
                # This is the most common failure mode for GPs
                # Reduce dataset size by half and retry
                subset_size = subset_size // 2
                if subset_size >= 10:
                    logger.debug(f"GP failed with {subset_size * 2} observations, trying {subset_size}...")
                continue

        # If GP training failed completely, fall back to random sampling
        # This ensures the optimizer never crashes, though it loses the benefits
        # of Bayesian optimization for this iteration
        if not gp_trained:
            logger.debug("GP training failed, falling back to random sampling")
            if n_suggestions == 1:
                suggestion = self.hyperparameters.sample(n=1)[0]
                return self.hyperparameters.to_dict(suggestion, fill), {"fallback": "random"}
            else:
                results = []
                suggestions = self.hyperparameters.sample(n_suggestions, scale=self.global_search_scale)
                for i in range(n_suggestions):
                    suggestion_dict = self.hyperparameters.to_dict(suggestions[i], fill)
                    results.append((suggestion_dict, {"fallback": "random"}))
                return results

        # === Build candidate suggestions from oriented Pareto centers ===
        candidates, pareto_idxs = pareto_points_oriented(
            self.success_observations, self.hyperparameters.optimize_direction
        )
        if len(candidates) == 0:
            # All observations were mutually dominated under strict epsilon; pick the best score as anchor.
            logger.warning("[Protein] No Pareto candidates found; using best observed score as center.")
            direction = self.hyperparameters.optimize_direction
            scores = np.array([direction * e["output"] for e in self.success_observations])
            best_idx = int(np.argmax(scores)) if len(scores) else 0
            candidates = [self.success_observations[best_idx]]
        search_centers = np.stack([e["input"] for e in candidates])
        suggestions = self.hyperparameters.sample(
            len(candidates) * self.suggestions_per_pareto, mu=search_centers, scale=self.global_search_scale
        )
        suggestions_t = torch.from_numpy(suggestions).float()

        # Predict standardized mean/var
        with torch.no_grad():
            mu_t, var_t = self.gp_score(suggestions_t, full_cov=False)
        mu = mu_t.numpy()
        # Note: var_t is returned but not used directly for scoring

        # For 'naive' normalization path (info/weighting)
        # Use subset statistics if subset was used
        if subset_size < n_obs:
            y_for_norm = y[subset_indices]
        else:
            y_for_norm = y
        min_y, max_y = np.min(y_for_norm), np.max(y_for_norm)
        mu_raw = mu * y_std + y_mean
        gp_y_norm = (mu_raw - min_y) / (np.abs(max_y - min_y) + 1e-12)

        # === Cost handling: skip GP if constant cost ===
        c = np.array([e["cost"] for e in self.success_observations])

        # Use same subset for cost GP as score GP (if subset was used)
        if subset_size < n_obs:
            c_subset = c[subset_indices]
        else:
            c_subset = c

        if np.max(c_subset) - np.min(c_subset) < 1e-12:
            gp_log_c_norm = np.full(len(suggestions), 0.5)
            gp_c = np.full(len(suggestions), c_subset[0])
        else:
            EPS = 1e-12
            log_c = np.log(np.maximum(c_subset, EPS))
            lc_min, lc_max = np.min(log_c), np.max(log_c)
            lc_norm = (log_c - lc_min) / (lc_max - lc_min + 1e-12)

            # params_t already points to the subset from score GP training
            self.gp_cost.set_data(params_t, torch.from_numpy(lc_norm).float())
            self.gp_cost.train()
            gp.util.train(self.gp_cost, self.cost_opt)
            self.gp_cost.eval()
            with torch.no_grad():
                gp_log_c_norm_t, _ = self.gp_cost(suggestions_t, full_cov=False)
            gp_log_c_norm = gp_log_c_norm_t.numpy()
            gp_log_c = gp_log_c_norm * (lc_max - lc_min) + lc_min
            gp_c = np.exp(gp_log_c)

        max_c_mask = gp_c < self.max_suggestion_cost
        cost_threshold_relaxed = False
        if not max_c_mask.any():
            max_c_mask = np.ones_like(max_c_mask, dtype=bool)
            cost_threshold_relaxed = True

        # Compute naive acquisition function
        # The naive approach uses a weighted combination of normalized score and cost
        target = (1 + self.expansion_rate) * np.random.rand()
        weight = np.maximum(1 - np.abs(target - gp_log_c_norm), 0.0)
        suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (gp_y_norm * weight)

        # Get top-k suggestions if requested
        if n_suggestions == 1:
            best_idx = np.argmax(suggestion_scores)
            # Map standardized mu/sd back to raw for info (not used in selection)
            info = dict(
                cost=float(gp_c[best_idx]),
                score=float(mu_raw[best_idx]),
                rating=float(suggestion_scores[best_idx]),
                randomize_acquisition=self.randomize_acquisition,
            )
            if cost_threshold_relaxed:
                info["cost_threshold_relaxed"] = True
            best = suggestions_t[best_idx].numpy()
            return self.hyperparameters.to_dict(best, fill), info
        else:
            # Return top-k suggestions
            k = min(n_suggestions, len(suggestion_scores))
            top_k_indices = np.argpartition(suggestion_scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(suggestion_scores[top_k_indices])][::-1]  # Sort by score

            results = []
            for idx in top_k_indices:
                info = dict(
                    cost=float(gp_c[idx]),
                    score=float(mu_raw[idx]),
                    rating=float(suggestion_scores[idx]),
                    randomize_acquisition=self.randomize_acquisition,
                )
                if cost_threshold_relaxed:
                    info["cost_threshold_relaxed"] = True

                suggestion = suggestions_t[idx].numpy()
                results.append((self.hyperparameters.to_dict(suggestion, fill), info))

            return results

    def observe(self, hypers, score, cost, is_failure=False):
        params = self.hyperparameters.from_dict(hypers)
        new_observation = dict(
            input=params,
            output=score,
            cost=cost,
            is_failure=is_failure,
        )
        if len(self.success_observations) == 0:
            self.success_observations.append(new_observation)
            return
        success_params = np.stack([e["input"] for e in self.success_observations])
        dist = np.linalg.norm(params - success_params, axis=1)
        same = np.where(dist < 1e-6)[0]
        if len(same) > 0:
            self.success_observations[same[0]] = new_observation
        else:
            self.success_observations.append(new_observation)


## Note: categorical parameters are intentionally not supported natively by
## Protein's GP space. Mapping from categories to numeric indices, and back,
## is handled in the optimizer adapter layer.
