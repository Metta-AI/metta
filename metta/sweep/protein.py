import math
import random
from copy import deepcopy

import numpy as np
import pufferlib
import pyro
import torch
from pyro.contrib import gp as gp


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
        self.flat_spaces = dict(pufferlib.unroll_nested_dict(self.spaces))
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
        flat_params = dict(pufferlib.unroll_nested_dict(params))
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
    scores = np.array([e["output"] for e in observations])
    costs = np.array([e["cost"] for e in observations])
    pareto = []
    idxs = []
    for idx, obs in enumerate(observations):
        try:
            higher_score = scores + eps > scores[idx]
        except Exception as e:
            raise RuntimeError(f"Failed to compare protein scores: {e}") from e
        lower_cost = costs - eps < costs[idx]
        better = higher_score & lower_cost
        better[idx] = False
        if not better.any():
            pareto.append(obs)
            idxs.append(idx)
    return pareto, idxs


class Random:
    def __init__(
        self,
        sweep_config,
        global_search_scale=1,
        random_suggestions=1024,
    ):
        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = global_search_scale
        self.random_suggestions = random_suggestions
        self.success_observations = []

    def suggest(self, fill=None):
        suggestions = self.hyperparameters.sample(self.random_suggestions)
        self.suggestion = random.choice(suggestions)
        return self.hyperparameters.to_dict(self.suggestion, fill), {}

    def observe(self, hypers, score, cost, is_failure=False):
        self.success_observations.append(
            dict(
                input=hypers,
                output=score,
                cost=cost,
                is_failure=is_failure,
            )
        )


class ParetoGenetic:
    def __init__(
        self,
        sweep_config,
        global_search_scale=1,
        suggestions_per_pareto=1,
        bias_cost=True,
        log_bias=False,
    ):
        self.hyperparameters = Hyperparameters(sweep_config)
        self.global_search_scale = global_search_scale
        self.suggestions_per_pareto = suggestions_per_pareto
        self.bias_cost = bias_cost
        self.log_bias = log_bias
        self.success_observations = []

    def suggest(self, fill=None):
        if len(self.success_observations) == 0:
            suggestion = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(suggestion, fill), {}
        candidates, _ = pareto_points(self.success_observations)
        pareto_costs = np.array([e["cost"] for e in candidates])
        if self.bias_cost:
            if self.log_bias:
                cost_dists = np.abs(np.log(pareto_costs[:, None]) - np.log(pareto_costs[None, :]))
            else:
                cost_dists = np.abs(pareto_costs[:, None] - pareto_costs[None, :])
            cost_dists += (np.max(pareto_costs) + 1) * np.eye(len(pareto_costs))  # mask self-distance
            idx = np.argmax(np.min(cost_dists, axis=1))
            search_centers = candidates[idx]["input"]
        else:
            search_centers = np.stack([e["input"] for e in candidates])
        suggestions = self.hyperparameters.sample(len(candidates) * self.suggestions_per_pareto, mu=search_centers)
        suggestion = suggestions[np.random.randint(0, len(suggestions))]
        return self.hyperparameters.to_dict(suggestion, fill), {}

    def observe(self, hypers, score, cost, is_failure=False):
        params = self.hyperparameters.from_dict(hypers)
        self.success_observations.append(
            dict(
                input=params,
                output=score,
                cost=cost,
                is_failure=is_failure,
            )
        )


def create_gp(x_dim, scale_length=1.0):
    X = scale_length * torch.ones((1, x_dim))
    y = torch.zeros((1,))
    matern_kernel = gp.kernels.Matern32(input_dim=x_dim, lengthscale=X)
    linear_kernel = gp.kernels.Polynomial(x_dim, degree=1)
    kernel = gp.kernels.Sum(linear_kernel, matern_kernel)
    model = gp.models.GPRegression(X, y, kernel=kernel, jitter=1.0e-4)
    model.noise = pyro.nn.PyroSample(pyro.distributions.LogNormal(math.log(1e-2), 0.5))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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
        self.success_observations = []
        self.failure_observations = []
        self.suggestion_idx = 0
        self.gp_score, self.score_opt = create_gp(self.hyperparameters.num)
        self.gp_cost, self.cost_opt = create_gp(self.hyperparameters.num)

    def suggest(self, fill):
        info = {}
        self.suggestion_idx += 1
        if len(self.success_observations) == 0 and self.seed_with_search_center:
            best = self.hyperparameters.search_centers
            return self.hyperparameters.to_dict(best, fill), info
        elif not self.seed_with_search_center and len(self.success_observations) < self.num_random_samples:
            suggestions = self.hyperparameters.sample(self.random_suggestions)
            self.suggestion = random.choice(suggestions)
            return self.hyperparameters.to_dict(self.suggestion, fill), info
        elif self.resample_frequency and self.suggestion_idx % self.resample_frequency == 0:
            candidates, _ = pareto_points(self.success_observations)
            suggestions = np.stack([e["input"] for e in candidates])
            best_idx = np.random.randint(0, len(candidates))
            best = suggestions[best_idx]
            return self.hyperparameters.to_dict(best, fill), info
        params = np.array([e["input"] for e in self.success_observations])
        params = torch.from_numpy(params)
        y = np.array([e["output"] for e in self.success_observations])
        min_score = np.min(y)
        max_score = np.max(y)
        y_norm = (y - min_score) / (np.abs(max_score - min_score) + 1e-6)
        self.gp_score.set_data(params, torch.from_numpy(y_norm))
        self.gp_score.train()
        gp.util.train(self.gp_score, self.score_opt)
        self.gp_score.eval()
        c = np.array([e["cost"] for e in self.success_observations])
        log_c = np.log(c)
        log_c_min = np.min(log_c)
        log_c_max = np.max(log_c)
        log_c_norm = (log_c - log_c_min) / (log_c_max - log_c_min + 1e-6)
        self.gp_cost.mean_function = lambda x: 1
        self.gp_cost.set_data(params, torch.from_numpy(log_c_norm))
        self.gp_cost.train()
        gp.util.train(self.gp_cost, self.cost_opt)
        self.gp_cost.eval()
        candidates, pareto_idxs = pareto_points(self.success_observations)
        search_centers = np.stack([e["input"] for e in candidates])
        suggestions = self.hyperparameters.sample(len(candidates) * self.suggestions_per_pareto, mu=search_centers)
        suggestions = torch.from_numpy(suggestions)
        with torch.no_grad():
            gp_y_norm, gp_y_norm_var = self.gp_score(suggestions)
            gp_log_c_norm, _ = self.gp_cost(suggestions)
        gp_y_norm = gp_y_norm.numpy()
        gp_log_c_norm = gp_log_c_norm.numpy()
        gp_y = gp_y_norm * (max_score - min_score) + min_score
        gp_log_c = gp_log_c_norm * (log_c_max - log_c_min) + log_c_min
        gp_c = np.exp(gp_log_c)

        max_c_mask = gp_c < self.max_suggestion_cost
        target = (1 + self.expansion_rate) * np.random.rand()
        weight = 1 - abs(target - gp_log_c_norm)
        suggestion_scores = self.hyperparameters.optimize_direction * max_c_mask * (gp_y_norm * weight)
        best_idx = np.argmax(suggestion_scores)
        info = dict(
            cost=gp_c[best_idx].item(),
            score=gp_y[best_idx].item(),
            rating=suggestion_scores[best_idx].item(),
        )
        best = suggestions[best_idx].numpy()
        return self.hyperparameters.to_dict(best, fill), info

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
