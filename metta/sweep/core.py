"""Common sweep parameter helpers for Ray-based sweeps."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, List, Literal, Optional, Union

from pydantic import Field, model_validator

from mettagrid.config import Config

from mettagrid.base_config import Config

logger = logging.getLogger(__name__)

_TUNE = None


def _require_tune():
    global _TUNE
    if _TUNE is None:
        try:
            from ray import tune  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "Ray Tune is required for building sweep search spaces. Install the 'ray[tune]' "
                "extra or ensure the runtime environment provides Ray."
            ) from exc
        _TUNE = tune
    return _TUNE


@dataclass(frozen=True)
class ParameterSpec:
    """A single sweep parameter: dotted path + Ray Tune domain."""

    path: str
    space: Any


class Distribution(StrEnum):
    """Supported numeric distributions for convenience helpers."""

    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    UNIFORM_POW2 = "uniform_pow2"
    LOG_NORMAL = "log_normal"
    LOGIT_NORMAL = "logit_normal"


class SweepParameters:
    """Convenience presets used by recipes."""

    @staticmethod
    def param(
        name: str,
        distribution: Distribution,
        min: float,
        max: float,
        search_center: float | None = None,  # kept for API parity
        scale: str = "auto",  # retained for compatibility (unused by Ray)
    ) -> ParameterSpec:
        tune = _require_tune()
        dist = distribution.value

        if dist == Distribution.UNIFORM.value:
            space = tune.uniform(min, max)
        elif dist == Distribution.INT_UNIFORM.value:
            lo = int(math.floor(min))
            hi = int(math.ceil(max))
            if hi < lo:
                raise ValueError(f"int_uniform requires min <= max, got {min} > {max}")
            space = tune.randint(lo, hi + 1)
        elif dist == Distribution.UNIFORM_POW2.value:
            try:
                min_power = math.ceil(math.log(min, 2))
                max_power = math.floor(math.log(max, 2))
            except ValueError as exc:
                raise ValueError(f"uniform_pow2 requires positive bounds, got min={min}, max={max}") from exc
            if max_power < min_power:
                raise ValueError(f"No powers of two between {min} and {max}")
            values = [2**exp for exp in range(min_power, max_power + 1)]
            space = tune.choice(values)
        elif dist == Distribution.LOG_NORMAL.value:
            if min <= 0 or max <= 0:
                raise ValueError(f"log_normal requires positive bounds, got min={min}, max={max}")
            space = tune.loguniform(min, max)
        elif dist == Distribution.LOGIT_NORMAL.value:
            if not (0 < min < max < 1):
                raise ValueError(f"logit_normal requires bounds within (0, 1); received min={min}, max={max}")
            base = 10

            def _sample(_: Any) -> float:
                z = random.uniform(-1.0, 1.0)
                zero_one = (z + 1.0) / 2.0
                span = math.log(1 - max, base) - math.log(1 - min, base)
                offset = math.log(1 - min, base)
                log_val = zero_one * span + offset
                return 1 - base**log_val

            space = tune.sample_from(_sample)
        else:
            raise ValueError(f"Unsupported distribution '{distribution}'")

        return ParameterSpec(path=name, space=space)

    @staticmethod
    def categorical(name: str, choices: Iterable[Any]) -> ParameterSpec:
        choices_list = list(choices)
        if not choices_list:
            raise ValueError("Categorical choices must be a non-empty iterable")
        tune = _require_tune()
        return ParameterSpec(path=name, space=tune.choice(choices_list))

    LEARNING_RATE = ParameterSpec(
        path="trainer.optimizer.learning_rate",
        space=_require_tune().loguniform(1e-5, 1e-2),
    )

    PPO_CLIP_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.clip_coef",
        space=_require_tune().uniform(0.05, 0.3),
    )

    PPO_ENT_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.ent_coef",
        space=_require_tune().loguniform(1e-4, 3e-2),
    )

    PPO_GAE_LAMBDA = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.gae_lambda",
        space=_require_tune().uniform(0.8, 0.99),
    )

    PPO_VF_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.vf_coef",
        space=_require_tune().uniform(0.1, 1.0),
    )

    ADAM_EPS = ParameterSpec(
        path="trainer.optimizer.eps",
        space=_require_tune().loguniform(1e-8, 1e-4),
    )


class ParameterConfig(Config):
    """Legacy parameter declaration used by Protein-based sweep tooling."""

    path: Optional[str] = Field(default=None, description="Optional dotted path for the parameter")
    min: float = Field(description="Minimum value for the parameter")
    max: float = Field(description="Maximum value for the parameter")
    distribution: Literal["uniform", "int_uniform", "uniform_pow2", "log_normal", "logit_normal"]
    mean: float = Field(description="Mean/centre value for the search")
    scale: float | str = Field(description="Scale for the parameter search")

    @model_validator(mode="before")
    @classmethod
    def _sanitize_and_default(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        dist = data.get("distribution")

        if dist == "logit_normal":
            eps = 1e-6
            try:
                lo = float(data.get("min"))
                hi = float(data.get("max"))
            except Exception:
                return data
            lo = max(lo, eps)
            hi = min(hi, 1 - eps)
            data["min"] = lo
            data["max"] = hi

        if data.get("mean") is None:
            try:
                lo = float(data.get("min"))
                hi = float(data.get("max"))
            except Exception:
                return data
            if dist in ("log_normal", "uniform_pow2"):
                data["mean"] = (lo * hi) ** 0.5
            else:
                data["mean"] = (lo + hi) / 2.0

        try:
            if float(data.get("min")) >= float(data.get("max")):
                raise ValueError("min must be less than max")
        except Exception:
            return data

        return data

    def to_tune_domain(self) -> Any:
        tune = _require_tune()
        dist = self.distribution

        if dist == "uniform":
            return tune.uniform(self.min, self.max)

        if dist == "int_uniform":
            lo = int(math.floor(self.min))
            hi = int(math.ceil(self.max))
            if hi < lo:
                raise ValueError(f"int_uniform requires min <= max, got {self.min} > {self.max}")
            return tune.randint(lo, hi + 1)

        if dist == "uniform_pow2":
            try:
                min_power = math.ceil(math.log(self.min, 2))
                max_power = math.floor(math.log(self.max, 2))
            except ValueError as exc:
                raise ValueError(f"uniform_pow2 requires positive bounds, got min={self.min}, max={self.max}") from exc
            if max_power < min_power:
                raise ValueError(f"No powers of two between {self.min} and {self.max}")
            values = [2**exp for exp in range(min_power, max_power + 1)]
            return tune.choice(values)

        if dist == "log_normal":
            if self.min <= 0 or self.max <= 0:
                raise ValueError(f"log_normal requires positive bounds, got min={self.min}, max={self.max}")
            return tune.loguniform(self.min, self.max)

        if dist == "logit_normal":
            if not (0 < self.min < self.max < 1):
                raise ValueError(
                    f"logit_normal requires bounds within (0, 1); received min={self.min}, max={self.max}"
                )
            base = 10

            def _sample(_: Any) -> float:
                z = random.uniform(-1.0, 1.0)
                zero_one = (z + 1.0) / 2.0
                span = math.log(1 - self.max, base) - math.log(1 - self.min, base)
                offset = math.log(1 - self.min, base)
                log_val = zero_one * span + offset
                return 1 - base**log_val

            return tune.sample_from(_sample)

        raise ValueError(f"Unsupported distribution '{dist}'")


class CategoricalParameterConfig(Config):
    """Legacy categorical parameter declaration used by Protein sweeps."""

    choices: List[Any] = Field(description="List of allowed categorical values")



__all__ = [
    "ParameterSpec",
    "Distribution",
    "SweepParameters",
    "ParameterConfig",
    "CategoricalParameterConfig",
]
