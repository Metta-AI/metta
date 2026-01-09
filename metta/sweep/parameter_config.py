from __future__ import annotations

from enum import StrEnum
from typing import Any, List, Literal

from pydantic import Field, model_validator

from mettagrid.base_config import Config


class Distribution(StrEnum):
    """Supported parameter distributions."""

    UNIFORM = "uniform"
    INT_UNIFORM = "int_uniform"
    UNIFORM_POW2 = "uniform_pow2"
    LOG_NORMAL = "log_normal"
    LOGIT_NORMAL = "logit_normal"


class ParameterConfig(Config):
    """Configuration for a single hyperparameter to optimize.

    Performs internal validation/sanitization:
    - For "logit_normal", clamps bounds to (1e-6, 1 - 1e-6)
    - If "mean" is omitted, defaults to geometric mean for log/log2 and arithmetic mean otherwise
    - Ensures min < max
    """

    min: float = Field(description="Minimum value for the parameter")
    max: float = Field(description="Maximum value for the parameter")
    distribution: Literal["uniform", "int_uniform", "uniform_pow2", "log_normal", "logit_normal"] = Field(
        description="Distribution type for sampling"
    )
    mean: float = Field(description="Mean/center value for search")
    scale: float | str = Field(description="Scale for the parameter search")

    @model_validator(mode="before")
    @classmethod
    def _sanitize_and_default(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        v = dict(values)
        dist = v.get("distribution")

        # Clamp for logit-normal to avoid 0/1 boundary issues
        if dist == "logit_normal":
            eps = 1e-6
            try:
                v_min = float(v.get("min"))
                v_max = float(v.get("max"))
            except Exception:
                return v
            v_min = max(v_min, eps)
            v_max = min(v_max, 1 - eps)
            v["min"] = v_min
            v["max"] = v_max

        # Default mean if not provided
        if v.get("mean") is None:
            try:
                v_min = float(v.get("min"))
                v_max = float(v.get("max"))
            except Exception:
                return v
            if dist in ("log_normal", "uniform_pow2"):
                v["mean"] = (v_min * v_max) ** 0.5
            else:
                v["mean"] = (v_min + v_max) / 2.0

        # Basic bound validation
        try:
            if float(v.get("min")) >= float(v.get("max")):
                raise ValueError("min must be less than max")
        except Exception:
            return v

        return v


class CategoricalParameterConfig(Config):
    """Configuration for a categorical hyperparameter.

    Optimizer adapters may map this to their native categorical representation.
    For optimizers without native categorical support, adapters may encode
    categories via indices or one-hot schemes as appropriate.
    """

    choices: List[Any] = Field(description="List of allowed categorical values")


# Type alias for any supported parameter specification
ParameterSpec = ParameterConfig | CategoricalParameterConfig
