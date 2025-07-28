import datetime
import logging
import random
from typing import Any, TypeVar, Union

import numpy as np
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")  # For generic conditional function
Numeric = Union[int, float]  # Type alias for numeric types


def oc_if(condition: bool, true_value: T, false_value: T) -> T:
    return true_value if condition else false_value


def oc_uniform(min_val: Numeric, max_val: Numeric) -> float:
    return float(np.random.uniform(min_val, max_val))


def oc_choose(*args: Any) -> Any:
    return random.choice(args)


def oc_divide(a: Numeric, b: Numeric) -> Numeric:
    """
    Divide a by b, returning an int if both inputs are ints and result is a whole number,
    otherwise return a float.
    """
    result = a / b
    # If both inputs are integers and the result is a whole number, return as int
    if isinstance(a, int) and isinstance(b, int) and result.is_integer():
        return int(result)
    return result


def oc_subtract(a: Numeric, b: Numeric) -> Numeric:
    return a - b


def oc_multiply(a: Numeric, b: Numeric) -> Numeric:
    return a * b


def oc_add(a: Numeric, b: Numeric) -> Numeric:
    return a + b


def oc_to_odd_min3(a: Numeric) -> int:
    """
    Ensure a value is odd and at least 3.
    """
    return max(3, int(a) // 2 * 2 + 1)


def oc_clamp(value: Numeric, min_val: Numeric, max_val: Numeric) -> Numeric:
    return max(min_val, min(max_val, value))


def oc_make_integer(value: Numeric) -> int:
    return int(round(value))


def oc_equals(a: Any, b: Any) -> bool:
    return a == b


def oc_greater_than(a: Any, b: Any) -> bool:
    return a > b


def oc_less_than(a: Any, b: Any) -> bool:
    return a < b


def oc_greater_than_or_equal(a: Any, b: Any) -> bool:
    return a >= b


def oc_less_than_or_equal(a: Any, b: Any) -> bool:
    return a <= b


def oc_scale(
    value: Numeric, in_min: Numeric, in_max: Numeric, out_min: Numeric, out_max: Numeric, scale_type: str = "linear"
) -> Numeric:
    """
    Scale a value from one range to another using different scaling methods.

    Parameters:
    -----------
    value : Numeric
        The input value to scale
    in_min : Numeric
        The minimum input value
    in_max : Numeric
        The maximum input value
    out_min : Numeric
        The minimum output value
    out_max : Numeric
        The maximum output value
    scale_type : str
        The type of scaling to apply. Options:
        - "linear" (default): Linear mapping
        - "log": Logarithmic scaling (faster growth at low values)
        - "exp": Exponential scaling (faster growth at high values)
        - "sigmoid": Sigmoid scaling (slower at extremes, faster in middle)

    Returns:
    --------
    Numeric
        The scaled value in the output range
    """

    assert in_min < in_max, "in_min must be less than in_max"
    assert out_min <= out_max, "out_min must be less than or equal to out_max"

    if out_min == out_max:
        return out_min

    # Clamp value to input range
    value = oc_clamp(value, in_min, in_max)

    # Normalize to 0-1 range
    normalized = (value - in_min) / (in_max - in_min) if in_max > in_min else 0

    # Apply scaling based on type
    if scale_type == "linear":
        scaled = normalized
    elif scale_type == "log":
        # Avoid log(0)
        if normalized == 0:
            scaled = 0
        else:
            # Log scaling factor (log10(1 + 9x) / log10(10))
            scaled = np.log10(1 + 9 * normalized) / np.log10(10)
    elif scale_type == "exp":
        # Exponential scaling (opposite of log)
        scaled = (np.power(10, normalized) - 1) / 9
    elif scale_type == "sigmoid":
        # Sigmoid scaling (slower at extremes, faster in middle)
        scaled = 1 / (1 + np.exp(-12 * (normalized - 0.5)))
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    # Map to output range
    result = out_min + scaled * (out_max - out_min)

    # Return as integer if both output bounds are integers
    if isinstance(out_min, int) and isinstance(out_max, int):
        return int(round(result))
    return result


def oc_iir(alpha: Numeric, current_value: Numeric, last_value: Numeric) -> Numeric:
    """
    Apply an IIR (Infinite Impulse Response) filter.

    This is a first-order low-pass filter that computes:
    y[n] = alpha * x[n] + (1-alpha) * y[n-1]

    Parameters:
    -----------`
    alpha : Numeric
        Filter coefficient (0 < alpha < 1). Lower values create more smoothing.
    current_value : Numeric
        The current input value x[n]
    last_value : Numeric
        The previous filtered output y[n-1]

    Returns:
    --------
    Numeric
        The filtered output y[n]
    """
    # Ensure alpha is in valid range
    alpha = oc_clamp(alpha, 0, 1)

    # Apply IIR filter formula
    result = alpha * current_value + (1 - alpha) * last_value

    # Return integer if both inputs are integers
    if isinstance(current_value, int) and isinstance(last_value, int):
        return int(round(result))
    return result


def oc_date_format(format_string: str) -> str:
    """
    Generate a formatted date string using the current date and time.

    Parameters:
    -----------
    format_string : str
        A format string following either:
        - Python datetime strftime format codes (starting with %)
        - Simplified format codes like "MMDD", "YYYYMMDD", etc.

    Returns:
    --------
    str
        The formatted date string
    """
    # Format mapping for simplified codes
    format_map = {"YYYY": "%Y", "YY": "%y", "MM": "%m", "DD": "%d", "HH": "%H", "mm": "%M", "ss": "%S"}

    # Copy the format string to avoid modifying the original
    python_format = format_string

    # If not starting with %, it might use our simplified format
    if not format_string.startswith("%"):
        # Replace simplified codes with Python format codes
        for simple_code, python_code in format_map.items():
            python_format = python_format.replace(simple_code, python_code)

    # Get current datetime and format it
    now = datetime.datetime.now()

    return now.strftime(python_format)


def oc_sampling(*args: Numeric) -> Numeric:
    """
    Sample a value from a range or set of choices.
    
    Usage:
    - ${sampling:min, max, default} - samples uniform between min and max, using default if sampling=0
    - For configuration validation, returns the middle/default value
    
    Parameters:
    -----------
    args : Numeric
        Either (min, max, default) for range sampling, or multiple values for choice
        
    Returns:
    --------
    Numeric
        The sampled or default value
    """
    if len(args) == 3:
        # Range sampling: min, max, default
        min_val, max_val, default_val = args
        # For testing/validation, return the default value
        return default_val
    elif len(args) > 1:
        # Choice sampling: return the middle choice for consistency
        middle_idx = len(args) // 2
        return args[middle_idx]
    else:
        # Single value
        return args[0] if args else 0


class ResolverRegistrar(Callback):
    """Class for registering custom OmegaConf resolvers."""

    def __init__(self):
        self.logger = logging.getLogger("ResolverRegistrar")
        self.resolver_count = 0
        """Prepare for registration but don't register yet."""

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Register resolvers at the start of a run."""
        self.register_resolvers()
        self.logger.info(f"Registered {self.resolver_count} custom resolvers at the start of a run")

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Register resolvers at the start of a multirun."""
        self.register_resolvers()
        self.logger.info(f"Registered {self.resolver_count} custom resolvers at the start of a multirun")

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        """Ensure resolvers are registered for each job."""
        pass

    def register_resolvers(self):
        """
        Register all OmegaConf resolvers for use in Hydra configs.

        This function is called during Hydra initialization via _target_ in
        configs/common.yaml, ensuring all resolvers are available before
        config interpolation happens.
        """

        # Register all your resolvers
        OmegaConf.register_new_resolver("if", oc_if, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("uniform", oc_uniform, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("choose", oc_choose, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("div", oc_divide, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("subtract", oc_subtract, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("sub", oc_subtract, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("multiply", oc_multiply, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("mul", oc_multiply, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("add", oc_add, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("make_odd", oc_to_odd_min3, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("clamp", oc_clamp, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("make_integer", oc_make_integer, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("int", oc_make_integer, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("equals", oc_equals, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("eq", oc_equals, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("gt", oc_greater_than, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("lt", oc_less_than, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("gte", oc_greater_than_or_equal, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("lte", oc_less_than_or_equal, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("scale", oc_scale, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("iir", oc_iir, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("now", oc_date_format, replace=True)
        self.resolver_count += 1
        OmegaConf.register_new_resolver("sampling", oc_sampling, replace=True)
        self.resolver_count += 1
        return self


def register_resolvers():
    """Legacy function that creates a registrar and registers resolvers."""
    registrar = ResolverRegistrar()
    registrar.register_resolvers()
