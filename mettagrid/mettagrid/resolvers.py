import random
from typing import Any, Dict, TypeVar, Union

import numpy as np
from omegaconf import OmegaConf

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
        - "sigmoid": Sigmoid scaling (slower growth at extremes, faster in middle)

    Returns:
    --------
    Numeric
        The scaled value in the output range
    """
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


def oc_scaled_range(lower_limit: Numeric, upper_limit: Numeric, center: Numeric, *, _root_: Dict[str, Any]) -> Numeric:
    """
    Generates a value centered around a specified point based on a "sampling" parameter that controls how
    widely the distribution spreads between the limiting values.

    Parameters:
    -----------
    lower_limit : Numeric
        The minimum allowed value (lower boundary).
    upper_limit : Numeric
        The maximum allowed value (upper boundary).
    center : Numeric
        The center point of the distribution. When sampling=0, this value is returned directly.
    _root_ : dict (a named argument provided by OmegaConf)
        A dictionary containing the "sampling" parameter. If None, sampling defaults to 0. Must be between 0 and 1.
        IMPORTANT: this parameter must be named "_root_" exactly as shown here.

    Returns:
    --------
    Numeric
        A value between lower_limit and upper_limit, with distribution controlled by the sampling parameter.
        Returns integer if center is an integer, float otherwise.
    """

    # Get sampling parameter from root, defaulting to 0
    _root_ = _root_ or {}
    sampling = _root_.get("sampling", 0)

    # Fast path: return center when sampling is 0
    if sampling == 0:
        return center

    assert sampling >= 0 and sampling <= 1, 'Environment configuration for "sampling" must be in range [0, 1]!'

    # Calculate the scaled range on both sides of the center
    left_range = sampling * (center - lower_limit)
    right_range = sampling * (upper_limit - center)

    # Generate a random value within the scaled range
    val = np.random.uniform(center - left_range, center + right_range)

    # Return integer if the center was an integer
    return int(round(val)) if isinstance(center, int) else val


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


def register_resolvers() -> None:
    """
    Register all OmegaConf resolvers defined in this module.
    This function should be called before using any configuration that depends on these resolvers.
    """
    OmegaConf.register_new_resolver("if", oc_if, replace=True)
    OmegaConf.register_new_resolver("uniform", oc_uniform, replace=True)
    OmegaConf.register_new_resolver("choose", oc_choose, replace=True)
    OmegaConf.register_new_resolver("div", oc_divide, replace=True)
    OmegaConf.register_new_resolver("subtract", oc_subtract, replace=True)
    OmegaConf.register_new_resolver("sub", oc_subtract, replace=True)
    OmegaConf.register_new_resolver("multiply", oc_multiply, replace=True)
    OmegaConf.register_new_resolver("mul", oc_multiply, replace=True)
    OmegaConf.register_new_resolver("add", oc_add, replace=True)
    OmegaConf.register_new_resolver("make_odd", oc_to_odd_min3, replace=True)
    OmegaConf.register_new_resolver("clamp", oc_clamp, replace=True)
    OmegaConf.register_new_resolver("make_integer", oc_make_integer, replace=True)
    OmegaConf.register_new_resolver("int", oc_make_integer, replace=True)
    OmegaConf.register_new_resolver("equals", oc_equals, replace=True)
    OmegaConf.register_new_resolver("eq", oc_equals, replace=True)
    OmegaConf.register_new_resolver("sampling", oc_scaled_range, replace=True)
    OmegaConf.register_new_resolver("gt", oc_greater_than, replace=True)
    OmegaConf.register_new_resolver("lt", oc_less_than, replace=True)
    OmegaConf.register_new_resolver("gte", oc_greater_than_or_equal, replace=True)
    OmegaConf.register_new_resolver("lte", oc_less_than_or_equal, replace=True)
    OmegaConf.register_new_resolver("scale", oc_scale, replace=True)
    OmegaConf.register_new_resolver("iir", oc_iir, replace=True)
