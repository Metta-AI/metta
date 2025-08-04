"""Check system for tAXIOM - runtime validation with statistical sampling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class CheckLevel(Enum):
    """Severity levels for checks."""
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Check:
    """A runtime validation check.
    
    Checks verify data integrity at stage boundaries without affecting outcomes.
    They can be sampled for noisy domains where checking every value would be expensive
    or where transient failures are expected.
    """
    
    name: str
    predicate: Callable[[Any], bool]
    level: CheckLevel = CheckLevel.WARN
    sample_every: int = 1  # Check every Nth call
    n_of_m: tuple[int, int] | None = None  # Warn if N failures in last M checks
    
    def __post_init__(self):
        self._call_count = 0
        self._history = deque(maxlen=self.n_of_m[1] if self.n_of_m else 100)
    
    def should_check(self) -> bool:
        """Determine if this call should be checked based on sampling."""
        self._call_count += 1
        return self._call_count % self.sample_every == 0
    
    def check(self, value: Any) -> tuple[bool, str | None]:
        """Run the check on a value.
        
        Returns:
            (passed, error_message)
        """
        if not self.should_check():
            return True, None
        
        try:
            passed = self.predicate(value)
        except Exception as e:
            passed = False
            error_msg = f"Check '{self.name}' raised exception: {e}"
            return False, error_msg
        
        # Track history for n_of_m checks
        if self.n_of_m:
            self._history.append(passed)
            
            # Check if we've exceeded failure threshold
            if len(self._history) >= self.n_of_m[1]:
                failures = sum(1 for x in self._history if not x)
                if failures >= self.n_of_m[0]:
                    return False, f"Check '{self.name}' failed {failures} of last {self.n_of_m[1]} times"
        
        if not passed:
            return False, f"Check '{self.name}' failed"
        
        return True, None


def WARN(predicate: Callable[[Any], bool], sample_every: int = 1, n_of_m: tuple[int, int] | None = None) -> Check:
    """Create a warning-level check."""
    return Check(
        name="unnamed",
        predicate=predicate,
        level=CheckLevel.WARN,
        sample_every=sample_every,
        n_of_m=n_of_m
    )


def FAIL(predicate: Callable[[Any], bool], sample_every: int = 1) -> Check:
    """Create a failure-level check (stops execution)."""
    return Check(
        name="unnamed",
        predicate=predicate,
        level=CheckLevel.FAIL,
        sample_every=sample_every
    )


# Common check predicates
def required_keys(keys: list[str]) -> Callable[[Any], bool]:
    """Check that a dict has required keys."""
    def check(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return all(k in value for k in keys)
    return check


def no_nan() -> Callable[[Any], bool]:
    """Check for NaN values."""
    import math
    import torch
    import numpy as np
    
    def check(value: Any) -> bool:
        if isinstance(value, (float, int)):
            return not math.isnan(float(value))
        elif isinstance(value, torch.Tensor):
            return not torch.isnan(value).any().item()
        elif isinstance(value, np.ndarray):
            return not np.isnan(value).any()
        elif isinstance(value, dict):
            return all(check(v) for v in value.values())
        elif isinstance(value, (list, tuple)):
            return all(check(v) for v in value)
        return True
    return check


def prob_simplex(tolerance: float = 1e-6) -> Callable[[Any], bool]:
    """Check that values form a probability simplex."""
    import torch
    import numpy as np
    
    def check(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                return False
            return (
                (value >= 0).all().item() and 
                abs(value.sum().item() - 1.0) < tolerance
            )
        elif isinstance(value, np.ndarray):
            if value.ndim == 0:
                return False
            return (
                (value >= 0).all() and 
                abs(value.sum() - 1.0) < tolerance
            )
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return False
            total = sum(value)
            return all(v >= 0 for v in value) and abs(total - 1.0) < tolerance
        return False
    return check


def grad_band(min_val: float = 1e-6, max_val: float = 100.0) -> Callable[[Any], bool]:
    """Check that gradients are within a reasonable band."""
    import torch
    
    def check(value: Any) -> bool:
        if isinstance(value, dict) and "gradients" in value:
            grads = value["gradients"]
            if isinstance(grads, torch.Tensor):
                grad_norm = grads.norm().item()
                return min_val <= grad_norm <= max_val
        return True
    return check


def in_range(min_val: float, max_val: float) -> Callable[[Any], bool]:
    """Check that value is in range."""
    def check(value: Any) -> bool:
        if isinstance(value, (int, float)):
            return min_val <= value <= max_val
        elif isinstance(value, dict) and "score" in value:
            return min_val <= value["score"] <= max_val
        return False
    return check