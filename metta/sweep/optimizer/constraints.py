from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Any, Iterable, Literal, Protocol

from pydantic import Field

from mettagrid.base_config import Config

class Constraint(Protocol):
    """Interface for inter-parameter constraints."""

    def apply(self, suggestion: dict[str, Any]) -> dict[str, Any]:
        """Return a new/updated suggestion; do not rely on in-place mutation."""
        ...

    def is_satisfied(self, suggestion: dict[str, Any]) -> bool:
        ...


def _get_nested(data: dict[str, Any], path: str) -> Any:
    """Fetch a value from a nested dict using dot-separated keys."""
    if path in data:
        return data[path]
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(data: dict[str, Any], path: str, value: Any) -> bool:
    """Set a value on a nested dict using dot-separated keys."""
    if path in data:
        data[path] = value
        return True

    current: Any = data
    parts = path.split(".")
    for part in parts[:-1]:
        next_item = current.get(part)
        if not isinstance(next_item, dict):
            return False
        current = next_item

    current[parts[-1]] = value
    return True


class ConstraintAwareMixin:
    """Mixin that applies registered constraints to optimizer suggestions."""

    def __init__(self) -> None:
        self._constraints: list[Constraint] = []

    def register_constraint(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def register_constraints(self, constraints: Iterable[Constraint]) -> None:
        for constraint in constraints:
            self.register_constraint(constraint)

    def _apply_constraints_to_suggestion(self, suggestion: dict[str, Any]) -> dict[str, Any]:
        for constraint in self._constraints:
            suggestion = constraint.apply(suggestion)
        return suggestion

    def _apply_constraints(self, suggestions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._constraints or not suggestions:
            return suggestions
        constrained: list[dict[str, Any]] = []
        for suggestion in suggestions:
            constrained.append(self._apply_constraints_to_suggestion(suggestion))
        return constrained


class DivisionConstraint:
    """Clamp divisor or dividend so divisibility holds."""

    def __init__(self, divisor_key: str, dividend_key: str, adjust_target: Literal["divisor", "dividend"] = "divisor"):
        self.divisor_key = divisor_key
        self.dividend_key = dividend_key
        self.adjust_target = adjust_target

    def is_satisfied(self, suggestion: dict[str, Any]) -> bool:
        divisor = _get_nested(suggestion, self.divisor_key)
        dividend = _get_nested(suggestion, self.dividend_key)
        if divisor is None or dividend is None:
            return True  # Vacuously satisfied when keys are absent
        try:
            divisor_int = int(round(float(divisor)))
            dividend_int = int(round(float(dividend)))
        except (TypeError, ValueError):
            return False
        if divisor_int == 0:
            return False
        return dividend_int % divisor_int == 0

    def apply(self, suggestion: dict[str, Any]) -> dict[str, Any]:
        updated = copy.deepcopy(suggestion)
        divisor = _get_nested(updated, self.divisor_key)
        dividend = _get_nested(updated, self.dividend_key)
        if divisor is None or dividend is None:
            return updated
        try:
            divisor_int = int(round(float(divisor)))
            dividend_int = int(round(float(dividend)))
        except (TypeError, ValueError):
            return updated

        if divisor_int <= 0:
            return updated

        if self.adjust_target == "dividend":
            corrected_dividend = self._largest_multiple_at_most(dividend_int, divisor_int)
            if corrected_dividend != dividend_int:
                _set_nested(updated, self.dividend_key, corrected_dividend)
        else:
            corrected_divisor = self._largest_divisor_at_most(dividend_int, divisor_int)
            if corrected_divisor != divisor_int:
                _set_nested(updated, self.divisor_key, corrected_divisor)
        return updated

    def _largest_divisor_at_most(self, dividend: int, ceiling: int) -> int:
        upper = min(abs(dividend), ceiling)
        if upper <= 0:
            return ceiling
        for candidate in range(upper, 0, -1):
            if dividend % candidate == 0:
                return candidate
        return 1

    def _largest_multiple_at_most(self, dividend: int, divisor: int) -> int:
        if divisor <= 0:
            return dividend
        multiple = (dividend // divisor) * divisor
        return multiple


class ConstraintSpec(Config, ABC):
    """Abstract constraint spec that can build a concrete Constraint."""

    type: str = Field(description="Constraint type identifier")

    @abstractmethod
    def build(self) -> Constraint:
        ...


class DivisionConstraintSpec(ConstraintSpec):
    """Declarative spec for DivisionConstraint."""

    type: Literal["division"] = "division"
    divisor_key: str = Field(description="Parameter that should divide the dividend")
    dividend_key: str = Field(description="Parameter that must be divisible by the divisor")
    adjust_target: Literal["divisor", "dividend"] = Field(
        default="divisor", description="Which parameter to adjust to enforce the constraint"
    )

    def build(self) -> Constraint:
        return DivisionConstraint(
            divisor_key=self.divisor_key,
            dividend_key=self.dividend_key,
            adjust_target=self.adjust_target,
        )
