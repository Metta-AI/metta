import pytest
from pydantic import ValidationError, validate_call


@validate_call
def add(x: int, y: int) -> int:
    return x + y


def test_validate_correct_types():
    assert add(1, 2) == 3


def test_validate_incorrect_types():
    # Note: Pydantic will try to coerce types, so "1" would be converted to 1
    # We need to use a value that can't be coerced to test validation
    with pytest.raises(ValidationError):
        add("not_a_number", 2)
