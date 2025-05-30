import pytest

from metta.util import validate_arg_types


@validate_arg_types
def add(x: int, y: int) -> int:
    return x + y


def test_validate_correct_types():
    assert add(1, 2) == 3


def test_validate_incorrect_types():
    with pytest.raises(AssertionError):
        add("1", 2)
