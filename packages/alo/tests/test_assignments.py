from __future__ import annotations

import numpy as np
import pytest

from alo.assignments import allocate_counts, build_assignments


def test_allocate_counts_even_split() -> None:
    counts = allocate_counts(10, [1.0, 1.0])

    assert counts == [5, 5]


def test_allocate_counts_zero_total_allow() -> None:
    counts = allocate_counts(0, [1.0, 2.0], allow_zero_total=True)

    assert counts == [0, 0]


def test_allocate_counts_rejects_zero_weights() -> None:
    with pytest.raises(ValueError):
        allocate_counts(3, [0.0, 0.0])


def test_build_assignments_counts() -> None:
    assignments = build_assignments(5, [1.0, 1.0])

    assert assignments.shape == (5,)
    assert int(np.sum(assignments == 0)) == 3
    assert int(np.sum(assignments == 1)) == 2
