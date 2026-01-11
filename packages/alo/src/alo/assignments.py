from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def allocate_counts(total: int, weights: Sequence[float], *, allow_zero_total: bool = False) -> list[int]:
    if total < 0:
        raise ValueError("Total must be non-negative.")
    if not weights:
        if allow_zero_total:
            return []
        raise ValueError("At least one weight is required.")

    total_weight = sum(weights)
    if total_weight <= 0:
        if allow_zero_total:
            return [0] * len(weights)
        raise ValueError("Total weight must be positive.")

    fractions = [weight / total_weight for weight in weights]
    ideals = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in ideals]
    remaining = total - sum(counts)

    remainders = [ideal - count for ideal, count in zip(ideals, counts, strict=True)]
    for idx in sorted(range(len(remainders)), key=remainders.__getitem__, reverse=True)[:remaining]:
        counts[idx] += 1
    return counts


def build_assignments(total: int, weights: Sequence[float], *, allow_zero_total: bool = False) -> np.ndarray:
    counts = allocate_counts(total, weights, allow_zero_total=allow_zero_total)
    return np.repeat(np.arange(len(counts)), counts)
