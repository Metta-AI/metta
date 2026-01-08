import numpy as np
import pytest

from metta.cogworks.curriculum.learning_progress_algorithm import (
    LearningProgressAlgorithm,
    LearningProgressConfig,
)


def make_algorithm(progress_smoothing: float) -> LearningProgressAlgorithm:
    config = LearningProgressConfig(progress_smoothing=progress_smoothing)
    return LearningProgressAlgorithm(num_tasks=3, hypers=config)


def test_scalar_and_vector_reweight_match_across_range():
    algo = make_algorithm(progress_smoothing=0.2)
    probs = np.array([-0.9, -0.5, -0.1, 0.0, 0.25, 0.6, 0.9])

    vector = algo._reweight(probs)
    scalar = np.array([algo._reweight(float(p)) for p in probs])

    np.testing.assert_allclose(vector, scalar)


def test_reweight_still_smooths_when_denominator_would_flip_sign():
    algo = make_algorithm(progress_smoothing=0.2)
    prob = -0.5  # yields denominator <= 0 without clamping

    smoothed_scalar = algo._reweight(prob)
    smoothed_vector = algo._reweight(np.array([prob]))[0]

    assert smoothed_scalar == pytest.approx(smoothed_vector)
    # Smoothing should change the value instead of skipping it
    assert smoothed_scalar == pytest.approx(prob * (1 - algo.hypers.progress_smoothing))
    assert smoothed_scalar != prob
