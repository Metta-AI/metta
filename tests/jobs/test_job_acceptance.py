"""Behavioral tests for job acceptance criteria evaluation.

Tests that acceptance criteria are correctly evaluated against metrics.
"""

import time
from unittest.mock import patch

from metta.jobs.job_config import AcceptanceCriterion
from tests.jobs.conftest import MockProcess, simple_job_config


def test_acceptance_criterion_operators():
    """AcceptanceCriterion.evaluate() correctly evaluates all operators."""
    criterion = AcceptanceCriterion(metric="test", operator=">=", threshold=10.0)
    assert criterion.evaluate(10.0) is True
    assert criterion.evaluate(11.0) is True
    assert criterion.evaluate(9.0) is False

    criterion = AcceptanceCriterion(metric="test", operator=">", threshold=10.0)
    assert criterion.evaluate(11.0) is True
    assert criterion.evaluate(10.0) is False

    criterion = AcceptanceCriterion(metric="test", operator="<=", threshold=10.0)
    assert criterion.evaluate(9.0) is True
    assert criterion.evaluate(10.0) is True
    assert criterion.evaluate(11.0) is False

    criterion = AcceptanceCriterion(metric="test", operator="<", threshold=10.0)
    assert criterion.evaluate(9.0) is True
    assert criterion.evaluate(10.0) is False

    criterion = AcceptanceCriterion(metric="test", operator="==", threshold=10.0)
    assert criterion.evaluate(10.0) is True
    assert criterion.evaluate(10.1) is False


def test_acceptance_evaluation_with_metrics(temp_job_manager):
    """Acceptance criteria are evaluated correctly when metrics present."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=0, complete_after_polls=1)

    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config(
            "test_job",
            acceptance_criteria=[
                AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=10000),
                AcceptanceCriterion(metric="reward", operator=">", threshold=0.5),
            ],
        )
        manager.submit(config)

        time.sleep(0.2)

        job_state = manager.get_job_state("test_job")
        job_state.metrics = {"overview/sps": 15000.0, "reward": 0.8}

        result = manager._evaluate_acceptance(job_state)
        assert result is True

        job_state.metrics = {"overview/sps": 15000.0, "reward": 0.3}
        result = manager._evaluate_acceptance(job_state)
        assert result is False


def test_acceptance_with_no_criteria(temp_job_manager):
    """Jobs without acceptance criteria evaluate to True (vacuous truth)."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=0, complete_after_polls=1)

    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config("test_job", acceptance_criteria=[])
        manager.submit(config)

        time.sleep(0.2)

        job_state = manager.get_job_state("test_job")
        result = manager._evaluate_acceptance(job_state)
        assert result is True


def test_acceptance_with_missing_metrics(temp_job_manager):
    """Acceptance fails when required metrics are missing."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=0, complete_after_polls=1)

    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config(
            "test_job",
            acceptance_criteria=[
                AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=10000),
            ],
        )
        manager.submit(config)

        time.sleep(0.2)

        job_state = manager.get_job_state("test_job")
        assert job_state.metrics == {}

        result = manager._evaluate_acceptance(job_state)
        assert result is False


def test_multiple_criteria_all_must_pass(temp_job_manager):
    """When job has multiple criteria, all must pass."""
    manager = temp_job_manager()
    mock_process = MockProcess(exit_code=0, complete_after_polls=1)

    with patch("subprocess.Popen", return_value=mock_process):
        config = simple_job_config(
            "test_job",
            acceptance_criteria=[
                AcceptanceCriterion(metric="metric1", operator=">=", threshold=10),
                AcceptanceCriterion(metric="metric2", operator=">=", threshold=20),
                AcceptanceCriterion(metric="metric3", operator=">=", threshold=30),
            ],
        )
        manager.submit(config)

        time.sleep(0.2)

        job_state = manager.get_job_state("test_job")

        job_state.metrics = {"metric1": 15, "metric2": 25, "metric3": 35}
        assert manager._evaluate_acceptance(job_state) is True

        job_state.metrics = {"metric1": 15, "metric2": 15, "metric3": 35}
        assert manager._evaluate_acceptance(job_state) is False

        job_state.metrics = {"metric1": 5, "metric2": 15, "metric3": 25}
        assert manager._evaluate_acceptance(job_state) is False
