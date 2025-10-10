"""Tests for acceptance criteria evaluation."""

from operator import eq, ge, gt, le, lt, ne

from devops.stable.tasks import _evaluate_thresholds


def test_all_operators_pass():
    """Test that all comparison operators work correctly when passing."""
    metrics = {"a": 5, "b": 5, "c": 5, "d": 5, "e": 5, "f": 5}
    checks = [
        ("a", ge, 5),
        ("b", gt, 4),
        ("c", le, 5),
        ("d", lt, 6),
        ("e", eq, 5),
        ("f", ne, 4),
    ]
    outcome, failures = _evaluate_thresholds(metrics, checks)
    assert outcome == "passed"
    assert failures == []


def test_all_operators_fail():
    """Test that all comparison operators detect failures correctly."""
    metrics = {"a": 5, "b": 5, "c": 5, "d": 5, "e": 5, "f": 5}

    # Test each operator's failure case
    test_cases = [
        (("a", ge, 6), "a"),
        (("b", gt, 5), "b"),
        (("c", le, 4), "c"),
        (("d", lt, 5), "d"),
        (("e", eq, 6), "e"),
        (("f", ne, 5), "f"),
    ]

    for rule, expected_metric in test_cases:
        outcome, failures = _evaluate_thresholds(metrics, [rule])
        assert outcome == "failed"
        assert len(failures) == 1
        assert expected_metric in failures[0]


def test_missing_metric_fails():
    """Test that checking a missing metric results in failure."""
    metrics = {"a": 5}
    outcome, failures = _evaluate_thresholds(metrics, [("missing", gt, 0)])
    assert outcome == "failed"
    assert "missing" in failures[0]


def test_empty_rules_pass():
    """Test that empty acceptance criteria always pass."""
    metrics = {"a": 5}
    outcome, failures = _evaluate_thresholds(metrics, [])
    assert outcome == "passed"
    assert failures == []


def test_multiple_failures_reported():
    """Test that all failing criteria are reported."""
    metrics = {"a": 1, "b": 2, "c": 3}
    checks = [
        ("a", gt, 10),  # Fail
        ("b", gt, 20),  # Fail
        ("c", gt, 30),  # Fail
    ]
    outcome, failures = _evaluate_thresholds(metrics, checks)
    assert outcome == "failed"
    assert len(failures) == 3
