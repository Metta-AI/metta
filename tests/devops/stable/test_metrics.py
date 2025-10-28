"""Tests for metrics extraction."""

import pytest

from metta.jobs.job_metrics import extract_skypilot_job_id, fetch_wandb_metrics


def test_extract_skypilot_job_id_pattern1():
    """Test extraction of SkyPilot job ID from launcher logs."""
    log_text = """
    Launching job...
    Job submitted with ID: 12345
    Job is running
    """
    job_id = extract_skypilot_job_id(log_text)
    assert job_id == "12345"


def test_extract_skypilot_job_id_pattern2():
    """Test extraction using alternative pattern."""
    log_text = "Submitted job 67890"
    job_id = extract_skypilot_job_id(log_text)
    assert job_id == "67890"


def test_extract_skypilot_job_id_no_match():
    """Test that missing job ID returns None."""
    log_text = "Training logs without job ID"
    job_id = extract_skypilot_job_id(log_text)
    assert job_id is None


def test_fetch_metrics_with_mocked_wandb(monkeypatch):
    """Test wandb metric fetching with mocked wandb API."""

    # Mock wandb API
    class FakeRun:
        def history(self, keys, pandas=False):
            if keys == ["overview/sps"]:
                return [
                    {"overview/sps": 48000.0},
                    {"overview/sps": 49000.0},
                    {"overview/sps": 50000.0},
                    {"overview/sps": 51000.0},
                ]
            elif keys == ["env_agent/heart.get"]:
                return [
                    {"env_agent/heart.get": 1.0},
                    {"env_agent/heart.get": 1.23},
                    {"env_agent/heart.get": 1.5},
                    {"env_agent/heart.get": 1.3},
                ]
            return []

    class FakeApi:
        def runs(self, path, filters=None):
            # Return an iterator with our fake run
            return iter([FakeRun()])

    import metta.jobs.job_metrics as metrics_module

    monkeypatch.setattr(metrics_module.wandb, "Api", FakeApi)

    metrics, current_step = fetch_wandb_metrics(
        entity="team",
        project="proj",
        run_name="test_run",
        metric_keys=["overview/sps", "env_agent/heart.get"],
        last_n_percent=0.5,
    )

    assert "overview/sps" in metrics
    assert "env_agent/heart.get" in metrics
    # Should average last 50% (last 2 values)
    assert metrics["overview/sps"]["value"] == pytest.approx(50500.0)  # avg(50000, 51000)
    assert metrics["overview/sps"]["count"] == 2
    assert metrics["env_agent/heart.get"]["value"] == pytest.approx(1.4)  # avg(1.5, 1.3)
    assert metrics["env_agent/heart.get"]["count"] == 2
    assert current_step is None  # Not requested
