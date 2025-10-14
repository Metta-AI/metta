"""Tests for metrics extraction."""

import pytest

from devops.stable.metrics import extract_metrics, extract_wandb_run_info


def test_extract_wandb_url():
    """Test extraction of wandb run information from URL."""
    log_text = """
    Training started
    View run at: https://wandb.ai/myteam/myproject/runs/abc123xyz
    Training complete
    """
    info = extract_wandb_run_info(log_text)

    assert info is not None
    entity, project, run_id = info
    assert entity == "myteam"
    assert project == "myproject"
    assert run_id == "abc123xyz"


def test_extract_wandb_no_url():
    """Test that missing wandb URL returns None."""
    log_text = "Training logs without wandb URL"
    info = extract_wandb_run_info(log_text)

    assert info is None


def test_extract_metrics_empty_log():
    """Test that empty logs return empty metrics dict."""
    metrics = extract_metrics("")
    assert metrics == {}


def test_extract_metrics_no_wandb_metrics():
    """Test that extract_metrics returns empty dict when no wandb_metrics specified."""
    log_text = """
    Training started
    Some log output
    View run: https://wandb.ai/team/proj/runs/run123
    """
    metrics = extract_metrics(log_text)
    assert metrics == {}


def test_extract_metrics_no_wandb_url():
    """Test that extract_metrics returns empty dict when wandb URL not found."""
    log_text = "Training logs without wandb URL"
    metrics = extract_metrics(log_text, wandb_metrics=["some/metric"])
    assert metrics == {}


def test_extract_with_wandb_metrics_mocked(monkeypatch):
    """Test wandb metric fetching when wandb_metrics list is provided."""
    log_text = "https://wandb.ai/team/proj/runs/abc123"

    # Mock fetch_wandb_metric to avoid real API call
    def fake_fetch(entity, project, run_id, metric_key, **kwargs):
        if metric_key == "overview/sps":
            return 50000.0
        elif metric_key == "env_agent/heart.get":
            return 1.23
        return None

    import devops.stable.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "fetch_wandb_metric", fake_fetch)

    metrics = extract_metrics(log_text, wandb_metrics=["overview/sps", "env_agent/heart.get"])

    assert "overview/sps" in metrics
    assert "env_agent/heart.get" in metrics
    assert metrics["overview/sps"] == pytest.approx(50000.0)
    assert metrics["env_agent/heart.get"] == pytest.approx(1.23)
