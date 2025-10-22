"""Tests for metrics extraction."""

import pytest

from metta.jobs.metrics import extract_wandb_info, fetch_wandb_metrics


def test_extract_wandb_url():
    """Test extraction of wandb run information from URL."""
    log_text = """
    Training started
    wandb: View run at https://wandb.ai/myteam/myproject/runs/abc123xyz
    Training complete
    """
    info = extract_wandb_info(log_text)

    assert info is not None
    assert info.entity == "myteam"
    assert info.project == "myproject"
    assert info.run_id == "abc123xyz"
    assert info.url == "https://wandb.ai/myteam/myproject/runs/abc123xyz"


def test_extract_wandb_no_url():
    """Test that missing wandb URL returns None."""
    log_text = "Training logs without wandb URL"
    info = extract_wandb_info(log_text)

    assert info is None


def test_fetch_metrics_with_mocked_wandb(monkeypatch):
    """Test wandb metric fetching with mocked wandb API."""
    from metta.jobs.metrics import WandBInfo

    wandb_info = WandBInfo(
        project="proj",
        entity="team",
        run_id="abc123",
        run_name="test_run",
        url="https://wandb.ai/team/proj/runs/abc123",
    )

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
        def run(self, path):
            return FakeRun()

    import metta.jobs.metrics as metrics_module

    monkeypatch.setattr(metrics_module.wandb, "Api", FakeApi)

    metrics = fetch_wandb_metrics(wandb_info, ["overview/sps", "env_agent/heart.get"], last_n_percent=0.5)

    assert "overview/sps" in metrics
    assert "env_agent/heart.get" in metrics
    # Should average last 50% (last 2 values)
    assert metrics["overview/sps"] == pytest.approx(50500.0)  # avg(50000, 51000)
    assert metrics["env_agent/heart.get"] == pytest.approx(1.4)  # avg(1.5, 1.3)
