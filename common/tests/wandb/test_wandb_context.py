"""
Unit tests for WandbContext.
"""

import os
import tempfile

import wandb
from wandb.errors import CommError

from metta.common.wandb.wandb_context import WandbConfig, WandbContext
from metta.mettagrid.config import Config


class SampleConfig(Config):
    """Simple configuration for testing."""

    test_value: str = "test"
    number: int = 42


def test_wandb_config_factories():
    """Test WandbConfig factory methods."""
    # Test Off()
    cfg_off = WandbConfig.Off()
    assert cfg_off.enabled is False
    assert cfg_off.project == "na"
    assert cfg_off.entity == "na"

    # Test Unconfigured()
    cfg_unconf = WandbConfig.Unconfigured()
    assert cfg_unconf.enabled is False
    assert cfg_unconf.project == "unconfigured"
    assert cfg_unconf.entity == "unconfigured"


def test_wandb_config_uri():
    """Test URI generation."""
    cfg = WandbConfig(enabled=True, project="p", entity="e", run_id="test-123")
    assert cfg.uri == "wandb://run/test-123"


def test_disabled_context():
    """Test that disabled config skips initialization."""
    cfg = WandbConfig.Off()
    with WandbContext(cfg, SampleConfig()) as run:
        assert run is None


def test_context_initialization():
    """Test context attributes initialization."""
    cfg = WandbConfig(enabled=True, project="test", entity="test")
    ctx = WandbContext(cfg, SampleConfig(), timeout=45)

    assert ctx.cfg == cfg
    assert ctx.timeout == 45
    assert ctx.wandb_host == "api.wandb.ai"
    assert ctx.wandb_port == 443
    assert ctx.run is None


def test_network_check_failure():
    """Test behavior when network check fails."""
    cfg = WandbConfig(enabled=True, project="test", entity="test")
    ctx = WandbContext(cfg, SampleConfig())
    ctx.wandb_host = "invalid.host.example.com"

    with ctx as run:
        assert run is None


def test_wandb_offline_mode():
    """Test initialization in offline mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = tmpdir

        cfg = WandbConfig(enabled=True, project="offline-test", entity="test-entity", data_dir=tmpdir)

        try:
            with WandbContext(cfg, SampleConfig(), timeout=5) as _run:
                # Offline mode might still create a run
                pass
        finally:
            os.environ.pop("WANDB_MODE", None)
            os.environ.pop("WANDB_DIR", None)


def test_config_fields():
    """Test WandbConfig with various field combinations."""
    # Test defaults
    cfg_min = WandbConfig(enabled=True, project="p", entity="e")
    assert cfg_min.group is None
    assert cfg_min.tags == []
    assert cfg_min.notes == ""

    # Test full config
    cfg_full = WandbConfig(
        enabled=True,
        project="my-project",
        entity="my-entity",
        group="experiment-1",
        name="run-name",
        run_id="custom-id",
        data_dir="/tmp/data",
        job_type="training",
        tags=["tag1", "tag2"],
        notes="Experiment notes",
    )
    assert cfg_full.group == "experiment-1"
    assert cfg_full.tags == ["tag1", "tag2"]
    assert cfg_full.notes == "Experiment notes"


def test_cleanup_run_static():
    """Test static cleanup_run method."""
    WandbContext.cleanup_run(None)  # Should not raise


# Tests using monkeypatch instead of mocks
class FakeRun:
    """Simple fake wandb run object."""

    def __init__(self, name="test-run", id="run-123"):
        self.name = name
        self.id = id


def test_successful_init_params(monkeypatch):
    """Test successful wandb.init call with correct parameters."""
    init_calls = []
    save_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    def fake_save(*args, **kwargs):
        save_calls.append((args, kwargs))

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "save", fake_save)
    monkeypatch.setenv("METTA_USER", "alice")

    cfg = WandbConfig(
        enabled=True,
        project="test-proj",
        entity="test-ent",
        group="test-group",
        run_id="custom-id",
        job_type="training",
        tags=["custom"],
        notes="Test notes",
        data_dir="/tmp/test",
    )

    with WandbContext(cfg, SampleConfig(), timeout=15) as run:
        assert isinstance(run, FakeRun)
        assert run.name == "test-run"
        assert run.id == "run-123"

    # Verify init parameters
    assert len(init_calls) == 1
    kwargs = init_calls[0]
    assert kwargs["id"] == "custom-id"
    assert kwargs["project"] == "test-proj"
    assert kwargs["entity"] == "test-ent"
    assert kwargs["group"] == "test-group"
    assert kwargs["job_type"] == "training"
    assert kwargs["notes"] == "Test notes"
    assert "custom" in kwargs["tags"]
    assert "user:alice" in kwargs["tags"]
    assert kwargs["resume"] is True
    assert kwargs["settings"].init_timeout == 15

    # Verify save calls for log/yaml/json files
    assert len(save_calls) == 3


def test_exception_handling(monkeypatch):
    """Test various exception handling paths."""
    cfg = WandbConfig(enabled=True, project="test", entity="test")

    # Test TimeoutError
    def raise_timeout(**kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setattr(wandb, "init", raise_timeout)
    with WandbContext(cfg, SampleConfig()) as run:
        assert run is None

    # Test CommError
    def raise_comm_error(**kwargs):
        raise CommError("comm error")

    monkeypatch.setattr(wandb, "init", raise_comm_error)
    with WandbContext(cfg, SampleConfig()) as run:
        assert run is None

    # Test generic Exception
    def raise_runtime(**kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(wandb, "init", raise_runtime)
    with WandbContext(cfg, SampleConfig()) as run:
        assert run is None


def test_cleanup_calls_finish(monkeypatch):
    """Test cleanup calls wandb.finish for active run."""
    finish_called = []

    def fake_init(**kwargs):
        return FakeRun()

    def fake_finish():
        finish_called.append(True)

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, "finish", fake_finish)

    with WandbContext(WandbConfig(enabled=True, project="p", entity="e"), SampleConfig()):
        pass

    assert len(finish_called) == 1

    # Test cleanup error handling
    def fake_finish_error():
        raise Exception("cleanup error")

    monkeypatch.setattr(wandb, "finish", fake_finish_error)
    # Should not raise
    WandbContext.cleanup_run(FakeRun())  # type: ignore
