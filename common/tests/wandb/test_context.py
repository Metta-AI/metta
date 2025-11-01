"""
Unit tests for WandbContext - simplified version with good coverage.
"""

import pytest
import wandb
from wandb.errors import CommError

from metta.common.wandb.context import WandbConfig, WandbContext
from mettagrid.config import Config


class SampleConfig(Config):
    """Simple configuration for testing."""

    test_value: str = "test"
    number: int = 42


class FakeRun:
    """Simple fake wandb run object."""

    def __init__(self, name="test-run", id="run-123"):
        self.name = name
        self.id = id


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


def test_wandb_config_uri_raises_error():
    """Test that accessing URI raises RuntimeError."""
    cfg = WandbConfig(enabled=True, project="p", entity="e")
    with pytest.raises(RuntimeError, match="Policy artifacts are no longer stored on WandB"):
        _ = cfg.uri


def test_disabled_context_returns_none():
    """Test that disabled config skips initialization and returns None."""
    cfg = WandbConfig.Off()
    with WandbContext(cfg, SampleConfig()) as run:
        assert run is None


def test_network_check_failure_returns_none():
    """Test behavior when network check fails - should return None."""
    cfg = WandbConfig(enabled=True, project="test", entity="test")
    ctx = WandbContext(cfg, SampleConfig())
    ctx.wandb_host = "invalid.host.example.com"

    with ctx as run:
        assert run is None


def test_successful_initialization_and_parameters(monkeypatch):
    """Test successful wandb initialization with all parameters."""
    init_calls = []
    save_calls = []
    finish_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    def fake_save(*args, **kwargs):
        save_calls.append((args, kwargs))

    def fake_finish():
        finish_calls.append(True)

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "save", fake_save)
    monkeypatch.setattr(wandb, "finish", fake_finish)
    monkeypatch.setenv("METTA_USER", "alice")

    # Test with full configuration
    cfg = WandbConfig(
        enabled=True,
        project="test-proj",
        entity="test-ent",
        group="test-group",
        run_id="custom-id",
        job_type="training",
        tags=["custom", "test"],
        notes="Test notes",
        data_dir="/tmp/test",
    )

    with WandbContext(cfg, SampleConfig(), timeout=15, run_config_name="my_config") as run:
        assert isinstance(run, FakeRun)

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
    assert "test" in kwargs["tags"]
    assert "user:alice" in kwargs["tags"]
    assert kwargs["resume"] == "allow"
    assert kwargs["settings"].init_timeout == 15
    assert "my_config" in kwargs["config"]

    # Verify save calls
    assert len(save_calls) == 3  # .log, .yaml, .json

    # Verify cleanup
    assert len(finish_calls) == 1

    # Test with missing METTA_USER
    monkeypatch.delenv("METTA_USER", raising=False)
    # Ensure USER fallback is deterministic across CI environments
    # CI often sets USER=runner; force to 'unknown' for this assertion
    monkeypatch.setenv("USER", "unknown")
    init_calls.clear()

    cfg2 = WandbConfig(enabled=True, project="test", entity="test")
    with WandbContext(cfg2, None) as run:
        assert run is not None

    assert "user:unknown" in init_calls[0]["tags"]


def test_run_config_types(monkeypatch, caplog):
    """Test all different run_config types and error handling."""
    init_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return FakeRun()

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "save", lambda *args, **kwargs: None)

    cfg = WandbConfig(enabled=True, project="test", entity="test")

    # Test 1: Dict config
    config_dict = {"learning_rate": 0.01, "batch_size": 32}
    with WandbContext(cfg, config_dict) as run:
        assert run is not None
    assert init_calls[-1]["config"] == {"extra_config_dict": config_dict}

    # Test 2: Dict config with custom name
    with WandbContext(cfg, config_dict, run_config_name="params") as run:
        assert run is not None
    assert init_calls[-1]["config"] == {"params": config_dict}

    # Test 3: String config
    config_str = "simple string config"
    with WandbContext(cfg, config_str) as run:
        assert run is not None
    assert init_calls[-1]["config"] == config_str

    # Test 4: Config object
    sample_config = SampleConfig()
    with WandbContext(cfg, sample_config) as run:
        assert run is not None
    assert init_calls[-1]["config"]["SampleConfig"] == sample_config.model_dump()

    # Test 5: Invalid type
    invalid_config = 12345
    with WandbContext(cfg, invalid_config) as run:
        assert run is not None
    assert init_calls[-1]["config"] is None
    assert "Invalid extra_cfg: 12345" in caplog.text


def test_file_saving_behavior(monkeypatch):
    """Test wandb.save behavior with and without data_dir."""
    save_calls = []

    def fake_init(**kwargs):
        return FakeRun()

    def fake_save(glob_pattern, base_path=None, policy=None):
        save_calls.append({"pattern": glob_pattern, "base_path": base_path, "policy": policy})

    monkeypatch.setattr(wandb, "init", fake_init)
    monkeypatch.setattr(wandb, "save", fake_save)
    monkeypatch.setattr(wandb, "finish", lambda: None)

    # Test with data_dir
    cfg_with_dir = WandbConfig(enabled=True, project="test", entity="test", data_dir="/data")
    with WandbContext(cfg_with_dir, None):
        pass

    assert len(save_calls) == 3
    patterns = [call["pattern"] for call in save_calls]
    assert "/data/*.log" in patterns
    assert "/data/*.yaml" in patterns
    assert "/data/*.json" in patterns
    assert all(call["base_path"] == "/data" for call in save_calls)
    assert all(call["policy"] == "live" for call in save_calls)

    # Test without data_dir
    save_calls.clear()
    cfg_no_dir = WandbConfig(enabled=True, project="test", entity="test", data_dir=None)
    with WandbContext(cfg_no_dir, None):
        pass

    assert len(save_calls) == 0


def test_exception_handling(monkeypatch):
    """Test all exception handling paths."""
    cfg = WandbConfig(enabled=True, project="test", entity="test")

    # Test TimeoutError
    def raise_timeout(**kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setattr(wandb, "init", raise_timeout)
    with WandbContext(cfg, None) as run:
        assert run is None

    # Test CommError
    def raise_comm_error(**kwargs):
        raise CommError("comm error")

    monkeypatch.setattr(wandb, "init", raise_comm_error)
    with WandbContext(cfg, None) as run:
        assert run is None

    # Test generic Exception
    def raise_runtime(**kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(wandb, "init", raise_runtime)
    with WandbContext(cfg, None) as run:
        assert run is None

    # Test cleanup error handling
    def fake_finish_error():
        raise Exception("cleanup error")

    monkeypatch.setattr(wandb, "finish", fake_finish_error)
    # Should not raise
    WandbContext.cleanup_run(FakeRun())  # type: ignore
