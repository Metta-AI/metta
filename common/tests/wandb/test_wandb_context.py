#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pytest",
#     "wandb",
#     "omegaconf",
# ]
# ///
"""
Unit tests for WandbContext.
"""

import logging
import socket
from unittest.mock import MagicMock, patch

import pytest
import wandb
from wandb.errors import CommError

from metta.common.config import Config
from metta.common.wandb.wandb_context import WandbConfig, WandbContext

logger = logging.getLogger("Test")


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    """Patch wandb.save and socket to avoid real network calls."""
    # Patch wandb.save to no-op
    monkeypatch.setattr(wandb, "save", lambda *args, **kwargs: None)

    # Dummy socket to bypass real network calls
    class DummySock:
        def settimeout(self, timeout):
            pass

        def connect(self, addr):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: DummySock())
    yield


@pytest.fixture
def mock_wandb_save(monkeypatch):
    """Mock wandb.save to avoid file operations."""
    mock_save = MagicMock()
    monkeypatch.setattr(wandb, "save", mock_save)
    return mock_save


@pytest.fixture
def mock_wandb_init(monkeypatch):
    """Mock wandb.init with a proper return value."""
    mock_run = MagicMock()
    mock_run.id = "test-run-id"
    mock_run.name = "test-run-name"

    mock_init = MagicMock(return_value=mock_run)
    monkeypatch.setattr(wandb, "init", mock_init)
    return mock_init, mock_run


@pytest.fixture
def mock_wandb_finish(monkeypatch):
    """Mock wandb.finish."""
    mock_finish = MagicMock()
    monkeypatch.setattr(wandb, "finish", mock_finish)
    return mock_finish


@pytest.fixture
def mock_socket(monkeypatch):
    """Mock socket for connection testing."""
    mock_sock = MagicMock()
    mock_socket_class = MagicMock(return_value=mock_sock)
    monkeypatch.setattr(socket, "socket", mock_socket_class)
    return mock_sock


class SampleGlobalConfig(Config):
    """Sample configuration class for testing."""

    test_param: str = "test_value"
    nested: dict = {"key": "value"}


def test_wandb_config_off():
    """Test WandbConfig.Off() creates disabled config."""
    cfg = WandbConfig.Off()
    assert cfg.enabled is False
    assert cfg.project == "na"
    assert cfg.entity == "na"


def test_wandb_config_unconfigured():
    """Test WandbConfig.Unconfigured() creates unconfigured config."""
    cfg = WandbConfig.Unconfigured()
    assert cfg.enabled is False
    assert cfg.project == "unconfigured"
    assert cfg.entity == "unconfigured"


def test_wandb_config_uri():
    """Test WandbConfig.uri property."""
    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", name="test-run", run_id="test-run-id")
    assert cfg.uri == "wandb://run/test-run-id"


def test_context_disabled_no_init(mock_wandb_init):
    """Test that disabled config doesn't initialize wandb."""
    cfg = WandbConfig.Off()
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    result = ctx.__enter__()

    assert result is None
    mock_wandb_init[0].assert_not_called()


def test_context_no_network_connection(mock_socket, mock_wandb_init):
    """Test behavior when no network connection is available."""
    # Configure socket to raise connection error
    mock_socket.connect.side_effect = ConnectionError("No network")

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir="/tmp/test")
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    result = ctx.__enter__()

    # Should handle gracefully and return None
    assert result is None
    mock_wandb_init[0].assert_not_called()


def test_context_successful_init(mock_socket, mock_wandb_init, mock_wandb_save, tmp_path):
    """Test successful wandb initialization."""
    mock_init, mock_run = mock_wandb_init

    cfg = WandbConfig(
        enabled=True,
        project="test-project",
        entity="test-entity",
        group="test-group",
        name="test-run",
        run_id="custom-run-id",
        data_dir=str(tmp_path),
        job_type="test-job",
        tags=["tag1", "tag2"],
        notes="Test notes",
    )
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg, timeout=15)
    result = ctx.__enter__()

    # Should return the mock run
    assert result == mock_run
    assert ctx.run == mock_run

    # Verify wandb.init was called with correct parameters
    mock_init.assert_called_once()
    call_kwargs = mock_init.call_args.kwargs

    assert call_kwargs["id"] == "custom-run-id"
    assert call_kwargs["project"] == "test-project"
    assert call_kwargs["entity"] == "test-entity"
    assert call_kwargs["group"] == "test-group"
    assert call_kwargs["job_type"] == "test-job"
    assert call_kwargs["notes"] == "Test notes"
    assert "tag1" in call_kwargs["tags"]
    assert "tag2" in call_kwargs["tags"]
    assert "user:unknown" in call_kwargs["tags"]  # Default user tag
    assert call_kwargs["config"] == global_cfg.model_dump()
    assert call_kwargs["settings"].init_timeout == 15

    # Verify save was called for log files
    assert mock_wandb_save.call_count == 3  # .log, .yaml, .json


def test_context_with_user_env(mock_socket, mock_wandb_init, mock_wandb_save, tmp_path, monkeypatch):
    """Test that METTA_USER environment variable is used in tags."""
    monkeypatch.setenv("METTA_USER", "test-user")
    mock_init, mock_run = mock_wandb_init

    cfg = WandbConfig(
        enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path), tags=["custom-tag"]
    )
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    ctx.__enter__()

    # Verify user tag was added
    call_kwargs = mock_init.call_args.kwargs
    assert "user:test-user" in call_kwargs["tags"]
    assert "custom-tag" in call_kwargs["tags"]


def test_context_timeout_error(mock_socket, monkeypatch, mock_wandb_save, tmp_path):
    """Test handling of timeout errors during init."""

    def mock_init_timeout(*args, **kwargs):
        raise TimeoutError("Initialization timed out")

    monkeypatch.setattr(wandb, "init", mock_init_timeout)

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path))
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    result = ctx.__enter__()

    # Should handle gracefully
    assert result is None
    assert ctx.run is None


def test_context_comm_error(mock_socket, monkeypatch, mock_wandb_save, tmp_path):
    """Test handling of wandb communication errors."""

    def mock_init_comm_error(*args, **kwargs):
        raise CommError("Communication failed")

    monkeypatch.setattr(wandb, "init", mock_init_comm_error)

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path))
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    result = ctx.__enter__()

    # Should handle gracefully
    assert result is None
    assert ctx.run is None


def test_context_unexpected_error(mock_socket, monkeypatch, mock_wandb_save, tmp_path):
    """Test handling of unexpected errors during init."""

    def mock_init_error(*args, **kwargs):
        raise RuntimeError("Unexpected error")

    monkeypatch.setattr(wandb, "init", mock_init_error)

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path))
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    result = ctx.__enter__()

    # Should handle gracefully
    assert result is None
    assert ctx.run is None


def test_context_exit_with_run(mock_socket, mock_wandb_init, mock_wandb_finish, mock_wandb_save, tmp_path):
    """Test __exit__ properly cleans up the run."""
    mock_init, mock_run = mock_wandb_init

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path))
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    ctx.__enter__()
    ctx.__exit__(None, None, None)

    # Should call wandb.finish
    mock_wandb_finish.assert_called_once()


def test_context_exit_without_run(mock_wandb_finish):
    """Test __exit__ when no run was created."""
    cfg = WandbConfig.Off()
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    ctx.__enter__()
    ctx.__exit__(None, None, None)

    # Should not call wandb.finish
    mock_wandb_finish.assert_not_called()


def test_context_exit_with_error(mock_socket, mock_wandb_init, mock_wandb_finish, mock_wandb_save, tmp_path):
    """Test __exit__ handles errors during cleanup."""
    mock_init, mock_run = mock_wandb_init
    mock_wandb_finish.side_effect = RuntimeError("Cleanup failed")

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path))
    global_cfg = SampleGlobalConfig()

    ctx = WandbContext(cfg, global_cfg)
    ctx.__enter__()

    # Should not raise even if finish fails
    ctx.__exit__(None, None, None)
    mock_wandb_finish.assert_called_once()


def test_cleanup_run_static_method(mock_wandb_finish):
    """Test the static cleanup_run method."""
    mock_run = MagicMock()

    # Test with a run
    WandbContext.cleanup_run(mock_run)
    mock_wandb_finish.assert_called_once()

    # Reset mock
    mock_wandb_finish.reset_mock()

    # Test with None
    WandbContext.cleanup_run(None)
    mock_wandb_finish.assert_not_called()


def test_cleanup_run_with_error(mock_wandb_finish):
    """Test cleanup_run handles errors gracefully."""
    mock_run = MagicMock()
    mock_wandb_finish.side_effect = RuntimeError("Finish failed")

    # Should not raise
    WandbContext.cleanup_run(mock_run)
    mock_wandb_finish.assert_called_once()


def test_context_as_context_manager(mock_socket, mock_wandb_init, mock_wandb_finish, mock_wandb_save, tmp_path):
    """Test using WandbContext as a context manager."""
    mock_init, mock_run = mock_wandb_init

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path))
    global_cfg = SampleGlobalConfig()

    with WandbContext(cfg, global_cfg) as run:
        assert run == mock_run

    # Verify cleanup was called
    mock_wandb_finish.assert_called_once()


def test_socket_timeout_configuration(mock_socket, mock_wandb_init, mock_wandb_save, tmp_path):
    """Test that socket timeout is properly configured."""
    mock_init, mock_run = mock_wandb_init

    cfg = WandbConfig(enabled=True, project="test-project", entity="test-entity", data_dir=str(tmp_path))
    global_cfg = SampleGlobalConfig()

    # Track setdefaulttimeout calls
    timeout_calls = []

    def track_timeout(timeout):
        timeout_calls.append(timeout)

    with patch("socket.setdefaulttimeout", side_effect=track_timeout):
        ctx = WandbContext(cfg, global_cfg)
        ctx.__enter__()

    # Should have set timeout to 5 seconds for connection check
    assert 5 in timeout_calls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
