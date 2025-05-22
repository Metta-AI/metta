import os
import socket
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import wandb
from omegaconf import OmegaConf

from metta.util.logging import setup_mettagrid_logger
from metta.util.wandb.wandb_context import WandbConfigOff, WandbConfigOn, WandbContext

logger = setup_mettagrid_logger("Test")


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    # Bypass version check by patching the check_wandb_version function
    monkeypatch.setattr("metta.util.wandb.wandb_context.check_wandb_version", lambda: True)

    # Patch wandb.save to no-op
    monkeypatch.setattr(wandb, "save", lambda *args, **kwargs: None)

    # Dummy socket to bypass real network calls
    class DummySock:
        def settimeout(self, timeout):
            pass

        def connect(self, addr):
            pass

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: DummySock())

    # Patch file operations that might create config.yaml
    # This prevents any yaml file writes
    original_open = open

    def patched_open(filename, mode="r", *args, **kwargs):
        if isinstance(filename, str) and filename.endswith(".yaml") and "w" in mode:
            # Return a mock file object for yaml writes
            mock_file = MagicMock()
            mock_file.__enter__ = lambda self: mock_file
            mock_file.__exit__ = lambda self, *args: None
            return mock_file
        return original_open(filename, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", patched_open)

    # Also patch OmegaConf.save if it's being used
    monkeypatch.setattr(OmegaConf, "save", lambda *args, **kwargs: None)

    # Change to a temporary directory to isolate file operations
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    yield

    # Restore original directory
    os.chdir(original_cwd)


@dataclass
class DummyRun:
    id: str
    job_type: str
    project: str
    entity: str
    config: dict
    group: str
    allow_val_change: bool
    name: str
    monitor_gym: bool
    save_code: bool
    resume: bool
    tags: list[str]
    settings: wandb.Settings


@pytest.fixture
def dummy_init(monkeypatch):
    def create_dummy_run(*args, **kwargs):
        # Create a dummy run with default values for missing fields
        return DummyRun(
            id=kwargs.get("id", "default_id"),
            job_type=kwargs.get("job_type", "default_job"),
            project=kwargs.get("project", "default_project"),
            entity=kwargs.get("entity", "default_entity"),
            config=kwargs.get("config", {}),
            group=kwargs.get("group", "default_group"),
            allow_val_change=kwargs.get("allow_val_change", True),
            name=kwargs.get("name", "default_name"),
            monitor_gym=kwargs.get("monitor_gym", True),
            save_code=kwargs.get("save_code", True),
            resume=kwargs.get("resume", True),
            tags=kwargs.get("tags", []),
            settings=kwargs.get("settings", wandb.Settings()),
        )

    monkeypatch.setattr(wandb, "init", create_dummy_run)
    yield


def test_enter_disabled_does_not_init(monkeypatch):
    # Prepare disabled config
    cfg_off = OmegaConf.create(dict(enabled=False))
    # Spy on wandb.init
    init_called = False

    def fake_init(*args, **kwargs):
        nonlocal init_called
        init_called = True

    monkeypatch.setattr(wandb, "init", fake_init)

    ctx = WandbContext(cfg_off, global_cfg={"foo": "bar"})
    run = ctx.__enter__()
    assert run is None, "Expected no run when enabled=False"
    assert not init_called, "wandb.init should not be called when disabled"


def test_structured_config(monkeypatch, dummy_init):
    # Prepare config that's already validated
    cfg_off = WandbConfigOff(enabled=False)

    ctx = WandbContext(cfg_off, OmegaConf.create())
    run = ctx.__enter__()
    assert run is None

    assert ctx.cfg == cfg_off


def test_run_fields(monkeypatch, dummy_init, tmp_path):
    # Prepare enabled config
    cfg_on = OmegaConf.create(
        dict(
            enabled=True,
            project="proj",
            entity="ent",
            group="grp",
            name="nm",
            run_id="id",
            data_dir=str(tmp_path),
            job_type="jt",
        )
    )
    global_cfg = OmegaConf.create({"a": 1})

    ctx = WandbContext(cfg_on, global_cfg)
    run = ctx.__enter__()

    assert run is not None
    assert isinstance(run, DummyRun)

    # Check fields
    assert run.id == "id"
    assert run.job_type == "jt"
    assert run.project == "proj"
    assert run.entity == "ent"
    assert run.config == global_cfg
    assert run.group == "grp"
    assert run.name == "nm"
    assert run.resume is True
    assert run.monitor_gym is True
    assert run.save_code is True


def test_exit_finishes_run(monkeypatch, dummy_init):
    # Prepare enabled config
    cfg_on = WandbConfigOn(
        enabled=True,
        project="p",
        entity="e",
        group="g",
        name="n",
        run_id="r",
        data_dir=os.getcwd(),
        job_type="j",
    )

    # Spy on finish
    finished = False

    def fake_finish():
        nonlocal finished
        finished = True

    monkeypatch.setattr(wandb, "finish", fake_finish)

    ctx = WandbContext(cfg_on, global_cfg=OmegaConf.create({}))
    _ = ctx.__enter__()
    ctx.__exit__(None, None, None)
    assert finished, "wandb.finish should be called on exit"


def test_no_config_yaml_created(tmp_path):
    """Verify that running tests doesn't create config.yaml files"""
    # Run a simple test that might trigger config creation
    # Include all required fields for WandbConfigOn
    cfg_on = OmegaConf.create(
        dict(
            enabled=True,
            project="test_proj",
            entity="test_ent",
            group="test_group",
            name="test_name",
            run_id="test_run_id",
            data_dir=str(tmp_path),
            job_type="test_job",
        )
    )

    # Check that no config.yaml exists before
    config_file = tmp_path / "config.yaml"
    assert not config_file.exists(), "config.yaml should not exist before test"

    # This normally creates a config file, but our patches should prevent it
    _ctx = WandbContext(cfg_on, OmegaConf.create({"test": "data"}))

    # Check that no config.yaml was created
    assert not config_file.exists(), "config.yaml should not be created during test"
