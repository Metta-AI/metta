import socket
from dataclasses import dataclass

import pytest
import wandb
from omegaconf import OmegaConf

from metta.common.util.logging_helpers import setup_mettagrid_logger
from metta.common.wandb.wandb_context import WandbConfigOff, WandbConfigOn, WandbContext

logger = setup_mettagrid_logger("Test")


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
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
    notes: str | None
    settings: wandb.Settings


@pytest.fixture
def dummy_init(monkeypatch):
    monkeypatch.setattr(wandb, "init", lambda *args, **kwargs: DummyRun(*args, **kwargs))
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


def test_tags_and_notes(monkeypatch, dummy_init, tmp_path):
    cfg_on = OmegaConf.create(
        dict(
            enabled=True,
            project="p",
            entity="e",
            group="g",
            name="n",
            run_id="r",
            data_dir=str(tmp_path),
            job_type="j",
            tags=["a", "b"],
            notes="hello",
        )
    )

    ctx = WandbContext(cfg_on, OmegaConf.create({}))
    run = ctx.__enter__()

    assert run is not None
    assert run.tags == ["a", "b", "user:unknown"]
    assert run.notes == "hello"


def test_exit_finishes_run(monkeypatch, dummy_init, tmp_path):
    # Prepare enabled config
    cfg_on = WandbConfigOn(
        enabled=True,
        project="p",
        entity="e",
        group="g",
        name="n",
        run_id="r",
        data_dir=str(tmp_path),
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
