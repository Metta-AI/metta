from __future__ import annotations

from metta.sweep.ray import ray_controller
from metta.tools import ray_sweep as ray_sweep_module
from metta.tools.ray_sweep import RaySweepTool


def test_ray_sweep_tool_passes_configuration_and_args(monkeypatch):
    captured: dict[str, object] = {}

    def fake_ray_sweep(*, sweep_config, search_space, ray_address):
        captured["sweep_config"] = sweep_config
        captured["search_space"] = search_space
        captured["ray_address"] = ray_address
        return "sentinel"

    monkeypatch.setattr(ray_sweep_module, "ray_sweep", fake_ray_sweep)

    tool = RaySweepTool(
        sweep_config=ray_controller.SweepConfig(sweep_id="exp", gpus_per_trial=2),
        search_space={"trainer.batch_size": 256},
    )

    result = tool.invoke({"ray_address": "ray://cluster"})

    assert result == "sentinel"
    assert captured["sweep_config"].sweep_id == "exp"
    assert captured["search_space"] == {"trainer.batch_size": 256}
    assert captured["ray_address"] == "ray://cluster"


def test_ray_sweep_tool_local_test_overrides(monkeypatch):
    captured: dict[str, object] = {}

    def fake_ray_sweep(*, sweep_config, search_space, ray_address):
        captured["sweep_config"] = sweep_config
        captured["search_space"] = search_space
        captured["ray_address"] = ray_address
        return None

    monkeypatch.setattr(ray_sweep_module, "ray_sweep", fake_ray_sweep)
    monkeypatch.setenv("RAY_ADDRESS", "ray://env-cluster")

    tool = RaySweepTool(
        sweep_config=ray_controller.SweepConfig(sweep_id="local-test", gpus_per_trial=4),
        search_space={},
        local_test=True,
    )

    tool.invoke({})

    assert captured["ray_address"] == "ray://env-cluster"
    assert captured["search_space"]["trainer.total_timesteps"] == 100_000
    assert tool.sweep_config.gpus_per_trial == 0
