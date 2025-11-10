from __future__ import annotations

from types import SimpleNamespace

from metta.sweep.ray import ray_controller as rc


class DummyOptunaSearch:
    instances: list["DummyOptunaSearch"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        DummyOptunaSearch.instances.append(self)


class DummyConcurrencyLimiter:
    instances: list["DummyConcurrencyLimiter"] = []

    def __init__(self, search_alg, max_concurrent):
        self.search_alg = search_alg
        self.max_concurrent = max_concurrent
        DummyConcurrencyLimiter.instances.append(self)


class DummyTuneConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyRunConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyTuner:
    last_init: dict[str, object] | None = None
    fit_called: bool = False

    def __init__(self, trainable, *, tune_config, run_config, param_space):
        DummyTuner.last_init = {
            "trainable": trainable,
            "tune_config": tune_config,
            "run_config": run_config,
            "param_space": param_space,
        }

    def fit(self):
        DummyTuner.fit_called = True


def test_ray_sweep_controller_uses_cluster_resources(monkeypatch):
    init_calls: list[dict[str, object]] = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)

    monkeypatch.setattr(rc, "init", fake_init)
    monkeypatch.setattr(rc, "ray", SimpleNamespace(cluster_resources=lambda: {"CPU": 32.0, "GPU": 8.0}))

    with_resources_calls: list[dict[str, float]] = []

    def dummy_with_resources(trainable, resources):
        with_resources_calls.append(resources)
        return f"wrapped({trainable})"

    failure_configs: list[dict[str, object]] = []

    class DummyFailureConfig:
        def __init__(self, **kwargs):
            failure_configs.append(kwargs)

    dummy_tune = SimpleNamespace(with_resources=dummy_with_resources, FailureConfig=DummyFailureConfig)

    monkeypatch.setattr(rc, "tune", dummy_tune)
    monkeypatch.setattr(rc, "OptunaSearch", DummyOptunaSearch)
    monkeypatch.setattr(rc, "ConcurrencyLimiter", DummyConcurrencyLimiter)
    monkeypatch.setattr(rc, "TuneConfig", DummyTuneConfig)
    monkeypatch.setattr(rc, "RunConfig", DummyRunConfig)
    monkeypatch.setattr(rc, "Tuner", DummyTuner)
    monkeypatch.setattr(rc, "metta_train_fn", "train_fn")
    monkeypatch.setenv("METTA_DETECTED_GPUS_PER_NODE", "4")

    sweep_config = rc.SweepConfig(
        sweep_id="demo",
        cpus_per_trial="auto",
        gpus_per_trial="auto",
        max_concurrent_trials=4,
        max_failures_per_trial=2,
        fail_fast=True,
        num_samples=5,
    )

    rc.ray_sweep(search_space={"trainer.batch_size": 42}, sweep_config=sweep_config, ray_address="ray://cluster")

    assert init_calls and init_calls[0]["address"] == "ray://cluster"
    assert init_calls[0]["runtime_env"] == {"working_dir": None}

    assert with_resources_calls[0] == {"cpu": float(7), "gpu": float(4)}

    assert DummyOptunaSearch.instances[0].kwargs == {"metric": "reward", "mode": "max"}
    assert DummyConcurrencyLimiter.instances[0].max_concurrent == sweep_config.max_concurrent_trials
    assert failure_configs[0] == {"max_failures": 2, "fail_fast": True}

    assert DummyTuner.last_init is not None
    assert DummyTuner.last_init["trainable"] == "wrapped(train_fn)"
    assert DummyTuner.fit_called

    param_space = DummyTuner.last_init["param_space"]
    assert param_space["params"] == {"trainer.batch_size": 42}
    assert param_space["sweep_config"]["sweep_id"] == "demo"

    assert sweep_config.gpus_per_trial == 4
    assert sweep_config.cpus_per_trial == 7
