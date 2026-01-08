import torch
from tensordict import TensorDict

from metta.agent.components.noise import NoiseLayer, NoiseLayerConfig

BASE_CONFIG = dict(in_key="features", out_key="noisy")


def _build_td(batch_size: int = 4, features: int = 6) -> TensorDict:
    data = torch.randn(batch_size, features)
    return TensorDict({"features": data}, batch_size=[batch_size])


def _make_layer(std: float, *, seed: int, noise_during_eval: bool = False) -> NoiseLayer:
    torch.manual_seed(seed)
    return NoiseLayer(NoiseLayerConfig(**BASE_CONFIG, std=std, noise_during_eval=noise_during_eval))


def _run(layer: NoiseLayer, td: TensorDict, *, seed: int, out_key: str | None = None) -> torch.Tensor:
    torch.manual_seed(seed)
    key = BASE_CONFIG["out_key"] if out_key is None else out_key
    return layer(td.clone())[key]


def test_noise_layer_training_adds_noise() -> None:
    baseline = _make_layer(0.0, seed=0)
    noisy = _make_layer(0.3, seed=0)
    td = _build_td()

    baseline.train()
    noisy.train()

    base_out = _run(baseline, td, seed=123)
    noisy_out = _run(noisy, td, seed=123)

    assert not torch.allclose(noisy_out, base_out)


def test_noise_layer_eval_disabled_by_default() -> None:
    baseline = _make_layer(0.0, seed=1)
    noisy = _make_layer(0.5, seed=1)
    td = _build_td()

    baseline.eval()
    noisy.eval()

    base_out = _run(baseline, td, seed=77)
    noisy_out = _run(noisy, td, seed=77)

    assert torch.allclose(noisy_out, base_out)


def test_noise_layer_eval_enabled_when_requested() -> None:
    baseline = _make_layer(0.0, seed=2)
    noisy = _make_layer(0.5, seed=2, noise_during_eval=True)
    td = _build_td()

    baseline.eval()
    noisy.eval()

    base_out = _run(baseline, td, seed=99)
    noisy_out = _run(noisy, td, seed=99)

    assert not torch.allclose(noisy_out, base_out)


def test_noise_layer_set_noise_updates_parameters() -> None:
    layer = NoiseLayer(NoiseLayerConfig(in_key="features", out_key="features", std=0.0))
    layer.train()

    td = _build_td()
    base_out = layer(td.clone())["features"]

    layer.set_noise(std=0.4, noise_during_eval=True)

    noisy_out = _run(layer, td, seed=5, out_key="features")

    assert not torch.allclose(noisy_out, base_out)
