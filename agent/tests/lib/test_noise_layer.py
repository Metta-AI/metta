import torch
from tensordict import TensorDict

from metta.agent.components.noise import NoiseLayer, NoiseLayerConfig


def _build_td(batch_size: int = 4, features: int = 6) -> TensorDict:
    data = torch.randn(batch_size, features)
    return TensorDict({"features": data}, batch_size=[batch_size])


def test_noise_layer_training_adds_noise() -> None:
    config = dict(in_key="features", out_key="noisy")

    torch.manual_seed(0)
    baseline = NoiseLayer(NoiseLayerConfig(**config, std=0.0))
    torch.manual_seed(0)
    noisy = NoiseLayer(NoiseLayerConfig(**config, std=0.3))

    td = _build_td()
    td_baseline = td.clone()
    td_noisy = td.clone()

    baseline.train()
    noisy.train()

    torch.manual_seed(123)
    base_out = baseline(td_baseline)["noisy"]
    torch.manual_seed(123)
    noisy_out = noisy(td_noisy)["noisy"]

    assert not torch.allclose(noisy_out, base_out)


def test_noise_layer_eval_disabled_by_default() -> None:
    config = dict(in_key="features", out_key="noisy")

    torch.manual_seed(1)
    baseline = NoiseLayer(NoiseLayerConfig(**config, std=0.0))
    torch.manual_seed(1)
    noisy = NoiseLayer(NoiseLayerConfig(**config, std=0.5))

    td = _build_td()
    td_baseline = td.clone()
    td_noisy = td.clone()

    baseline.eval()
    noisy.eval()

    torch.manual_seed(77)
    base_out = baseline(td_baseline)["noisy"]
    torch.manual_seed(77)
    noisy_out = noisy(td_noisy)["noisy"]

    assert torch.allclose(noisy_out, base_out)


def test_noise_layer_eval_enabled_when_requested() -> None:
    config = dict(in_key="features", out_key="noisy")

    torch.manual_seed(2)
    baseline = NoiseLayer(NoiseLayerConfig(**config, std=0.0))
    torch.manual_seed(2)
    noisy = NoiseLayer(NoiseLayerConfig(**config, std=0.5, noise_during_eval=True))

    td = _build_td()
    td_baseline = td.clone()
    td_noisy = td.clone()

    baseline.eval()
    noisy.eval()

    torch.manual_seed(99)
    base_out = baseline(td_baseline)["noisy"]
    torch.manual_seed(99)
    noisy_out = noisy(td_noisy)["noisy"]

    assert not torch.allclose(noisy_out, base_out)


def test_noise_layer_set_noise_updates_parameters() -> None:
    layer = NoiseLayer(NoiseLayerConfig(in_key="features", out_key="features", std=0.0))
    layer.train()

    td = _build_td()
    base_out = layer(td.clone())["features"]

    layer.set_noise(std=0.4, noise_during_eval=True)

    torch.manual_seed(5)
    noisy_out = layer(td.clone())["features"]

    assert not torch.allclose(noisy_out, base_out)
