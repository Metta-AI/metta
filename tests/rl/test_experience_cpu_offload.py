import torch
from torchrl.data import Composite, UnboundedContinuous

from metta.rl.training.experience import Experience


def _make_basic_experience_spec() -> Composite:
    return Composite(
        rewards=UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32),
        dones=UnboundedContinuous(shape=torch.Size([]), dtype=torch.float32),
    )


def test_experience_uses_storage_device_for_buffers() -> None:
    spec = _make_basic_experience_spec()
    compute_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    storage_device = torch.device("cpu")

    experience = Experience(
        total_agents=2,
        batch_size=4,
        bptt_horizon=2,
        minibatch_size=2,
        max_minibatch_size=2,
        experience_spec=spec,
        device=compute_device,
        storage_device=storage_device,
        pin_memory=torch.cuda.is_available(),
    )

    assert experience.buffer.device == storage_device
    assert experience.ep_lengths.device == storage_device
    assert experience.ep_indices.device == storage_device

    if torch.cuda.is_available():
        assert experience.pin_memory is True
        assert experience.buffer.is_pinned()
    else:
        assert experience.pin_memory is False


def test_experience_defaults_to_compute_device_when_no_storage_device() -> None:
    spec = _make_basic_experience_spec()
    compute_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    experience = Experience(
        total_agents=2,
        batch_size=4,
        bptt_horizon=2,
        minibatch_size=2,
        max_minibatch_size=2,
        experience_spec=spec,
        device=compute_device,
    )

    assert experience.buffer.device == compute_device
