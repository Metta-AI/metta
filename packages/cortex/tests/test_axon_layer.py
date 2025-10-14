from __future__ import annotations

import torch
from cortex.cells.core import AxonLayer
from tensordict import TensorDict


@torch.inference_mode(False)
def test_axon_layer_shapes_and_grads_sequence() -> None:
    B, T, Hin, Hout = 4, 16, 32, 48
    layer = AxonLayer(Hin, Hout)

    x = torch.randn(B, T, Hin, dtype=torch.float32)
    target = torch.randn(B, T, Hout, dtype=torch.float32)
    state = TensorDict({}, batch_size=[B])

    y = layer(x, state=state, resets=None)
    assert y.shape == (B, T, Hout)

    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    # Ensure some parameters received gradients and they are finite
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all().item() for g in grads if g is not None)


def test_axon_layer_state_auto_parent_and_reset() -> None:
    B, T, Hin, Hout = 3, 8, 16, 16
    layer = AxonLayer(Hin, Hout, name="proj")

    x = torch.randn(B, T, Hin)
    state = TensorDict({}, batch_size=[B])

    # First call creates state under state["axon"]["proj"]
    y = layer(x, state=state, resets=None)
    assert y.shape == (B, T, Hout)
    assert "axon" in state.keys()
    axon_group = state.get("axon")
    assert axon_group is not None and "proj" in axon_group.keys()

    sub = axon_group.get("proj")
    assert sub is not None
    # Expect AxonsCell base keys present
    assert "hc1" in sub.keys() and "hc2" in sub.keys()
    assert sub["hc1"].shape == (B, Hin)

    # Reset first batch element and verify it zeroes the substate row 0
    mask = torch.zeros(B, dtype=torch.float32)
    mask[0] = 1.0
    layer.reset_state(mask, state)
    sub_after = state.get("axon").get("proj")  # type: ignore[union-attr]
    assert torch.allclose(sub_after["hc1"][0], torch.zeros_like(sub_after["hc1"][0]))
    assert torch.allclose(sub_after["hc2"][0], torch.zeros_like(sub_after["hc2"][0]))


def test_axon_layer_local_state_step_and_batch_change() -> None:
    B1, B2, H = 5, 7, 24
    layer = AxonLayer(H, H)

    x1 = torch.randn(B1, H)
    y1 = layer(x1, state=None, resets=None)
    assert y1.shape == (B1, H)

    # Different batch size should re-initialize internal state without error
    x2 = torch.randn(B2, H)
    y2 = layer(x2, state=None, resets=None)
    assert y2.shape == (B2, H)
