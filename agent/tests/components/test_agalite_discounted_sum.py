import pytest
import torch

from metta.agent.components.agalite_optimized import (
    _python_discounted_sum,
    discounted_sum,
)


def _make_inputs(device: torch.device):
    start = torch.randn(2, 3, device=device, dtype=torch.float32, requires_grad=True)
    x = torch.randn(5, 2, 3, device=device, dtype=torch.float32, requires_grad=True)
    discounts = torch.rand(5, 2, 3, device=device, dtype=torch.float32, requires_grad=True)
    return start, x, discounts


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_discounted_sum_matches_python(device):
    start, x, discounts = _make_inputs(device)

    ref = _python_discounted_sum(start, x, discounts)
    fused = discounted_sum(start, x, discounts)

    torch.testing.assert_close(fused, ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", [torch.device("cpu")])
def test_discounted_sum_gradients(device):
    start, x, discounts = _make_inputs(device)
    grad_signal = torch.randn_like(x)

    start_ref = start.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)
    discounts_ref = discounts.clone().detach().requires_grad_(True)

    out_ref = _python_discounted_sum(start_ref, x_ref, discounts_ref)
    loss_ref = (out_ref * grad_signal).sum()
    loss_ref.backward()

    start_fused = start.clone().detach().requires_grad_(True)
    x_fused = x.clone().detach().requires_grad_(True)
    discounts_fused = discounts.clone().detach().requires_grad_(True)

    out_fused = discounted_sum(start_fused, x_fused, discounts_fused)
    loss_fused = (out_fused * grad_signal).sum()
    loss_fused.backward()

    torch.testing.assert_close(start_fused.grad, start_ref.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(discounts_fused.grad, discounts_ref.grad, atol=1e-6, rtol=1e-5)
