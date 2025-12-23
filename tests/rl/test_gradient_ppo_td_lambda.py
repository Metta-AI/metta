import torch

from metta.rl.loss.ppo_critic import _td_lambda_error


def test_td_lambda_error_matches_simple_sum_without_terminals() -> None:
    values = torch.zeros((1, 4), dtype=torch.float32)
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    dones = torch.zeros_like(values)

    delta_lambda = _td_lambda_error(
        values=values,
        rewards=rewards,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected = torch.tensor([[6.0, 5.0, 3.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(delta_lambda, expected)


def test_td_lambda_error_resets_on_done() -> None:
    values = torch.zeros((1, 4), dtype=torch.float32)
    rewards = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    dones = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    delta_lambda = _td_lambda_error(
        values=values,
        rewards=rewards,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected = torch.tensor([[3.0, 2.0, 3.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(delta_lambda, expected)


def test_td_lambda_error_is_differentiable_wrt_values() -> None:
    values = torch.randn((2, 5), dtype=torch.float32, requires_grad=True)
    rewards = torch.randn((2, 5), dtype=torch.float32)
    dones = torch.zeros_like(values)

    delta_lambda = _td_lambda_error(
        values=values,
        rewards=rewards,
        dones=dones,
        gamma=0.9,
        gae_lambda=0.8,
    )
    loss = delta_lambda.sum()
    loss.backward()

    assert values.grad is not None
    assert float(values.grad.abs().sum().item()) > 0.0
