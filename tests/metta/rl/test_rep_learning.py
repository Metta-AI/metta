from __future__ import annotations

import torch

from metta.rl.losses import RepresentationLoss
from metta.rl.modules import DynamicsModel, ProjectionHead
from metta.rl.trainer import rep_step
from metta.rl.trainer_config import RepresentationLearningConfig


def _make_rep_loss(
    cfg: RepresentationLearningConfig | None = None,
) -> tuple[RepresentationLoss, ProjectionHead, DynamicsModel]:
    cfg = cfg or RepresentationLearningConfig()
    proj = ProjectionHead(4, 4)
    dyn = DynamicsModel(4, 2)
    return RepresentationLoss(proj, dyn, cfg), proj, dyn


def test_infonce_masking() -> None:
    cfg = RepresentationLearningConfig()
    rep_loss, _, _ = _make_rep_loss(cfg)
    z = torch.randn(5, 2, 4)
    mask = torch.ones(5, 2, dtype=torch.bool)
    mask[-1, 0] = False
    loss = rep_loss.compute_contrastive(z, mask)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_geometric_sampling_mean() -> None:
    cfg = RepresentationLearningConfig(alpha=0.8)
    rep_loss, _, _ = _make_rep_loss(cfg)
    device = torch.device("cpu")
    T = 1000
    means = []
    for _ in range(200):
        ks = rep_loss.sample_positive_offsets(T, device)
        means.append(ks.float().mean().item())
    mean_k = sum(means) / len(means)
    expected = cfg.alpha / (1 - cfg.alpha)
    assert abs(mean_k - expected) < 0.5


def test_tc_respects_dones() -> None:
    cfg = RepresentationLearningConfig(loss_tc_type="l2")
    rep_loss, _, _ = _make_rep_loss(cfg)
    z = torch.tensor([[[0.0]], [[1.0]], [[10.0]], [[11.0]]])
    mask = torch.tensor([[True], [True], [False], [True]])
    loss = rep_loss.compute_tc(z, mask)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_prediction_stopgrad() -> None:
    cfg = RepresentationLearningConfig(loss_pred_type="l2")
    rep_loss, _, _ = _make_rep_loss(cfg)
    z = torch.randn(3, 1, 4, requires_grad=True)
    a = torch.randn(3, 1, 2)
    mask = torch.ones(3, 1, dtype=torch.bool)
    loss = rep_loss.compute_pred(z, a, mask)
    loss.backward()
    assert z.grad is not None
    assert torch.allclose(z.grad[-1], torch.zeros_like(z.grad[-1]))
    assert z.grad[0].abs().sum() > 0


def test_rep_step_freezes_policy_heads() -> None:
    cfg = RepresentationLearningConfig()
    rep_loss, proj, dyn = _make_rep_loss(cfg)

    class SimplePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(3, 4)
            self.actor = torch.nn.Linear(4, 2)
            self.value = torch.nn.Linear(4, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            z = self.encoder(x)
            return z, self.actor(z), self.value(z)

    policy = SimplePolicy()
    policy_opt = torch.optim.Adam(policy.parameters(), lr=0.01)
    rep_opt = torch.optim.Adam(
        list(policy.encoder.parameters()) + list(proj.parameters()) + list(dyn.parameters()), lr=0.005
    )

    obs = torch.randn(4, 3)
    z, logits, value = policy(obs)
    target = torch.randint(0, 2, (4,))
    policy_loss = torch.nn.functional.cross_entropy(logits, target)
    value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), torch.zeros_like(value.squeeze(-1)))
    (policy_loss + value_loss).backward()
    policy_opt.step()
    policy_opt.zero_grad()

    actor_before = {k: v.clone() for k, v in policy.actor.named_parameters()}
    value_before = {k: v.clone() for k, v in policy.value.named_parameters()}
    encoder_before = {k: v.clone() for k, v in policy.encoder.named_parameters()}
    dyn_before = {k: v.clone() for k, v in dyn.named_parameters()}

    z = policy.encoder(obs)
    z_seq = z.view(2, 2, 4)
    a_seq = torch.nn.functional.one_hot(target, num_classes=2).float().view(2, 2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)
    rep_step(rep_loss, rep_opt, z_seq, a_seq, mask, steps=1)

    for k, v in policy.actor.named_parameters():
        assert torch.allclose(v, actor_before[k])
    for k, v in policy.value.named_parameters():
        assert torch.allclose(v, value_before[k])

    assert any(not torch.allclose(v, encoder_before[k]) for k, v in policy.encoder.named_parameters())
    assert any(not torch.allclose(v, dyn_before[k]) for k, v in dyn.named_parameters())
