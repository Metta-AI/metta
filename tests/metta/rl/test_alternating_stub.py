from types import SimpleNamespace

import torch

from metta.rl.rep.dynamics import LatentRollout
from metta.rl.rep.encoder import TCEncoder
from metta.rl.rep.momentum import ema_update
from metta.rl.rep.tc_loss import cosine_tc


# Light integration stub verifying TC step and encoder freezing.
def test_alternating_tc_then_freeze() -> None:
    cfg = SimpleNamespace(rep=SimpleNamespace(K=3, gamma=0.99, lambda_r=0.5, tau=0.998, reward_head=True))
    device = "cpu"

    encoder = TCEncoder(30, 32).to(device)
    enc_mom = TCEncoder(30, 32).to(device).eval()
    enc_mom.load_state_dict(encoder.state_dict())
    for p in enc_mom.parameters():
        p.requires_grad = False
    dyn = LatentRollout(latent_dim=32, act_dim=2, predict_reward=True).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(dyn.parameters()), lr=1e-3)

    b, t, a = 8, 10, 2
    obs = torch.randn(b, t, 11, 11, 30)
    acts = torch.randn(b, t, a)
    rews = torch.randn(b, t, 1)

    starts = torch.randint(0, t - cfg.rep.K - 1, (b,))

    def gather_t(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x[torch.arange(b), idx]

    def gather_range(x: torch.Tensor, idx: torch.Tensor, k: int) -> torch.Tensor:
        return torch.stack([x[torch.arange(b), idx + i] for i in range(k)], dim=1)

    o0 = gather_t(obs, starts)
    ofut = gather_range(obs, starts + 1, cfg.rep.K)
    afut = gather_range(acts, starts, cfg.rep.K)
    rgt = gather_range(rews, starts + 1, cfg.rep.K)

    z0 = encoder(o0)
    with torch.no_grad():
        z_tgt = enc_mom(ofut.view(-1, 11, 11, 30)).view(b, cfg.rep.K, -1)
    z_pred, r_pred = dyn.rollout(z0, afut, cfg.rep.K)
    loss = cosine_tc(z_pred, z_tgt, gamma=cfg.rep.gamma) + cfg.rep.lambda_r * torch.nn.functional.mse_loss(r_pred, rgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    ema_update(enc_mom, encoder, cfg.rep.tau)

    for p in encoder.parameters():
        p.requires_grad = False
    x = torch.randn(b, 11, 11, 30)
    with torch.no_grad():
        z = encoder(x)
    assert z.shape[0] == b
    assert not any(p.requires_grad for p in encoder.parameters())
