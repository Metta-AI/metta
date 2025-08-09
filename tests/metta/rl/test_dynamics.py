import torch

from metta.rl.rep.dynamics import LatentRollout


def test_latent_rollout_shapes() -> None:
    b, k, d, a = 8, 5, 64, 3
    z0 = torch.randn(b, d)
    acts = torch.randn(b, k, a)
    dyn = LatentRollout(latent_dim=d, act_dim=a, predict_reward=True)
    z, r = dyn.rollout(z0, acts, k)
    assert z.shape == (b, k, d)
    assert r is not None and r.shape == (b, k, 1)
