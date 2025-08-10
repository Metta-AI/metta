import torch

from metta.rl.rep.tc_loss import cosine_tc


def test_cosine_tc_monotonicity() -> None:
    b, k, d = 16, 4, 32
    tgts = torch.randn(b, k, d)
    preds_bad = -tgts.clone()
    preds_good = tgts.clone()
    loss_bad = cosine_tc(preds_bad, tgts, gamma=0.99)
    loss_good = cosine_tc(preds_good, tgts, gamma=0.99)
    assert loss_good < loss_bad
