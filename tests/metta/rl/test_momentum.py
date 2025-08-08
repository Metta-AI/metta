import torch

from metta.rl.rep.encoder import TCEncoder
from metta.rl.rep.momentum import ema_update


def test_ema_update_moves_towards_online() -> None:
    online = TCEncoder(30, 16)
    target = TCEncoder(30, 16)
    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = sum((t - o).abs().sum().item() for t, o in zip(target.parameters(), online.parameters(), strict=False))
    ema_update(target, online, tau=0.9)
    after = sum((t - o).abs().sum().item() for t, o in zip(target.parameters(), online.parameters(), strict=False))
    assert after < before
