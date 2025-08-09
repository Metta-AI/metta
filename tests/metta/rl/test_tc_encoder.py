import torch

from metta.rl.rep.encoder import TCEncoder


def test_tc_encoder_shapes() -> None:
    enc = TCEncoder(in_channels=30, latent_dim=256)
    x = torch.randn(32, 11, 11, 30)
    z = enc(x)
    assert z.shape == (32, 256)
    x2 = torch.randn(32, 30, 11, 11)
    z2 = enc(x2)
    assert z2.shape == (32, 256)
