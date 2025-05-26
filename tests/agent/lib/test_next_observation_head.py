import torch
from tensordict import TensorDict

from metta.agent.lib.nn_layer_library import NextObservationHead


def test_next_observation_head_outputs_mean_and_var():
    B = 4
    hidden_dim = 8
    obs_shape = [2, 2]
    cfg = {"name": "_next_obs_", "sources": [{"name": "hidden"}]}
    head = NextObservationHead(obs_shape, **cfg)
    head._in_tensor_shapes = [[hidden_dim]]
    head._initialize()

    td = TensorDict({"hidden": torch.randn(B, hidden_dim)}, batch_size=[B])
    head(td)
    mean, var = td["_next_obs_"]

    assert mean.shape == (B, *obs_shape)
    assert var.shape == (B, *obs_shape)
    assert torch.all(var > 0)
