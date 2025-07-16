# import torch
# import torch.nn as nn
# from einops import rearrange
# from tensordict import TensorDict

# from metta.agent.lib.metta_layer import LayerBase

# class Transformer(LayerBase):
#     def __init__(self, obs_shape, hidden_size, **cfg):
#         super().__init__(**cfg)
#         self._obs_shape = list(obs_shape)
#         self.hidden_size = hidden_size
#         self.num_layers = self._nn_params["num_layers"]

#     def _make_net(self):
#         self._out_tensor_shape = [self.hidden_size]
#         return nn.Transformer(d_model=self.hidden_size, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

#     def forward(self, td: TensorDict):
