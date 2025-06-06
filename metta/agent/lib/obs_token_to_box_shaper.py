import torch
import einops
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase

BOX_WIDTH = 11
BOX_HEIGHT = 11

class ObsTokenToBoxShaper(LayerBase):
    """
    xcxc
    This class does the following:
    1) permutes input observations from [B, H, W, C] or [B, TT, H, W, C] to [..., C, H, W]
    2) inspects tensor shapes, ensuring that input observations match expectations from the environment
    3) inserts batch size, TT, and B * TT into the tensor dict for certain other layers in the network to use
       if they need reshaping.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent is instantiated
    and never again. I.e., not when it is reloaded from a saved policy.
    """
    def __init__(self, obs_shape, feature_normalizations, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        # xcxc don't hardcode. Or at least provide a reason. This should match the view window.
        self.out_width = BOX_WIDTH
        self.out_height = BOX_HEIGHT
        self.num_layers = len(feature_normalizations)
        self._out_tensor_shape = [self.num_layers, self.out_width, self.out_height]

    def _forward(self, td: TensorDict):
        token_observations = td["x"]

        B = token_observations.shape[0]
        TT = 1
        if token_observations.dim() != 3:  # hardcoding for shape [B, M, 3]
            TT = token_observations.shape[1]
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")
        td["_BxTT_"] = B * TT

        # coords_byte contains x and y coordinates in a single byte (first 4 bits are x, last 4 bits are y)
        coords_byte = token_observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range, but we need to make them long for indexing)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).long()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).long()  # Shape: [B_TT, M]
        atr_indices = token_observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        atr_values = token_observations[..., 2].float()  # Shape: [B_TT, M]

        # In ObservationShaper we permute. Here, we create the observations pre-permuted.
        # We'd like to pre-create this as part of initialization, but we don't know the batch size or time steps at
        # that point.
        box_obs = torch.zeros(
            (B * TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        batch_indices = torch.arange(B * TT, device=token_observations.device).unsqueeze(-1).expand_as(atr_values)

        valid_tokens = coords_byte != 0xff
        box_obs[batch_indices[valid_tokens], atr_indices[valid_tokens], x_coord_indices[valid_tokens], y_coord_indices[valid_tokens]] = atr_values[valid_tokens]

        td["_TT_"] = TT
        td["_batch_size_"] = B
        td["_BxTT_"] = B * TT
        td[self._name] = box_obs
        return td
