import einops
import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class ObsTokenToBoxShaper(LayerBase):
    """
    This class consumes token observations and outputs a box observation.

    I.e., its output will be a tensor of shape [B*TT, num_layers, obs_width, obs_height].

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent is instantiated
    and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, obs_shape, obs_width, obs_height, feature_normalizations, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        # These let us know the grid size from which tokens are being computed, and thus the shape of the box
        # observation.
        self.out_width = obs_width
        self.out_height = obs_height
        self.num_layers = max(feature_normalizations.keys()) + 1
        self._out_tensor_shape = [self.num_layers, self.out_width, self.out_height]

    def _forward(self, td: TensorDict):
        token_observations = td["x"]

        B = token_observations.shape[0]
        TT = 1
        if token_observations.dim() != 3:  # hardcoding for shape [B, M, 3]
            TT = token_observations.shape[1]
            token_observations = einops.rearrange(token_observations, "b t m c -> (b t) m c")
        td["_BxTT_"] = B * TT

        assert token_observations.shape[-1] == 3, f"Expected 3 channels per token. Got shape {token_observations.shape}"

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

        valid_tokens = coords_byte != 0xFF

        # Add bounds checking for feature indices
        if valid_tokens.any():
            max_atr_idx = atr_indices[valid_tokens].max().item()
            if max_atr_idx >= self.num_layers:
                raise IndexError(
                    f"Feature index {max_atr_idx} is out of bounds for tensor with "
                    f"{self.num_layers} layers. This likely means the environment is "
                    f"generating features that weren't present when the agent was created. "
                    f"Check if recipe_details_obs or resource_rewards are enabled in the "
                    f"environment but not in the agent's feature_normalizations."
                )

        box_obs[
            batch_indices[valid_tokens],
            atr_indices[valid_tokens],
            x_coord_indices[valid_tokens],
            y_coord_indices[valid_tokens],
        ] = atr_values[valid_tokens]

        td["_TT_"] = TT
        td["_batch_size_"] = B
        td["_BxTT_"] = B * TT
        td[self._name] = box_obs
        return td
