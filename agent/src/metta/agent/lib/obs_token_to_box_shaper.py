import warnings

import torch
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class ObsTokenToBoxShaper(LayerBase):
    """
    This class consumes token observations and outputs a box observation.

    I.e., its output will be a tensor of shape [B*TT, num_layers, obs_width, obs_height].

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent is instantiated
    and never again. I.e., not when it is reloaded from a saved policy.

    Backward Compatibility:
    This class gracefully handles cases where the environment provides observation channels (attribute indices)
    that exceed what the policy was trained with. Any observation tokens with attribute indices >= num_layers
    will be filtered out with a warning. This allows policies trained with fewer observation channels to run
    in environments that have since added new observation types, though the policy will not benefit from the
    new information.
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
        token_observations = td["env_obs"]
        B_TT = td.batch_size.numel()

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
            (B_TT, self.num_layers, self.out_width, self.out_height),
            dtype=atr_values.dtype,
            device=token_observations.device,
        )
        # Create mask for valid tokens

        valid_tokens = coords_byte != 0xFF

        # Additional validation: ensure atr_indices are within valid range
        valid_atr = atr_indices < self.num_layers
        valid_mask = valid_tokens & valid_atr

        # Log warning for out-of-bounds indices
        invalid_atr_mask = valid_tokens & ~valid_atr
        if invalid_atr_mask.any():
            invalid_indices = atr_indices[invalid_atr_mask].unique()
            warnings.warn(
                f"Found observation attribute indices {sorted(invalid_indices.tolist())} "
                f">= num_layers ({self.num_layers}). These tokens will be ignored. "
                f"This may indicate the policy was trained with fewer observation channels.",
                stacklevel=2,
            )

        # Use scatter-based write to avoid multi-dim advanced indexing that triggers index_put_
        # Compute flattened spatial index and a combined index that encodes (layer, x, y)
        flat_spatial_index = x_coord_indices * self.out_height + y_coord_indices  # [B_TT, M]
        dim_per_layer = self.out_width * self.out_height
        combined_index = atr_indices * dim_per_layer + flat_spatial_index  # [B_TT, M]

        # Mask out invalid entries by directing them to index 0 with value 0
        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, atr_values, torch.zeros_like(atr_values))

        # Scatter values into a flattened buffer, then reshape to [B_TT, L, W, H]
        box_flat = torch.zeros(
            (B_TT, self.num_layers * dim_per_layer), dtype=atr_values.dtype, device=token_observations.device
        )
        box_flat.scatter_(1, safe_index, safe_values)
        box_obs = box_flat.view(B_TT, self.num_layers, self.out_width, self.out_height)

        td[self._name] = box_obs
        return td
