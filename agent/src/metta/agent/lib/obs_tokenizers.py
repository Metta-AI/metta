from typing import Tuple

import einops
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.nn_layer_library import LayerBase


class ObsTokenPadStrip(LayerBase):
    """
    This is a top-level layer that grabs environment token observations and strips them of padding, returning a tensor
    of shape [B, M, 3] where M is the maximum number of tokens in _any_ sequence in the batch. It also adds batch size,
    TT, and B * TT to the tensor dict for downstream layers to use, if necessary.
    For clarification it does not strip all padding. It finds the sequence (out of all sequences in the batch) with the
    most dense tokens, gets that index, and then slices the obs tensor at that point. That means that it perfectly
    eliminates the padding tokens from the the sequence with the fewest padding tokens and also removes that number of
    padding tokens from all other sequences. In practice, the sequence with the most dense tokens can have many more
    dense tokens than the average sequence so there is room for improvement by computing attention over ragged tensors.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._obs_shape = obs_shape
        self._M = obs_shape[0]
        # Initialize feature remapping as identity by default
        self.register_buffer("feature_id_remap", torch.arange(256, dtype=torch.uint8))
        self._remapping_active = False

    def _make_net(self) -> None:
        # unfortunately, we can't know the output shape's 0th dim (seq length) until we see the data. however,
        # downstream layers should not need to know this in initializing any of their learnable parameters.
        # so we just set it to 0 for now.
        self._out_tensor_shape = [0, 3]

    def update_feature_remapping(self, feature_id_remap: torch.Tensor):
        """
        Update the feature ID remapping table.

        Args:
            feature_id_remap: A 256-element tensor where index is new_id and value is original_id
        """
        self.register_buffer("feature_id_remap", feature_id_remap.to(self.feature_id_remap.device))
        identity = torch.arange(256, dtype=torch.uint8, device=self.feature_id_remap.device)
        self._remapping_active = not torch.equal(self.feature_id_remap, identity)

    def _forward(self, td: TensorDict) -> TensorDict:
        # [B, M, 3] the 3 vector is: coord (unit8), attr_idx, attr_val
        observations = td["env_obs"]

        B = observations.shape[0]
        TT = 1
        td["_B_"] = B
        td["_TT_"] = TT
        if observations.dim() != 3:  # hardcoding for shape [B, M, 3]
            TT = observations.shape[1]
            td["_TT_"] = TT
            observations = einops.rearrange(observations, "b t h c -> (b t) h c")
        td["_BxTT_"] = B * TT

        # Apply feature remapping if active
        if self._remapping_active:
            observations = observations.clone()
            feature_ids = observations[..., 1].long()
            remapped_ids = self.feature_id_remap[feature_ids]
            observations[..., 1] = remapped_ids

        coords = observations[..., 0]
        obs_mask = coords == 255  # important! true means mask me

        # find each row's flip‐point ie when it goes from dense to padding
        flip_pts = obs_mask.int().argmax(dim=1)  # shape [B]

        # find the global max flip‐point as a 0‐d tensor
        max_flip = flip_pts.max()
        if max_flip == 0:
            max_flip = max_flip + self._M  # hack to avoid 0. should instead grab

        # build a 1‐D "positions" row [0,1,2,…,L−1]
        positions = torch.arange(self._M, device=obs_mask.device)

        # make a boolean column mask: keep all columns strictly before max_flip
        keep_cols = positions < max_flip  # shape [L], dtype=torch.bool

        observations = observations[:, keep_cols]  # shape [B, max_flip]
        obs_mask = obs_mask[:, keep_cols]

        td[self._name] = observations
        td["obs_mask"] = obs_mask
        return td


class ObsAttrValNorm(LayerBase):
    """Normalizes attr values based on the attr index."""

    def __init__(
        self,
        feature_normalizations: list[float],
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._feature_normalizations = list(feature_normalizations)
        self._max_embeds = 256

    def _make_net(self) -> None:
        self._out_tensor_shape = [0, 3]
        # Create a tensor for feature normalizations
        # We need to handle the case where attr_idx might be 0 (padding) or larger than defined normalizations.
        # Assuming max attr_idx is 256 (same as attr_embeds size - 1 for padding_idx).
        # Initialize with 1.0 to avoid division by zero for unmapped indices.
        norm_tensor = torch.ones(self._max_embeds, dtype=torch.float32)
        for i, val in enumerate(self._feature_normalizations):
            if i < len(norm_tensor):  # Ensure we don't go out of bounds
                norm_tensor[i] = val
            else:
                raise ValueError(f"Feature normalization {val} is out of bounds for Embedding layer size {i}")
        self.register_buffer("_norm_factors", norm_tensor)
        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        observations = td[self._sources[0]["name"]]

        attr_indices = observations[..., 1].long()
        norm_factors = self._norm_factors[attr_indices]
        observations = observations.clone()
        observations[..., 2] = observations[..., 2] / norm_factors

        td[self._name] = observations
        return td


class ObsAttrCoordEmbed(LayerBase):
    """Embeds attr index as, separately embeds coords, then adds them together. Finally concatenate attr value to the
    end of the embedding. Learnable coord embeddings have performed worse than Fourier features as of 6/16/2025."""

    def __init__(
        self,
        attr_embed_dim: int,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._attr_embed_dim = attr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._feat_dim = self._attr_embed_dim + self._value_dim
        self._max_embeds = 256

    def _make_net(self) -> None:
        self._out_tensor_shape = [0, self._feat_dim]

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        observations = td[self._sources[0]["name"]]

        coord_indices = observations[..., 0].long()
        coord_pair_embedding = self._coord_embeds(coord_indices)

        attr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding

        attr_embeds = self._attr_embeds(attr_indices)  # [B_TT, M, embed_dim]

        combined_embeds = attr_embeds + coord_pair_embedding

        attr_values = observations[..., 2].float()  # Shape: [B_TT, M]
        attr_values = einops.rearrange(attr_values, "... -> ... 1")

        # Assemble feature vectors
        # feat_vectors will have shape [B_TT, M, _feat_dim] where _feat_dim = _embed_dim + _value_dim
        feat_vectors = torch.empty(
            (*attr_embeds.shape[:-1], self._feat_dim),
            dtype=attr_embeds.dtype,
            device=attr_embeds.device,
        )
        # Combined embedding portion
        feat_vectors[..., : self._attr_embed_dim] = combined_embeds
        feat_vectors[..., self._attr_embed_dim : self._attr_embed_dim + self._value_dim] = attr_values

        td[self._name] = feat_vectors
        return td


class ObsAttrEmbedFourier(LayerBase):
    """An alternate to ObsAttrCoordEmbed that concatenates attr embeds w coord representation as Fourier features."""

    def __init__(
        self,
        attr_embed_dim: int,
        num_freqs: int = 3,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._attr_embed_dim = attr_embed_dim  # Dimension of attribute embeddings
        self._num_freqs = num_freqs  # fourier feature frequencies
        self._coord_rep_dim = 4 * self._num_freqs  # x, y, sin, cos for each freq
        self._value_dim = 1
        self._feat_dim = self._attr_embed_dim + self._coord_rep_dim + self._value_dim
        self._max_embeds = 256
        self._mu = 11.0  # hardcoding 11 as the max coord value for now (range 0-10). can grab from mettagrid_env.py

    def _make_net(self) -> None:
        self._out_tensor_shape = [0, self._feat_dim]

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        self.register_buffer("frequencies", 2.0 ** torch.arange(self._num_freqs))

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        observations = td[self._sources[0]["name"]]

        # [B, M, 3] the 3 vector is: coord (unit8), attr_idx, attr_val
        attr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        attr_embeds = self._attr_embeds(attr_indices)  # [B_TT, M, embed_dim]

        # Assemble feature vectors
        # Pre-allocating the tensor and filling it avoids multiple `torch.cat` calls,
        # which can be more efficient on GPU.
        feat_vectors = torch.empty(
            (*attr_embeds.shape[:-1], self._feat_dim),
            dtype=attr_embeds.dtype,
            device=attr_embeds.device,
        )
        feat_vectors[..., : self._attr_embed_dim] = attr_embeds

        # coords_byte contains x and y coordinates in a single byte (first 4 bits are x, last 4 bits are y)
        coords_byte = observations[..., 0].to(torch.uint8)

        # Extract x and y coordinate indices (0-15 range)
        x_coord_indices = ((coords_byte >> 4) & 0x0F).float()  # Shape: [B_TT, M]
        y_coord_indices = (coords_byte & 0x0F).float()  # Shape: [B_TT, M]

        # Normalize coordinates to [-1, 1] based on the data range [0, 10]
        x_coords_norm = x_coord_indices / (self._mu - 1.0) * 2.0 - 1.0
        y_coords_norm = y_coord_indices / (self._mu - 1.0) * 2.0 - 1.0

        # Expand dims for broadcasting with frequencies
        x_coords_norm = x_coords_norm.unsqueeze(-1)  # [B_TT, M, 1]
        y_coords_norm = y_coords_norm.unsqueeze(-1)  # [B_TT, M, 1]

        # Get frequencies and reshape for broadcasting
        # self.frequencies is [f], reshape to [1, 1, f]
        frequencies = self.get_buffer("frequencies").view(1, 1, -1)

        # Compute scaled coordinates for Fourier features
        x_scaled = x_coords_norm * frequencies
        y_scaled = y_coords_norm * frequencies

        # Compute and place Fourier features directly into the feature vector
        offset = self._attr_embed_dim
        feat_vectors[..., offset : offset + self._num_freqs] = torch.cos(x_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.sin(x_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.cos(y_scaled)
        offset += self._num_freqs
        feat_vectors[..., offset : offset + self._num_freqs] = torch.sin(y_scaled)

        attr_values = observations[..., 2].float()  # Shape: [B_TT, M]

        # Place normalized attribute values in the feature vector
        feat_vectors[..., self._attr_embed_dim + self._coord_rep_dim :] = einops.rearrange(attr_values, "... -> ... 1")

        td[self._name] = feat_vectors
        return td


class ObsAttrCoordValueEmbed(LayerBase):
    """An experiment that embeds attr value as a categorical variable. Using a normalization layer is not
    recommended."""

    def __init__(
        self,
        attr_embed_dim: int,
        **cfg,
    ) -> None:
        super().__init__(**cfg)
        self._attr_embed_dim = attr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._max_embeds = 256

    def _make_net(self) -> None:
        self._out_tensor_shape = [0, self._attr_embed_dim]

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        self._val_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._val_embeds.weight, std=0.02)

        return None

    def _forward(self, td: TensorDict) -> TensorDict:
        # [B, M, 3] the 3 vector is: coord (unit8), attr_idx, attr_val
        observations = td[self._sources[0]["name"]]

        # The first element of an observation is the coordinate, which can be used as a direct index for embedding.
        coord_indices = observations[..., 0].long()
        coord_pair_embedding = self._coord_embeds(coord_indices)

        attr_indices = observations[..., 1].long()  # Shape: [B_TT, M], ready for embedding
        attr_embeds = self._attr_embeds(attr_indices)  # [B_TT, M, embed_dim]

        # The attribute value is treated as a categorical variable for embedding.
        val_indices = observations[..., 2].long()
        # Clip values to be within the embedding range to avoid errors, e.g. if a value is 256 for _max_embeds=256
        val_indices = torch.clamp(val_indices, 0, self._max_embeds - 1)
        val_embeds = self._val_embeds(val_indices)

        combined_embeds = attr_embeds + coord_pair_embedding + val_embeds

        td[self._name] = combined_embeds
        return td


# TODO: try scaling attr index embed by normalized value
# TODO: try embed attr value but concat to attr index embed using list slicing (not torch.cat). Also cat fourier reps
