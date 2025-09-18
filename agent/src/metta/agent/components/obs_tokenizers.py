import einops
import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class ObsAttrCoordEmbedConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "obs_attr_coord_embed"
    attr_embed_dim: int

    def make_component(self, env=None):
        return ObsAttrCoordEmbed(config=self)


class ObsAttrCoordEmbed(nn.Module):
    """Embeds attr index as, separately embeds coords, then adds them together. Finally concatenate attr value to the
    end of the embedding. Learnable coord embeddings have performed worse than Fourier features as of 6/16/2025."""

    def __init__(
        self,
        config: ObsAttrCoordEmbedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self._attr_embed_dim = self.config.attr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._feat_dim = self._attr_embed_dim + self._value_dim
        self._max_embeds = 256

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        return None

    def forward(self, td: TensorDict) -> TensorDict:
        observations = td[self.config.in_key]

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

        td[self.config.out_key] = feat_vectors
        return td


class ObsAttrEmbedFourierConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "obs_attr_embed_fourier"
    attr_embed_dim: int = 12
    num_freqs: int = 6

    def make_component(self, env=None):
        return ObsAttrEmbedFourier(config=self)


class ObsAttrEmbedFourier(nn.Module):
    """An alternate to ObsAttrCoordEmbed that concatenates attr embeds w coord representation as Fourier features.

    The output feature dimension is calculated as:
    `output_dim = attr_embed_dim + (4 * num_freqs) + 1`

    Where:
    - `attr_embed_dim` is the dimension of the attribute embedding.
    - `num_freqs` is the number of frequencies for the Fourier features. The coordinate
      representation dimension is `4 * num_freqs` because we have sin and cos for
      both x and y coordinates for each frequency.
    - `1` is for the scalar attribute value that is concatenated at the end.
    """

    def __init__(self, config: ObsAttrEmbedFourierConfig) -> None:
        super().__init__()
        self.config = config
        self._attr_embed_dim = self.config.attr_embed_dim  # Dimension of attribute embeddings
        self._num_freqs = self.config.num_freqs  # fourier feature frequencies
        self._coord_rep_dim = 4 * self._num_freqs  # x, y, sin, cos for each freq
        self._value_dim = 1
        self._feat_dim = self._attr_embed_dim + self._coord_rep_dim + self._value_dim
        self._max_embeds = 256
        self._mu = 11.0  # hardcoding 11 as the max coord value for now (range 0-10). can grab from mettagrid_env.py

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        self.register_buffer("frequencies", 2.0 ** torch.arange(self._num_freqs))

        return None

    def forward(self, td: TensorDict) -> TensorDict:
        observations = td[self.config.in_key]

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

        td[self.config.out_key] = feat_vectors
        return td


class ObsAttrCoordValueEmbedConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "obs_attr_coord_value_embed"
    attr_embed_dim: int = 12

    def make_component(self, env=None):
        return ObsAttrCoordValueEmbed(config=self)


class ObsAttrCoordValueEmbed(nn.Module):
    """An experiment that embeds attr value as a categorical variable. Using a normalization layer is not
    recommended."""

    def __init__(self, config: ObsAttrCoordValueEmbedConfig) -> None:
        super().__init__()
        self.config = config
        self._attr_embed_dim = self.config.attr_embed_dim  # Dimension of attribute embeddings
        self._value_dim = 1
        self._max_embeds = 256

        self._attr_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim, padding_idx=255)
        nn.init.trunc_normal_(self._attr_embeds.weight, std=0.02)

        # Coord byte supports up to 16x16, so 256 possible coord values
        self._coord_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._coord_embeds.weight, std=0.02)

        self._val_embeds = nn.Embedding(self._max_embeds, self._attr_embed_dim)
        nn.init.trunc_normal_(self._val_embeds.weight, std=0.02)

        return None

    def forward(self, td: TensorDict) -> TensorDict:
        # [B, M, 3] the 3 vector is: coord (unit8), attr_idx, attr_val
        observations = td[self.config.in_key]

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

        td[self.config.out_key] = combined_embeds
        return td


# TODO: try scaling attr index embed by normalized value
# TODO: try embed attr value but concat to attr index embed using list slicing (not torch.cat). Also cat fourier reps
