import warnings
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig

# =========================== Token-based observation shaping ===========================
# The two nn.Module-based classes below are composed into ObsShaperTokens. You can simply call that class in your policy
# for token-based observation shaping. Or you can manipulate the two classes below directly in your policy.


class ObsTokenPadStrip(nn.Module):
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
        env,
        in_key: str = "env_obs",
        out_key: str = "obs_token_pad_strip",
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self._max_tokens = max_tokens
        # Initialize feature remapping as identity by default
        self.register_buffer("feature_id_remap", torch.arange(256, dtype=torch.uint8))
        self._remapping_active = False
        self.register_buffer("_positions_cache", torch.empty(0, dtype=torch.int64), persistent=False)

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> str:
        # Build feature mappings
        features = env.obs_features
        self.feature_id_to_name = {props.id: name for name, props in features.items()}
        self.feature_normalizations = {
            props.id: props.normalization for props in features.values() if hasattr(props, "normalization")
        }

        if not hasattr(self, "original_feature_mapping"):
            self.original_feature_mapping = {name: props.id for name, props in features.items()}  # {name: id}
            return f"Stored original feature mapping with {len(self.original_feature_mapping)} features"
        else:
            # Re-initialization - create remapping for agent portability
            UNKNOWN_FEATURE_ID = 255
            feature_remap: dict[int, int] = {}
            unknown_features = []

            for name, props in features.items():
                new_id = props.id
                if name in self.original_feature_mapping:
                    # Remap known features to their original IDs
                    original_id = self.original_feature_mapping[name]
                    if new_id != original_id:
                        feature_remap[new_id] = original_id
                elif not self.training:
                    # In eval mode, map unknown features to UNKNOWN_FEATURE_ID
                    feature_remap[new_id] = UNKNOWN_FEATURE_ID
                    unknown_features.append(name)
                else:
                    # In training mode, learn new features
                    self.original_feature_mapping[name] = new_id

            if feature_remap:
                # Apply the remapping
                self._apply_feature_remapping(feature_remap, features, UNKNOWN_FEATURE_ID, device)
                return f"Created feature remapping: {len(feature_remap)} remapped, {len(unknown_features)} unknown"
            else:
                return "No feature remapping created"

    def _apply_feature_remapping(
        self, mapping: dict[int, int], features: dict, unknown_id: int, device: torch.device
    ) -> None:
        """Apply feature remapping to policy for agent portability across environments."""
        # Build complete remapping tensor
        remap_tensor = torch.arange(256, dtype=torch.uint8, device=device)

        # Apply explicit remappings
        for new_id, original_id in mapping.items():
            remap_tensor[new_id] = original_id

        # Map unused feature IDs to UNKNOWN
        current_feature_ids = {props.id for props in features.values()}
        for feature_id in range(256):
            if feature_id not in mapping and feature_id not in current_feature_ids:
                remap_tensor[feature_id] = unknown_id

        self.register_buffer("feature_id_remap", remap_tensor.to(self.feature_id_remap.device))
        identity = torch.arange(256, dtype=torch.uint8, device=remap_tensor.device)
        self._remapping_active = not torch.equal(remap_tensor, identity)

    def forward(self, td: TensorDict) -> TensorDict:
        # [B, M, 3] the 3 vector is: coord (unit8), attr_idx, attr_val
        observations = td[self.in_key]
        M = observations.shape[1]

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
        has_padding = obs_mask.any(dim=1)
        # Treat rows without padding as having full length M so we don't truncate dense sequences.
        row_lengths = torch.where(has_padding, flip_pts, torch.full_like(flip_pts, M))

        # find the global max flip‐point as a 0‐d tensor
        max_flip = row_lengths.max()

        if self._max_tokens is not None:
            max_flip = max_flip.clamp(max=self._max_tokens)

        # build a 1‐D "positions" row [0,1,2,…,L−1]
        if self._positions_cache.numel() < M:
            self._positions_cache = torch.arange(M, dtype=torch.int64, device=obs_mask.device)
        positions = self._positions_cache[:M]

        # make a boolean column mask: keep all columns strictly before max_flip
        keep_cols = positions < max_flip  # shape [L], dtype=torch.bool

        observations = observations[:, keep_cols]  # shape [B, max_flip]
        obs_mask = obs_mask[:, keep_cols]

        td[self.out_key] = observations
        td["obs_mask"] = obs_mask
        return td


class ObsAttrValNorm(nn.Module):
    """Normalizes attr values based on the attr index."""

    def __init__(self, env, in_key: str = "obs_token_pad_strip", out_key: str = "obs_attr_val_norm") -> None:
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self._max_embeds = 256
        self._set_feature_normalizations(env)

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> None:
        self._set_feature_normalizations(env, device)

    def _set_feature_normalizations(self, env, device: Optional[torch.device] = None):
        features = env.feature_normalizations
        self._feature_normalizations = features
        self._update_norm_factors(device)
        return None

    def _update_norm_factors(self, device: Optional[torch.device] = None):
        # Create a tensor for feature normalizations
        # We need to handle the case where attr_idx might be 0 (padding) or larger than defined normalizations.
        # Assuming max attr_idx is 256 (same as attr_embeds size - 1 for padding_idx).
        # Initialize with 1.0 to avoid division by zero for unmapped indices.
        if device is None:
            device = "cpu"  # assume that the policy is sent to device after its built for the first time.
        norm_tensor = torch.ones(self._max_embeds, dtype=torch.float32, device=device)
        for i, val in self._feature_normalizations.items():
            if i < len(norm_tensor):  # Ensure we don't go out of bounds
                norm_tensor[i] = val
            else:
                raise ValueError("feature normalization index exceeds embedding size")
        self.register_buffer("_norm_factors", norm_tensor)
        return None

    def forward(self, td: TensorDict) -> TensorDict:
        observations = td[self.in_key]
        attr_indices = observations[..., 1].long()
        norm_factors = self._norm_factors[attr_indices]
        observations = observations.to(torch.float32)
        observations[..., 2] = observations[..., 2] / norm_factors

        td[self.out_key] = observations

        return td


class ObsShimTokensConfig(ComponentConfig):
    in_key: str
    out_key: str
    max_tokens: int | None = None
    name: str = "obs_shim_tokens"

    def make_component(self, env):
        return ObsShimTokens(env, config=self)


class ObsShimTokens(nn.Module):
    def __init__(self, env, config: ObsShimTokensConfig) -> None:
        super().__init__()
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.token_pad_striper = ObsTokenPadStrip(env, in_key=self.in_key, max_tokens=config.max_tokens)
        self.attr_val_normer = ObsAttrValNorm(env, out_key=self.out_key)

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> str:
        log = self.token_pad_striper.initialize_to_environment(env, device)
        self.attr_val_normer.initialize_to_environment(env, device)
        return log

    def forward(self, td: TensorDict) -> TensorDict:
        td = self.token_pad_striper(td)
        td = self.attr_val_normer(td)
        return td


# =========================== End Token-based observation shaping ===========================


# =========================== Box-based observation shaping ===========================


class ObsTokenToBoxShim(nn.Module):
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

    def __init__(self, env, in_key="env_obs", out_key="box_obs"):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.out_width = env.obs_width
        self.out_height = env.obs_height
        self.num_layers = max(env.feature_normalizations.keys()) + 1

    def forward(self, td: TensorDict):
        token_observations = td[self.in_key]
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

        td[self.out_key] = box_obs
        return td


class ObservationNormalizer(nn.Module):
    """
    Normalizes observation features by dividing each feature by its approximate maximum value.

    This class scales observation features to a range of approximately [0, 1] by dividing
    each feature by predefined normalization values from the OBS_NORMALIZATIONS dictionary.
    Normalization helps stabilize neural network training by preventing features with large
    magnitudes from dominating the learning process.
    """

    def __init__(self, env, in_key="box_obs", out_key="obs_normalizer"):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self._initialize_to_environment(env.feature_normalizations)

    def forward(self, td: TensorDict):
        td[self.out_key] = td[self.in_key] / self.obs_norm
        return td

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> None:
        self._initialize_to_environment(env.feature_normalizations, device)

    def _initialize_to_environment(self, features: dict[str, dict], device: Optional[torch.device] = None):
        self.feature_normalizations = features
        obs_norm = torch.ones(max(self.feature_normalizations.keys()) + 1, dtype=torch.float32)
        for i, val in self.feature_normalizations.items():
            obs_norm[i] = val
        obs_norm = obs_norm.view(1, len(self.feature_normalizations), 1, 1)

        self.register_buffer("obs_norm", obs_norm)
        if device:
            self.obs_norm = self.obs_norm.to(device)


class ObsShimBoxConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "obs_shim_box"

    def make_component(self, env):
        return ObsShimBox(env, config=self)


class ObsShimBox(nn.Module):
    def __init__(self, env, config: ObsShimBoxConfig) -> None:
        super().__init__()
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.token_to_box_shim = ObsTokenToBoxShim(env, in_key=self.in_key)
        self.observation_normalizer = ObservationNormalizer(env, out_key=self.out_key)

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> None:
        self.observation_normalizer.initialize_to_environment(env, device)

    def forward(self, td: TensorDict):
        td = self.token_to_box_shim(td)
        td = self.observation_normalizer(td)
        return td
