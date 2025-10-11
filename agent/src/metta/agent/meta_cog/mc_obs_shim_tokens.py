from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig
from metta.agent.meta_cog.mc import MetaCogAction

# =========================== Token-based observation shaping ===========================
# The two nn.Module-based classes below are composed into ObsShaperTokens. You can simply call that class in your policy
# for token-based observation shaping. Or you can manipulate the two classes below directly in your policy.


class ObsTokenPadStrip(nn.Module):
    """
    Token-based observation shaping layer that trims right-padding and exposes a per-environment
    focus window. Given token observations shaped [B_or_BxTT, M, 3], it computes the maximum
    dense length in the batch (global crop) and returns a cropped tensor [B_or_BxTT, L, 3] with
    an accompanying boolean mask under key "obs_mask" where True marks padding/filtered tokens.

    Extended design notes (MetaCog actions and control flow)
    -------------------------------------------------------
    This class participates in the MetaCog (MC) two-pass policy flow. Each environment step is
    evaluated in two phases:
      1) MC-action selection: The policy produces logits/probabilities over internal actions
         ("mc_actions"). `MCPolicyAutoBuilder` collects all `MetaCogAction` instances from
         components and assigns each a unique integer index. The chosen mc_action indices are
         passed into `MCPolicyAutoBuilder.apply_mc_actions`, which routes the selected env_ids to
         the corresponding action handlers on components.
      2) Main policy pass: After components mutate their transient internal state in response to
         mc_actions, the standard forward pass is run again on the original observations, now with
         the updated internal state.

    Internal actions in this class
    ------------------------------
    ObsTokenPadStrip defines two `MetaCogAction`s, `focus_1` and `focus_2`. These actions shrink
    the per-environment focus window for the current step to a configurable token limit
    (`focus_1_max_tokens`, `focus_2_max_tokens`). Implementation details:
      - Per-env mask buffer: `focus_mask` has shape [num_envs, _default_max_tokens]. A value of
        True means the token position is allowed; False means masked. Rows are grown on demand to
        accommodate new env_ids encountered during training/eval.
      - No Python loops: Focus windows are applied via advanced indexing on env rows, e.g.:
        `focus_mask[env_ids, :] = True; focus_mask[env_ids, clamped:] = False`.
      - One-step semantics and reset: A focus override is transient. After the forward pass
        applies `focus_mask` to build the output `obs_mask`, any env rows that were modified are
        reset back to the pass-through state (all True) to prepare for the next step.
      - Composition with global crop: The forward first determines a batch-global crop length L
        (max dense length across rows, clamped to `_default_max_tokens`). The per-env focus then
        further masks positions in [0:L) using a row-specific mask. This preserves a single
        sequence width L across the batch while honoring per-env shrinking.

    Double-buffering and env indexing
    ---------------------------------
    The system uses double-buffered workers; we therefore index per environment, not per row
    group. This module expects `td["training_env_ids"]` to be present and shaped [B*TT] in both
    rollout (TT==1) and training (TT>1). If absent, a contiguous range [0, B*TT) is synthesized.
    The same env id appearing at multiple timesteps within a batch will share the same per-env
    focus row; the mask is applied per-row during the step, then reset.

    Buffers, devices, and persistence
    ---------------------------------
    - Buffers such as `focus_mask`, `_pending_focus_reset`, `feature_id_remap`, and
      `_positions_cache` are registered buffers and will move with the module across devices.
    - `focus_mask` and `_pending_focus_reset` are marked `persistent=False` because they are
      ephemeral control/state for the current step and are recreated on module construction and
      grown on demand. Checkpoint reloads rely on `__init__` to restore a clean state rather than
      restoring potentially stale shapes/content from a checkpoint.

    Constraints and validation
    --------------------------
    - `_default_max_tokens` defines an upper bound on any crop in this layer; global crop length L
      is clamped to this value. The per-env focus window is clamped to the same bound.
    - `focus_1_max_tokens` and `focus_2_max_tokens` must be in (0, `_default_max_tokens`).
    - `training_env_ids` must index valid env rows. The focus buffers grow to accommodate the
      maximum env id seen so far.

    Extension guidance
    ------------------
    - New mc actions: Add `MetaCogAction` fields and attach callable handlers via
      `.attach_apply_method(callable)`. Handlers should only mutate transient per-env state and
      avoid loops by leveraging tensor indexing.
    - Additional per-env state: Register new buffers with `persistent=False` unless you have a
      strong reason to checkpoint them. Always support dynamic growth via `_ensure_*capacity`.
    - Alternative focus policies: Replace the rectangular window with learned or content-based
      boolean masks; integrate by writing into the per-env buffer before the main forward crop.
    - Keep computations device-agnostic by reading buffer `.device` and avoiding ad-hoc casts of
      runtime tensors such as `env_ids` (the caller ensures tensors are already on the module’s
      device).
    """

    def __init__(
        self,
        env,
        in_key: str = "env_obs",
        out_key: str = "obs_token_pad_strip",
        max_tokens: int | None = None,
        focus_1_max_tokens: int | None = None,
        focus_2_max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key

        # Always have a default max; policy config can override
        self._default_max_tokens = max_tokens if max_tokens is not None else 200
        self.register_buffer("max_tokens", torch.tensor(0))  # i think this is wrong

        # Per-env focus control. A row-wise mask of shape [num_envs, _default_max_tokens].
        # True means the token position is allowed to pass; False means masked.
        # Grows on demand to accommodate new env_ids.
        self.register_buffer(
            "focus_mask",
            torch.ones(0, self._default_max_tokens, dtype=torch.bool),
            persistent=False,
        )

        # Tracks which envs had a focus override applied and should be reset after forward
        self.register_buffer("_pending_focus_reset", torch.zeros(0, dtype=torch.bool), persistent=False)

        self.focus_1 = MetaCogAction("focus_1")
        self.focus_1.attach_apply_method(self.inject_focus_1)
        self.focus_1_max_tokens = 15 if focus_1_max_tokens is None else int(focus_1_max_tokens)
        self.focus_2 = MetaCogAction("focus_2")
        self.focus_2.attach_apply_method(self.inject_focus_2)
        self.focus_2_max_tokens = 30 if focus_2_max_tokens is None else int(focus_2_max_tokens)
        # self.noise_1 = MetaCogAction("noise_1")
        # self.noise_1.attach_apply_method(self.inject_noise_1)

        # Initialize feature remapping as identity by default
        self.register_buffer("feature_id_remap", torch.arange(256, dtype=torch.uint8))
        self._remapping_active = False
        self.register_buffer("_positions_cache", torch.empty(0, dtype=torch.int64), persistent=False)

    @torch.no_grad()
    def _ensure_focus_capacity(self, max_env_id: int, device: torch.device) -> None:
        """Ensure focus buffers can index up to max_env_id (inclusive)."""
        if max_env_id < 0:
            return
        current_rows = self.focus_mask.size(0)
        required_rows = max_env_id + 1
        if required_rows <= current_rows:
            return

        # Grow focus_mask with True (allow all by default)
        new_focus = torch.ones(required_rows, self._default_max_tokens, dtype=torch.bool, device=device)
        if current_rows > 0:
            new_focus[:current_rows].copy_(self.focus_mask)
        # In-place resize to preserve buffer registration
        self.focus_mask.resize_as_(new_focus)
        self.focus_mask.copy_(new_focus)

        # Grow pending reset flags
        new_pending = torch.zeros(required_rows, dtype=torch.bool, device=device)
        if self._pending_focus_reset.numel() > 0:
            new_pending[: self._pending_focus_reset.size(0)].copy_(self._pending_focus_reset)
        self._pending_focus_reset.resize_as_(new_pending)
        self._pending_focus_reset.copy_(new_pending)

    @torch.no_grad()
    def _set_focus_for_envs(self, env_ids: torch.Tensor, max_tokens: int) -> None:
        """Set per-env focus window to max_tokens for the given envs for the current step.

        This marks envs for reset after the next forward application.
        """
        if env_ids is None or env_ids.numel() == 0:
            return

        if env_ids.dtype != torch.long:
            env_ids = env_ids.long()
        self._ensure_focus_capacity(int(env_ids.max().item()), self.focus_mask.device)

        clamped = max(0, min(int(max_tokens), int(self._default_max_tokens)))

        # Allow [0:clamped), mask [clamped:]
        self.focus_mask[env_ids, :] = True
        if clamped < self._default_max_tokens:
            self.focus_mask[env_ids, clamped:] = False
        self._pending_focus_reset[env_ids] = True

    @torch.no_grad()
    def inject_focus_1(self, env_ids: torch.Tensor) -> None:
        # shrink to focus_1_max_tokens for the specified env_ids
        self._set_focus_for_envs(env_ids, self.focus_1_max_tokens)

    @torch.no_grad()
    def inject_focus_2(self, env_ids: torch.Tensor) -> None:
        # shrink to focus_2_max_tokens for the specified env_ids
        self._set_focus_for_envs(env_ids, self.focus_2_max_tokens)

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

        # Validate focus sizes relative to default
        if not (0 < int(self.focus_1_max_tokens) < int(self._default_max_tokens)):
            raise ValueError("focus_1_max_tokens must be > 0 and < _default_max_tokens")
        if not (0 < int(self.focus_2_max_tokens) < int(self._default_max_tokens)):
            raise ValueError("focus_2_max_tokens must be > 0 and < _default_max_tokens")

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

        B = td.batch_size.numel()
        if td["bptt"][0] != 1:
            TT = td["bptt"][0]
            self._in_training = True
        else:
            TT = 1
        B = B // TT

        # Resolve env indexing strictly from training_env_ids
        env_ids = td.get("training_env_ids", None)
        if env_ids is None:
            # Fallback: synthesize contiguous env ids for current batch rows
            env_ids = torch.arange(B * TT, device=observations.device)
        else:
            # Ensure 1D shape matching rows (B * TT)
            env_ids = env_ids.reshape(B * TT)

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

        max_flip = max_flip.clamp(max=self._default_max_tokens)

        # build a 1‐D "positions" row [0,1,2,…,L−1]
        if self._positions_cache.numel() < M:
            self._positions_cache = torch.arange(M, dtype=torch.int64, device=obs_mask.device)
        positions = self._positions_cache[:M]

        # make a boolean column mask: keep all columns strictly before max_flip
        keep_cols = positions < max_flip  # shape [L], dtype=torch.bool

        observations = observations[:, keep_cols]  # shape [B, max_flip]
        obs_mask = obs_mask[:, keep_cols]

        # Apply per-env focus after global crop via an additional mask. This preserves
        # uniform sequence width across the batch while allowing env-specific shrinking.
        if self.focus_mask.numel() > 0 and max_flip.item() > 0:
            current_len = int(max_flip.item())
            self._ensure_focus_capacity(int(env_ids.max().item()), observations.device)
            # Focus mask per env for current length
            per_env_focus = self.focus_mask[env_ids, :current_len]  # [B, L]
            # Block positions where per_env_focus is False
            obs_mask = obs_mask | (~per_env_focus)

            # Reset any envs that had a focus override applied this step
            if self._pending_focus_reset.numel() > 0:
                # Identify which of the current envs are pending reset
                pending_rows = self._pending_focus_reset[env_ids]
                if pending_rows.any():
                    envs_to_reset = env_ids[pending_rows]
                    # Restore to pass-through (all True)
                    self.focus_mask[envs_to_reset, :] = True
                    self._pending_focus_reset[envs_to_reset] = False

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
