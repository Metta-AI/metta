"""Transformer policies that mirror legacy PyTorch agents exactly."""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from pydantic import model_validator
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite, UnboundedDiscrete

import pufferlib.pytorch
from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.actor import (
    ActionProbsConfig,
    ActorKey,
    ActorKeyConfig,
    ActorQuery,
    ActorQueryConfig,
)
from metta.agent.components.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.components.heads import LinearHead, LinearHeadConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimBoxConfig
from metta.agent.components.transformer_core import (
    TransformerBackboneConfig,
    TransformerBackboneVariant,
)
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


def _tensor_stats(tensor: torch.Tensor, name: str) -> str:
    if tensor.numel() == 0:
        return f"{name}=empty"
    return (
        f"{name}(shape={tuple(tensor.shape)}, mean={tensor.float().mean().item():.4f}, "
        f"std={tensor.float().std().item():.4f}, min={tensor.float().min().item():.4f}, "
        f"max={tensor.float().max().item():.4f})"
    )


_POLICY_VARIANT_DEFAULTS: Dict[TransformerBackboneVariant, Dict[str, bool]] = {
    TransformerBackboneVariant.GTRXL: {"manual_init": False, "strict_attr_indices": False},
    TransformerBackboneVariant.TRXL: {"manual_init": False, "strict_attr_indices": False},
    TransformerBackboneVariant.TRXL_NVIDIA: {"manual_init": True, "strict_attr_indices": True},
}


class TransformerPolicyConfig(PolicyArchitecture):
    """Configures the end-to-end transformer policy."""

    class_path: str = "metta.agent.policies.transformer.TransformerPolicy"

    variant: TransformerBackboneVariant = TransformerBackboneVariant.GTRXL

    # Observation preprocessing
    obs_shim_config: ObsShimBoxConfig = ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig(
        in_key="obs_normalizer",
        out_key="encoded_obs",
        cnn1_cfg={"out_channels": 64, "kernel_size": 5, "stride": 3},
        cnn2_cfg={"out_channels": 64, "kernel_size": 3, "stride": 1},
        fc1_cfg={"out_features": 256},
        encoded_obs_cfg={"out_features": 256},
    )

    transformer: TransformerBackboneConfig | None = None

    # Actor / critic head dimensions
    critic_hidden_dim: int = 512
    actor_hidden_dim: int = 256
    action_embedding_dim: int = 16
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    # Implementation options
    manual_init: bool | None = None
    strict_attr_indices: bool | None = None

    @model_validator(mode="after")
    def _apply_variant_defaults(self) -> "TransformerPolicyConfig":
        overrides = {}
        if self.transformer is not None:
            overrides = self.transformer.model_dump(exclude_none=True)
        overrides["variant"] = self.variant
        self.transformer = TransformerBackboneConfig(**overrides)

        defaults = _POLICY_VARIANT_DEFAULTS[self.variant]
        if self.manual_init is None:
            self.manual_init = defaults["manual_init"]
        if self.strict_attr_indices is None:
            self.strict_attr_indices = defaults["strict_attr_indices"]

        return self


class TransformerPolicy(Policy):
    """Shared CNN + Transformer policy scaffolding."""

    ConfigClass = TransformerPolicyConfig

    def __init__(self, env, config: Optional[TransformerPolicyConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = self.ConfigClass()
        self.config = config

        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.transformer_cfg = self.config.transformer
        self.latent_size = self.transformer_cfg.latent_size
        self.hidden_size = self.transformer_cfg.hidden_size
        self.strict_attr_indices = getattr(self.config, "strict_attr_indices", False)
        self.num_layers = max(env.feature_normalizations.keys()) + 1

        encoder_out = self.config.cnn_encoder_config.encoded_obs_cfg.get("out_features")
        if encoder_out != self.latent_size:
            logger.info(
                "Adjusting CNN encoder output from %s to match transformer latent size %s.",
                encoder_out,
                self.latent_size,
            )
            self.config.cnn_encoder_config.encoded_obs_cfg["out_features"] = self.latent_size

        self.obs_shim = ObsShimBox(env=env, config=self.config.obs_shim_config)
        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, env=env)

        self._build_heads()
        self._build_transformer()

        self._memory: Dict[int, Optional[Dict[str, Optional[List[torch.Tensor]]]]] = {}

        # Expose action-metadata placeholders for compatibility with legacy helpers
        self.cum_action_max_params: Optional[torch.Tensor] = None
        self.action_index_tensor: Optional[torch.Tensor] = None

        self._diag_enabled = os.getenv("TRANSFORMER_DIAG", "0") == "1"
        self._diag_limit = int(os.getenv("TRANSFORMER_DIAG_STEPS", "5"))
        self._diag_counter = 0
        self._diag_train_logged = False

        if self.latent_size != self.hidden_size:
            self.input_projection = pufferlib.pytorch.layer_init(nn.Linear(self.latent_size, self.hidden_size), std=1.0)
        else:
            self.input_projection = nn.Identity()

    def _build_heads(self) -> None:
        manual = self.config.manual_init
        actor_init_std = 1.0
        critic_init_std = math.sqrt(2)
        value_init_std = 1.0

        self.actor_head = LinearHead(
            LinearHeadConfig(
                in_key="core",
                out_key="actor_1",
                in_features=self.hidden_size,
                out_features=self.config.actor_hidden_dim,
                activation="ReLU",
                manual_init=manual,
                init_std=actor_init_std,
            )
        )

        self.critic_head = LinearHead(
            LinearHeadConfig(
                in_key="core",
                out_key="critic_1",
                in_features=self.hidden_size,
                out_features=self.config.critic_hidden_dim,
                activation="Tanh",
                manual_init=manual,
                init_std=critic_init_std,
            )
        )

        self.value_head = LinearHead(
            LinearHeadConfig(
                in_key="critic_1",
                out_key="values",
                in_features=self.config.critic_hidden_dim,
                out_features=1,
                activation=None,
                manual_init=manual,
                init_std=value_init_std,
            )
        )

        self.action_embeddings = ActionEmbedding(
            ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=self.config.action_embedding_dim)
        )
        self.actor_query = ActorQuery(
            ActorQueryConfig(
                in_key="actor_1",
                out_key="actor_query",
                hidden_size=self.config.actor_hidden_dim,
                embed_dim=self.config.action_embedding_dim,
            )
        )
        self.actor_key = ActorKey(
            ActorKeyConfig(
                query_key="actor_query",
                embedding_key="action_embedding",
                out_key="logits",
                hidden_size=self.config.actor_hidden_dim,
                embed_dim=self.config.action_embedding_dim,
            )
        )
        self.action_probs = self.config.action_probs_config.make_component()

    def _build_transformer(self) -> None:
        self.transformer_module = self.transformer_cfg.make_component(self.env)
        self._memory_enabled = getattr(self.transformer_cfg, "memory_len", 0) > 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    @property
    def cnn1(self) -> nn.Module:
        """Expose first CNN layer for legacy hooks/tests."""

        return self.cnn_encoder.cnn1

    def _encode_observations(self, observations: torch.Tensor) -> TensorDict:
        """Run raw observations through preprocessing and CNN encoder."""

        if observations.dim() != 3:
            raise ValueError("Expected observations with shape (batch, tokens, features).")

        batch_size = observations.shape[0]
        device = observations.device
        td = TensorDict({"env_obs": observations}, batch_size=[batch_size])

        coords_byte = observations[..., 0].to(torch.long)
        attr_indices = observations[..., 1].to(torch.long)
        attr_values = observations[..., 2].to(torch.float32)

        valid_tokens = coords_byte != 0xFF
        valid_attr = attr_indices < self.num_layers
        valid_mask = valid_tokens & valid_attr

        x_indices = (coords_byte >> 4) & 0x0F
        y_indices = coords_byte & 0x0F
        flat_spatial_index = x_indices * self.env.obs_height + y_indices
        dim_per_layer = self.env.obs_width * self.env.obs_height
        combined_index = attr_indices * dim_per_layer + flat_spatial_index

        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, attr_values, torch.zeros_like(attr_values))

        grid_flat = torch.zeros((batch_size, self.num_layers * dim_per_layer), dtype=torch.float32, device=device)
        grid_flat.scatter_add_(1, safe_index, safe_values)
        box_obs = grid_flat.view(batch_size, self.num_layers, self.env.obs_width, self.env.obs_height)

        norm = self.obs_shim.observation_normalizer.obs_norm.to(device=device, dtype=box_obs.dtype)
        normalized = box_obs / norm

        td[self.obs_shim.out_key] = normalized
        self.cnn_encoder(td)
        return td

    @torch._dynamo.disable
    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None):
        td, observations, batch_size, tt, original_shape = self._prepare_observations(td)

        if self.strict_attr_indices:
            self._enforce_strict_attr_indices(td)

        self.obs_shim(td)
        self.cnn_encoder(td)

        encoded_key = self.config.cnn_encoder_config.out_key
        latent = td[encoded_key]
        latent = self.input_projection(latent)

        if self._diag_enabled and self._diag_counter < self._diag_limit:
            logger.info("[TRANSFORMER_DIAG] %s", _tensor_stats(latent, "latent"))

        core = self._forward_transformer(td, latent, batch_size, tt)
        td["core"] = core

        self.actor_head(td)
        self.critic_head(td)
        self.value_head(td)
        td["values"] = td["values"].flatten()

        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td)

        td = self.action_probs(td, action)
        if self._diag_enabled and self._diag_counter < self._diag_limit:
            logits = td.get("logits")
            if logits is not None:
                logger.info("[TRANSFORMER_DIAG] %s", _tensor_stats(logits, "logits"))
            values = td.get("values")
            if values is not None:
                logger.info("[TRANSFORMER_DIAG] %s", _tensor_stats(values, "values"))
            self._diag_counter += 1
        if original_shape is not None:
            td = td.reshape(original_shape)
        return td

    def _forward_transformer(self, td: TensorDict, latent: torch.Tensor, batch_size: int, tt: int) -> torch.Tensor:
        if tt <= 0:
            raise ValueError("bptt entries must be positive")

        total_batch = latent.shape[0]
        if total_batch != batch_size * tt:
            raise ValueError("encoded_obs batch dimension must be divisible by bptt")

        latent_seq = latent.view(batch_size, tt, self.hidden_size).transpose(0, 1)

        use_memory = self._memory_enabled
        memory = None
        env_key: Optional[int] = None
        if use_memory and tt == 1:
            env_key = self._get_env_start(td)
            memory = self._memory.get(env_key)

        core_out, new_memory = self.transformer_module(latent_seq, memory)
        core_flat = core_out.transpose(0, 1).reshape(batch_size * tt, self.hidden_size)

        if self._diag_enabled and self._diag_counter < self._diag_limit:
            logger.info("[TRANSFORMER_DIAG] %s", _tensor_stats(core_flat, "core"))
            if new_memory is not None:
                hidden_states = new_memory.get("hidden_states")
                if hidden_states:
                    mem_norms = []
                    for layer_mem in hidden_states:
                        if layer_mem is None or layer_mem.numel() == 0:
                            mem_norms.append(0.0)
                        else:
                            mem_norms.append(float(layer_mem.float().norm().item()))
                    logger.info("[TRANSFORMER_DIAG] memory_norms=%s", mem_norms)

        if use_memory and tt == 1 and env_key is not None:
            updated_memory = self._detach_memory(new_memory)
            if updated_memory is not None:
                dones = td.get("dones", None)
                truncateds = td.get("truncateds", None)
                reset_mask = self._compute_reset_mask(dones, truncateds, batch_size)
                if reset_mask is not None and reset_mask.any():
                    if self._diag_enabled and self._diag_counter < self._diag_limit:
                        logger.info(
                            "[TRANSFORMER_DIAG] reset_mask true_count=%s",
                            int(reset_mask.sum().item()),
                        )
                    hidden_states = updated_memory.get("hidden_states")
                    if hidden_states:
                        for idx, layer_mem in enumerate(hidden_states):
                            if layer_mem is None or layer_mem.numel() == 0:
                                continue
                            masked_layer = layer_mem.clone()
                            masked_layer[:, reset_mask] = 0
                            hidden_states[idx] = masked_layer
                self._memory[env_key] = updated_memory
                if self._diag_enabled and self._diag_counter < self._diag_limit:
                    logger.info("[TRANSFORMER_DIAG] memory cached for env %s", env_key)
        elif use_memory and tt > 1 and env_key is not None:
            self._memory.pop(env_key, None)
            if self._diag_enabled and self._diag_counter < self._diag_limit:
                logger.info("[TRANSFORMER_DIAG] cleared cached memory for env %s (tt=%s)", env_key, tt)

        return core_flat

    def _prepare_observations(self, td: TensorDict) -> Tuple[TensorDict, torch.Tensor, int, int, Optional[torch.Size]]:
        observations = td["env_obs"]

        meta_bptt = td.get("bptt", None)
        meta_batch = td.get("batch", None)
        if meta_bptt is not None and meta_bptt.numel() > 0 and meta_batch is not None and meta_batch.numel() > 0:
            tt = int(meta_bptt.reshape(-1)[0].item())
            batch_size = int(meta_batch.reshape(-1)[0].item())
            total_batch = batch_size * tt
            if td.batch_dims > 1:
                td = td.reshape(total_batch)
                observations = td["env_obs"]
            original_shape = torch.Size([batch_size, tt]) if tt > 1 else None
            if self._diag_enabled and self._diag_counter < self._diag_limit:
                logger.info("[TRANSFORMER_DIAG] prepare_obs meta batch=%s tt=%s", batch_size, tt)
        elif observations.dim() == 4:
            batch_size, tt = observations.shape[:2]
            total_batch = batch_size * tt
            if td.batch_dims > 1:
                td = td.reshape(total_batch)
                observations = td["env_obs"]
            device = observations.device
            if "bptt" not in td.keys():
                td.set("bptt", torch.full((total_batch,), tt, device=device, dtype=torch.long))
            if "batch" not in td.keys():
                td.set("batch", torch.full((total_batch,), batch_size, device=device, dtype=torch.long))
            original_shape: Optional[torch.Size] = torch.Size([batch_size, tt])
            if self._diag_enabled and self._diag_counter < self._diag_limit:
                logger.info("[TRANSFORMER_DIAG] prepare_obs sequence batch=%s tt=%s", batch_size, tt)
        else:
            batch_size = observations.shape[0]
            tt = 1
            if td.batch_dims > 1:
                td = td.reshape(batch_size)
                observations = td["env_obs"]
            device = observations.device
            if "bptt" not in td.keys():
                td.set("bptt", torch.ones((batch_size,), device=device, dtype=torch.long))
            if "batch" not in td.keys():
                td.set("batch", torch.full((batch_size,), batch_size, device=device, dtype=torch.long))
            original_shape = None
            if self._diag_enabled and self._diag_counter < self._diag_limit:
                logger.info("[TRANSFORMER_DIAG] prepare_obs batch=%s tt=1", batch_size)

        return td, observations, batch_size, tt, original_shape

    def _enforce_strict_attr_indices(self, td: TensorDict) -> None:
        obs = td.get("env_obs", None)
        if obs is None or obs.numel() == 0:
            return
        if obs.dim() == 4:
            obs = obs.view(-1, obs.shape[-2], obs.shape[-1])
        if obs.dim() != 3:
            return

        coords_byte = obs[..., 0].to(torch.uint8)
        attr_indices = obs[..., 1].long()
        valid_tokens = coords_byte != 0xFF
        invalid_mask = valid_tokens & (attr_indices >= self.num_layers)
        if invalid_mask.any():
            invalid_indices = torch.unique(attr_indices[invalid_mask]).cpu().tolist()
            raise ValueError(
                "Found observation attribute indices "
                f"{sorted(int(idx) for idx in invalid_indices)} >= num_layers ({self.num_layers})."
            )

    def _compute_reset_mask(
        self,
        dones: Optional[torch.Tensor],
        truncateds: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if dones is None and truncateds is None:
            return None

        if dones is None and truncateds is not None:
            dones = torch.zeros_like(truncateds)
        elif truncateds is None and dones is not None:
            truncateds = torch.zeros_like(dones)

        assert dones is not None
        assert truncateds is not None

        if dones.numel() == 0 and truncateds.numel() == 0:
            return None

        try:
            dones = rearrange(dones, "(b t) -> t b", b=batch_size)
            truncateds = rearrange(truncateds, "(b t) -> t b", b=batch_size)
        except ValueError:
            dones = dones.view(1, batch_size)
            truncateds = truncateds.view(1, batch_size)

        reset = dones.bool() | truncateds.bool()
        if self._diag_enabled and self._diag_counter < self._diag_limit:
            logger.info(
                "[TRANSFORMER_DIAG] compute_reset_mask dones_last=%s truncateds_last=%s",
                bool(dones[-1].any().item()),
                bool(truncateds[-1].any().item()),
            )
        return reset[-1]

    # Legacy compatibility placeholders (no longer used)
    def _build_memory_batch(self, *args, **kwargs):
        return None

    def _store_memory_batch(self, *args, **kwargs):
        return None

    def _detach_memory(
        self, memory: Optional[Dict[str, Optional[List[torch.Tensor]]]]
    ) -> Optional[Dict[str, Optional[List[torch.Tensor]]]]:
        if memory is None:
            return None
        hidden_states = memory.get("hidden_states")
        if hidden_states is None:
            return None
        return {"hidden_states": [layer.detach() if layer is not None else None for layer in hidden_states]}

    def _get_env_start(self, td: TensorDict) -> int:
        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is not None and training_env_ids.numel() > 0:
            env_start = int(training_env_ids.reshape(-1)[0].item())
            if self._diag_enabled and self._diag_counter < self._diag_limit:
                env_ids = training_env_ids.reshape(-1)
                logger.info(
                    "[TRANSFORMER_DIAG] env_ids start=%s min=%s max=%s",
                    env_start,
                    int(env_ids.min()),
                    int(env_ids.max()),
                )
            return env_start
        training_env_id = td.get("training_env_id", None)
        if training_env_id is not None and training_env_id.numel() > 0:
            env_start = int(training_env_id.reshape(-1)[0].item())
            if self._diag_enabled and self._diag_counter < self._diag_limit:
                logger.info("[TRANSFORMER_DIAG] env_start from training_env_id=%s", env_start)
            return env_start
        return 0

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def initialize_to_environment(self, env, device):
        device = torch.device(device)
        self.to(device)

        log = self.obs_shim.initialize_to_environment(env, device)
        self.action_embeddings.initialize_to_environment(env, device)
        self.action_probs.initialize_to_environment(env, device)
        self.cum_action_max_params = self.action_probs.cum_action_max_params
        self.action_index_tensor = self.action_probs.action_index_tensor
        self._memory.clear()
        return [log] if log is not None else []

    def reset_memory(self) -> None:
        self._memory.clear()

    # ------------------------------------------------------------------
    # Action helpers (legacy compatibility)
    # ------------------------------------------------------------------
    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        """Delegate action-to-logit conversion to ActionProbs component."""

        return self.action_probs._convert_action_to_logit_index(flattened_action)

    def _convert_logit_index_to_action(self, logit_indices: torch.Tensor) -> torch.Tensor:
        """Delegate logit-to-action conversion to ActionProbs component."""

        return self.action_probs._convert_logit_index_to_action(logit_indices)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )


def gtrxl_policy_config() -> TransformerPolicyConfig:
    """Return a policy config for the GTrXL variant."""

    return TransformerPolicyConfig(variant=TransformerBackboneVariant.GTRXL)


def trxl_policy_config() -> TransformerPolicyConfig:
    """Return a policy config for the vanilla Transformer-XL variant."""

    return TransformerPolicyConfig(variant=TransformerBackboneVariant.TRXL)


def trxl_nvidia_policy_config() -> TransformerPolicyConfig:
    """Return a policy config for the NVIDIA Transformer-XL variant."""

    return TransformerPolicyConfig(variant=TransformerBackboneVariant.TRXL_NVIDIA)


__all__ = [
    "TransformerPolicyConfig",
    "TransformerPolicy",
    "TransformerBackboneVariant",
    "gtrxl_policy_config",
    "trxl_policy_config",
    "trxl_nvidia_policy_config",
]
