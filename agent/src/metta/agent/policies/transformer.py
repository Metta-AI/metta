"""Transformer policies built on the Metta transformer backbones."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

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


_POLICY_VARIANT_DEFAULTS: Dict[TransformerBackboneVariant, Dict[str, Any]] = {
    TransformerBackboneVariant.GTRXL: {
        "manual_init": False,
        "strict_attr_indices": False,
        "learning_rate_hint": 7.5e-4,
    },
    TransformerBackboneVariant.TRXL: {
        "manual_init": False,
        "strict_attr_indices": False,
        "learning_rate_hint": 9.0e-4,
    },
    TransformerBackboneVariant.TRXL_NVIDIA: {
        "manual_init": True,
        "strict_attr_indices": True,
        "learning_rate_hint": 3.0e-4,
    },
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
    use_aux_tokens: bool = False
    learning_rate_hint: float | None = None

    @model_validator(mode="after")
    def _apply_variant_defaults(self) -> "TransformerPolicyConfig":
        overrides = {}
        if self.transformer is not None:
            overrides = self.transformer.model_dump(exclude_none=True)
        overrides["variant"] = self.variant
        self.transformer = TransformerBackboneConfig(**overrides)

        defaults = _POLICY_VARIANT_DEFAULTS[self.variant]
        if self.manual_init is None:
            self.manual_init = defaults.get("manual_init", False)
        if self.strict_attr_indices is None:
            self.strict_attr_indices = defaults.get("strict_attr_indices", False)
        if self.learning_rate_hint is None:
            self.learning_rate_hint = defaults.get("learning_rate_hint")

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
        self.use_aux_tokens = getattr(self.config, "use_aux_tokens", True)
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

        self._diag_enabled = os.getenv("TRANSFORMER_DIAG", "0") == "1"
        self._diag_limit = int(os.getenv("TRANSFORMER_DIAG_STEPS", "5"))
        self._diag_counter = 0
        self._diag_train_logged = False

        if self.latent_size != self.hidden_size:
            self.input_projection = pufferlib.pytorch.layer_init(nn.Linear(self.latent_size, self.hidden_size), std=1.0)
        else:
            self.input_projection = nn.Identity()

        self.memory_len = getattr(self.transformer_cfg, "memory_len", 0) or 0
        self.action_dim = self._infer_action_dim()
        if self.use_aux_tokens:
            self.reward_proj = nn.Linear(1, self.hidden_size)
            self.reset_proj = nn.Linear(1, self.hidden_size)
            self.last_action_proj = nn.Linear(self.action_dim, self.hidden_size)
            nn.init.xavier_uniform_(self.reward_proj.weight, gain=1.0)
            nn.init.constant_(self.reward_proj.bias, 0.0)
            nn.init.xavier_uniform_(self.reset_proj.weight, gain=1.0)
            nn.init.constant_(self.reset_proj.bias, 0.0)
            nn.init.xavier_uniform_(self.last_action_proj.weight, gain=1.0)
            nn.init.constant_(self.last_action_proj.bias, 0.0)

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
        """Expose first CNN layer for downstream tests and diagnostics."""

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

    def _infer_action_dim(self) -> int:
        space = self.action_space
        if hasattr(space, "nvec"):
            return int(len(space.nvec))
        if hasattr(space, "shape") and space.shape:
            return int(space.shape[0])
        return 1

    def _build_aux_tokens(self, td: TensorDict, batch_size: int, tt: int, device: torch.device) -> torch.Tensor | None:
        if not self.use_aux_tokens:
            return None

        total = batch_size * tt
        reward = td.get("rewards", None)
        if reward is None:
            reward = torch.zeros((total, 1), device=device)
        else:
            reward = reward.view(total, -1).float().to(device=device)
            if reward.size(1) != 1:
                reward = reward[:, :1]

        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is None and truncateds is None:
            resets = torch.zeros((total, 1), device=device)
        else:
            if dones is None:
                dones = torch.zeros_like(truncateds)
            if truncateds is None:
                truncateds = torch.zeros_like(dones)
            resets = torch.logical_or(dones.bool(), truncateds.bool()).float().view(total, -1).to(device=device)
            if resets.size(1) != 1:
                resets = resets[:, :1]

        last_actions = td.get("last_actions", None)
        if last_actions is not None:
            last_actions = last_actions.view(total, -1).float().to(device=device)
        else:
            actions = td.get("actions", None)
            if actions is not None:
                actions = actions.view(batch_size, tt, -1).float().to(device=device)
                prev_actions = torch.zeros_like(actions)
                if tt > 1:
                    prev_actions[:, 1:] = actions[:, :-1]
                last_actions = prev_actions.view(total, -1)
            else:
                last_actions = torch.zeros((total, self.action_dim), device=device)

        if last_actions.size(1) != self.action_dim:
            action_dim = last_actions.size(1)
            if action_dim > self.action_dim:
                last_actions = last_actions[:, : self.action_dim]
            else:
                pad = torch.zeros((total, self.action_dim - action_dim), device=device)
                last_actions = torch.cat([last_actions, pad], dim=1)

        aux = self.reward_proj(reward) + self.reset_proj(resets) + self.last_action_proj(last_actions)
        return aux

    def _pack_memory(self, memory: Optional[Dict[str, Optional[List[torch.Tensor]]]]) -> Optional[torch.Tensor]:
        if memory is None or self.memory_len <= 0:
            return None
        hidden_states = memory.get("hidden_states")
        if hidden_states is None or not hidden_states:
            return None
        packed_layers: List[torch.Tensor] = []
        for layer in hidden_states:
            if layer is None:
                return None
            layer_data = layer.transpose(0, 1)  # (batch, mem_len, hidden)
            current_len = layer_data.size(1)
            if current_len < self.memory_len:
                pad = torch.zeros(
                    layer_data.size(0),
                    self.memory_len - current_len,
                    layer_data.size(2),
                    device=layer_data.device,
                    dtype=layer_data.dtype,
                )
                layer_data = torch.cat([layer_data, pad], dim=1)
            elif current_len > self.memory_len:
                layer_data = layer_data[:, -self.memory_len :, :]
            packed_layers.append(layer_data)
        return torch.stack(packed_layers, dim=1)  # (batch, num_layers, mem_len, hidden)

    def _unpack_memory(
        self, packed: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> Optional[Dict[str, List[torch.Tensor]]]:
        if self.memory_len <= 0 or packed.numel() == 0:
            empty_layer = torch.zeros(0, packed.shape[0], self.hidden_size, device=device, dtype=dtype)
            return {"hidden_states": [empty_layer.clone() for _ in range(self.transformer_cfg.num_layers)]}
        packed_layers = packed.transpose(0, 1)  # (num_layers, batch, mem_len, hidden)
        hidden_states: List[torch.Tensor] = []
        for layer in packed_layers:
            if layer.size(2) > self.memory_len:
                layer = layer[:, -self.memory_len :, :]
            hidden_states.append(layer.transpose(0, 1).contiguous().to(device=device, dtype=dtype))
        return {"hidden_states": hidden_states}

    def _gather_memory_batch(
        self,
        env_ids: List[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, Optional[List[torch.Tensor]]]]:
        if not env_ids:
            return None
        base_memory = None
        for _batch_idx, env_id in enumerate(env_ids):
            env_memory = self._memory.get(env_id)
            if env_memory is None or env_memory.get("hidden_states") is None:
                env_memory = self.transformer_module.initialize_memory(1)
            hidden_states = env_memory.get("hidden_states")
            if hidden_states is None:
                return {"hidden_states": None}
            if base_memory is None:
                base_memory = []
                for layer in hidden_states:
                    if layer is None or layer.numel() == 0:
                        zeros = torch.zeros((0, 1, self.hidden_size), device=device, dtype=dtype)
                        base_memory.append(zeros)
                    else:
                        base_memory.append(layer[:, :1].detach().to(device=device, dtype=dtype).contiguous())
            else:
                for idx, layer in enumerate(hidden_states):
                    if layer is None or layer.numel() == 0:
                        addition = torch.zeros(
                            (base_memory[idx].size(0), 1, self.hidden_size), device=device, dtype=dtype
                        )
                    else:
                        addition = layer[:, :1].detach().to(device=device, dtype=dtype).contiguous()
                    base_memory[idx] = torch.cat([base_memory[idx], addition], dim=1)
        return {"hidden_states": base_memory}

    def _extract_env_id_list(self, td: TensorDict, batch_size: int) -> List[int]:
        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None or training_env_ids.numel() == 0:
            return list(range(batch_size))
        flat = training_env_ids.view(-1)
        if flat.numel() >= batch_size:
            return [int(flat[idx].item()) for idx in range(batch_size)]
        start = int(flat[0].item())
        return [start + idx for idx in range(batch_size)]

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
        aux_tokens = self._build_aux_tokens(td, batch_size, tt, latent.device)
        if aux_tokens is not None:
            latent = latent + aux_tokens

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
        env_ids = self._extract_env_id_list(td, batch_size) if use_memory else []
        device = latent.device
        dtype = latent.dtype

        if use_memory and tt == 1:
            memory_batch = self._gather_memory_batch(env_ids, batch_size, device, dtype)
            packed_memory = self._pack_memory(memory_batch)
            if packed_memory is not None:
                td.set("transformer_memory_pre", packed_memory.detach().to(dtype=torch.float16))
            core_out, new_memory = self.transformer_module(latent_seq, memory_batch)
        elif use_memory and tt > 1:
            packed_memory = td.get("transformer_memory_pre", None)
            if packed_memory is not None and packed_memory.numel() > 0:
                packed_memory = packed_memory.view(batch_size, tt, self.num_layers, self.memory_len, self.hidden_size)
            core_slices: list[torch.Tensor] = []
            for step in range(tt):
                if packed_memory is not None and self.memory_len > 0 and packed_memory.numel() > 0:
                    step_memory_tensor = packed_memory[:, step]
                    step_memory = self._unpack_memory(step_memory_tensor, device, dtype)
                else:
                    step_memory = None
                step_out, _ = self.transformer_module(latent_seq[step : step + 1], step_memory)
                core_slices.append(step_out)
            core_out = torch.cat(core_slices, dim=0)
            new_memory = None
        else:
            core_out, new_memory = self.transformer_module(latent_seq, None)
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

        if use_memory and tt == 1 and env_ids:
            updated_memory = self._detach_memory(new_memory)
            if updated_memory is not None:
                dones = td.get("dones", None)
                truncateds = td.get("truncateds", None)
                reset_mask = self._compute_reset_mask(dones, truncateds, batch_size)
                hidden_states = updated_memory.get("hidden_states")
                if hidden_states:
                    if reset_mask is not None and reset_mask.any():
                        if self._diag_enabled and self._diag_counter < self._diag_limit:
                            logger.info(
                                "[TRANSFORMER_DIAG] reset_mask true_count=%s",
                                int(reset_mask.sum().item()),
                            )
                        for idx_layer, layer_mem in enumerate(hidden_states):
                            if layer_mem is None or layer_mem.numel() == 0:
                                continue
                            masked_layer = layer_mem.clone()
                            masked_layer[:, reset_mask] = 0
                            hidden_states[idx_layer] = masked_layer
                    for batch_idx, env_id in enumerate(env_ids):
                        per_env_layers: List[torch.Tensor] = []
                        for layer_mem in hidden_states:
                            if layer_mem is None or layer_mem.numel() == 0:
                                per_env_layers.append(layer_mem)
                            else:
                                per_env_layers.append(layer_mem[:, batch_idx : batch_idx + 1].clone())
                        self._memory[env_id] = {"hidden_states": per_env_layers}
                if self._diag_enabled and self._diag_counter < self._diag_limit:
                    logger.info("[TRANSFORMER_DIAG] memory cached for envs %s", env_ids)
        elif use_memory and tt > 1 and env_ids:
            for env_id in env_ids:
                self._memory.pop(env_id, None)
            if self._diag_enabled and self._diag_counter < self._diag_limit:
                logger.info("[TRANSFORMER_DIAG] cleared cached memory for envs %s (tt=%s)", env_ids, tt)

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
        self._memory.clear()
        return [log] if log is not None else []

    def reset_memory(self) -> None:
        self._memory.clear()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_agent_experience_spec(self) -> Composite:
        spec = {
            "env_obs": UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            "dones": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            "truncateds": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        }
        if self.memory_len > 0:
            spec["transformer_memory_pre"] = UnboundedDiscrete(
                shape=torch.Size([self.transformer_cfg.num_layers, self.memory_len, self.hidden_size]),
                dtype=torch.float16,
            )
        return Composite(**spec)


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
