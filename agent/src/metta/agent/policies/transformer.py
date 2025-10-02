"""Transformer policies built on the Metta transformer backbones."""

from __future__ import annotations

import contextlib
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from pydantic import model_validator
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import Composite, UnboundedContinuous, UnboundedDiscrete

import pufferlib.pytorch
from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.actor import (
    ActionProbsConfig,
    ActorKey,
    ActorKeyConfig,
    ActorQuery,
    ActorQueryConfig,
)
from metta.agent.components.obs_enc import ObsPerceiverLatent, ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokens, ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourier, ObsAttrEmbedFourierConfig
from metta.agent.components.transformer_core import TransformerBackboneConfig
from metta.agent.components.transformers import get_backbone_spec
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


class TransformerPolicyConfig(PolicyArchitecture):
    """Configures the end-to-end transformer policy."""

    class_path: str = "metta.agent.policies.transformer.TransformerPolicy"

    variant: str = "gtrxl"

    # Observation preprocessing
    obs_shim_config: ObsShimTokensConfig = ObsShimTokensConfig(in_key="env_obs", out_key="obs_tokens", max_tokens=48)
    obs_tokenizer: ObsAttrEmbedFourierConfig = ObsAttrEmbedFourierConfig(
        in_key="obs_tokens",
        out_key="obs_attr_embed",
        num_freqs=3,
        attr_embed_dim=8,
    )

    obs_encoder: ObsPerceiverLatentConfig = ObsPerceiverLatentConfig(
        in_key="obs_attr_embed",
        out_key="encoded_obs",
        feat_dim=8 + (4 * 3) + 1,
        latent_dim=32,
        num_latents=12,
        num_heads=4,
        num_layers=1,
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

        entry = get_backbone_spec(self.variant)
        defaults = entry.policy_defaults
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
        self._uses_sliding_backbone = self.transformer_cfg.variant == "sliding"
        self.strict_attr_indices = getattr(self.config, "strict_attr_indices", False)
        self.use_aux_tokens = getattr(self.config, "use_aux_tokens", False)
        self.num_layers = max(env.feature_normalizations.keys()) + 1
        self._memory_len = int(getattr(self.transformer_cfg, "memory_len", 0) or 0)
        self._transformer_layers = int(getattr(self.transformer_cfg, "num_layers", 0) or 0)
        self._memory_len_initial = self._memory_len

        encoder_out = self.config.obs_encoder.latent_dim
        if encoder_out != self.latent_size:
            logger.info(
                "Adjusting token encoder latent dim from %s to match transformer latent size %s.",
                encoder_out,
                self.latent_size,
            )
            self.config.obs_encoder.latent_dim = self.latent_size

        self.obs_shim = ObsShimTokens(env, config=self.config.obs_shim_config)
        self.obs_tokenizer = ObsAttrEmbedFourier(config=self.config.obs_tokenizer)
        self.obs_encoder = ObsPerceiverLatent(config=self.config.obs_encoder)

        self._build_heads()
        self._build_transformer()

        self._memory_tensor: torch.Tensor | None = None
        self._memory: Dict[int, Optional[torch.Tensor]] = {}
        self._aux_zero_cache: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}

        self._diag_enabled = os.getenv("TRANSFORMER_DIAG", "0") == "1"
        self._diag_limit = int(os.getenv("TRANSFORMER_DIAG_STEPS", "5"))
        self._diag_counter = 0
        self._diag_train_logged = False

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            self._autocast_enabled = True
            self._autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._autocast_enabled = False
            self._autocast_dtype = torch.float16

        if self.latent_size != self.hidden_size:
            self.input_projection = pufferlib.pytorch.layer_init(nn.Linear(self.latent_size, self.hidden_size), std=1.0)
        else:
            self.input_projection = nn.Identity()

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

        def _init_linear(linear: nn.Linear, gain: float) -> nn.Linear:
            if manual:
                nn.init.orthogonal_(linear.weight, gain)
                nn.init.zeros_(linear.bias)
            else:
                linear = pufferlib.pytorch.layer_init(linear, std=gain)
            return linear

        actor_linear = _init_linear(
            nn.Linear(self.hidden_size, self.config.actor_hidden_dim), gain=1.0
        )
        self.actor_head = TensorDictModule(
            nn.Sequential(actor_linear, nn.ReLU()),
            in_keys=["core"],
            out_keys=["actor_1"],
        )

        critic_linear = _init_linear(
            nn.Linear(self.hidden_size, self.config.critic_hidden_dim), gain=math.sqrt(2)
        )
        self.critic_head = TensorDictModule(
            nn.Sequential(critic_linear, nn.Tanh()),
            in_keys=["core"],
            out_keys=["critic_1"],
        )

        value_linear = _init_linear(nn.Linear(self.config.critic_hidden_dim, 1), gain=1.0)
        self.value_head = TensorDictModule(
            value_linear,
            in_keys=["critic_1"],
            out_keys=["values"],
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
        self._memory_enabled = self.memory_len > 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    @property
    def cnn1(self) -> nn.Module:
        """Expose first CNN layer for downstream tests and diagnostics."""

        raise AttributeError("cnn1 is not available when using token-based encoder")

    def _encode_observations(self, observations: torch.Tensor) -> TensorDict:
        """Run raw observations through preprocessing and CNN encoder."""

        if observations.dim() != 3:
            raise ValueError("Expected observations with shape (batch, tokens, features).")

        batch_size = observations.shape[0]
        td = TensorDict({"env_obs": observations}, batch_size=[batch_size])
        self.obs_shim(td)
        self.obs_tokenizer(td)
        self.obs_encoder(td)
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
        proj_dtype = self.reward_proj.weight.dtype
        action_dtype = self.last_action_proj.weight.dtype

        reward = td.get("rewards", None)
        if reward is None:
            reward = self._get_zero_buffer((total, 1), device, proj_dtype)
        else:
            reward = reward.view(total, -1).to(device=device, dtype=proj_dtype)
            if reward.size(1) != 1:
                reward = reward[:, :1]

        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is None and truncateds is None:
            resets = self._get_zero_buffer((total, 1), device, proj_dtype)
        else:
            if dones is None:
                dones = torch.zeros_like(truncateds)
            if truncateds is None:
                truncateds = torch.zeros_like(dones)
            resets = torch.logical_or(dones.bool(), truncateds.bool()).view(total, -1)
            resets = resets.to(device=device, dtype=proj_dtype)
            if resets.size(1) != 1:
                resets = resets[:, :1]

        last_actions = td.get("last_actions", None)
        if last_actions is not None:
            last_actions = last_actions.view(total, -1).to(device=device, dtype=action_dtype)
        else:
            actions = td.get("actions", None)
            if actions is not None:
                actions = actions.view(batch_size, tt, -1).to(device=device, dtype=action_dtype)
                prev_actions = self._get_zero_buffer((batch_size, tt, actions.size(-1)), device, action_dtype)
                if tt > 1:
                    prev_actions[:, 1:] = actions[:, :-1]
                last_actions = prev_actions.view(total, -1)
            else:
                last_actions = self._get_zero_buffer((total, self.action_dim), device, action_dtype)

        if last_actions.size(1) != self.action_dim:
            action_dim = last_actions.size(1)
            if action_dim > self.action_dim:
                last_actions = last_actions[:, : self.action_dim]
            else:
                pad = self._get_zero_buffer((total, self.action_dim - action_dim), device, action_dtype)
                last_actions = torch.cat([last_actions, pad], dim=1)

        aux = self.reward_proj(reward)
        aux.add_(self.reset_proj(resets))
        aux.add_(self.last_action_proj(last_actions))
        return aux

    def _get_zero_buffer(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype, shape)
        buf = self._aux_zero_cache.get(key)
        if buf is None or buf.shape != shape or buf.device != device or buf.dtype != dtype:
            buf = torch.zeros(shape, device=device, dtype=dtype)
            self._aux_zero_cache[key] = buf
        else:
            buf.zero_()
        return buf

    def _pack_memory(self, memory: Optional[Dict[str, Optional[List[torch.Tensor]]]]) -> Optional[torch.Tensor]:
        if memory is None or self.memory_len <= 0:
            return None
        hidden_states = memory.get("hidden_states")
        if not hidden_states:
            return None
        stacked = torch.stack(hidden_states, dim=0)  # [num_layers, mem_len, batch, hidden]
        stacked = stacked[-self.transformer_layers :]
        stacked = stacked[:, -self.memory_len :, :, :]
        packed = stacked.permute(2, 0, 1, 3).contiguous()  # [batch, num_layers, mem_len, hidden]
        return packed.to(dtype=torch.float32)

    def _ensure_memory_capacity(self, capacity: int, device: torch.device, dtype: torch.dtype) -> None:
        if capacity <= 0 or self.memory_len <= 0 or self.transformer_layers <= 0:
            return
        if self._memory_tensor is None:
            self._memory_tensor = torch.zeros(
                capacity,
                self.transformer_layers,
                self.memory_len,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
            return
        if self._memory_tensor.device != device or self._memory_tensor.dtype != dtype:
            self._memory_tensor = self._memory_tensor.to(device=device, dtype=dtype)
        if self._memory_tensor.size(0) < capacity:
            pad = torch.zeros(
                capacity - self._memory_tensor.size(0),
                self.transformer_layers,
                self.memory_len,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )
            self._memory_tensor = torch.cat([self._memory_tensor, pad], dim=0)

    def _unpack_memory(
        self, packed: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> Optional[Dict[str, List[torch.Tensor]]]:
        if self.memory_len <= 0 or packed.numel() == 0:
            return {"hidden_states": None}
        if packed.dim() != 4:
            raise ValueError(
                "transformer_memory_pre must have shape (batch, num_layers, memory_len, hidden_size)"
            )
        packed = packed.to(device=device, dtype=dtype)
        packed = packed[:, : self.transformer_layers, -self.memory_len :, :]
        unpacked = packed.permute(1, 2, 0, 3).contiguous()  # [num_layers, mem_len, batch, hidden]
        hidden_states = [layer for layer in unpacked]
        return {"hidden_states": hidden_states}

    def _gather_memory_batch(
        self,
        env_ids: List[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Dict[str, Optional[List[torch.Tensor]]]]:
        if not env_ids or self.memory_len <= 0 or self.transformer_layers <= 0:
            return {"hidden_states": None}

        max_env_id = max(env_ids) + 1
        self._ensure_memory_capacity(max_env_id, device, dtype)
        if self._memory_tensor is None or self._memory_tensor.numel() == 0:
            return {"hidden_states": None}

        index = torch.tensor(env_ids, dtype=torch.long, device=self._memory_tensor.device)
        memory_slice = self._memory_tensor.index_select(0, index)  # [batch, layers, mem_len, hidden]
        if memory_slice.numel() == 0:
            return {"hidden_states": None}

        memory_slice = memory_slice.to(device=device, dtype=dtype)
        memory_slice = memory_slice.permute(1, 2, 0, 3).contiguous()  # [layers, mem_len, batch, hidden]
        hidden_states = [layer for layer in memory_slice]
        return {"hidden_states": hidden_states}

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

        device_type = self.device.type
        use_autocast = self._autocast_enabled and device_type == "cuda"
        autocast_ctx = (
            torch.autocast(device_type=device_type, dtype=self._autocast_dtype)
            if use_autocast
            else contextlib.nullcontext()
        )

        with autocast_ctx:
            self.obs_shim(td)
            self.obs_tokenizer(td)
            self.obs_encoder(td)

            encoded_key = self.config.obs_encoder.out_key
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
        self._cast_floating_tensors(td)
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

    def _cast_floating_tensors(self, td: TensorDict) -> None:
        for key, value in td.items(include_nested=True):
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
                leaf_key = key[-1] if isinstance(key, tuple) else key
                if leaf_key == "transformer_memory_pre":
                    continue
                td[key] = value.to(dtype=torch.float32)

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
            if self.memory_len > 0:
                packed_memory = self._pack_memory(memory_batch)
                if packed_memory is not None:
                    td.set(
                        "transformer_memory_pre",
                        packed_memory.detach().to(dtype=torch.float32, device=torch.device("cpu")),
                    )
            core_out, new_memory = self.transformer_module(latent_seq, memory_batch)
        elif use_memory and tt > 1:
            packed_memory = td.get("transformer_memory_pre", None)
            memory_batch = None
            if self.memory_len > 0 and packed_memory is not None and packed_memory.numel() > 0:
                packed_memory = packed_memory.view(
                    batch_size,
                    tt,
                    self.transformer_layers,
                    self.memory_len,
                    self.hidden_size,
                )
                initial_memory = packed_memory[:, 0]
                memory_batch = self._unpack_memory(initial_memory, device, dtype)

            core_out, _ = self.transformer_module(latent_seq, memory_batch)
            new_memory = None
        else:
            core_out, new_memory = self.transformer_module(latent_seq, None)
        core_flat = core_out.transpose(0, 1).reshape(batch_size * tt, self.hidden_size)

        if self._diag_enabled and self._diag_counter < self._diag_limit:
            logger.info("[TRANSFORMER_DIAG] %s", _tensor_stats(core_flat, "core"))
            if new_memory is not None and isinstance(new_memory, dict):
                hidden_states = new_memory.get("hidden_states")
                if hidden_states:
                    norms = [
                        float(layer.float().norm().item()) if layer is not None and layer.numel() > 0 else 0.0
                        for layer in hidden_states
                    ]
                    logger.info("[TRANSFORMER_DIAG] memory_norms=%s", norms)

        if use_memory and env_ids:
            dones = td.get("dones", None)
            truncateds = td.get("truncateds", None)
            reset_mask = self._compute_reset_mask(dones, truncateds, batch_size)

            if tt == 1:
                updated_memory = self._detach_memory(new_memory)
                hidden_states = None
                if updated_memory is not None and isinstance(updated_memory, dict):
                    hidden_states = updated_memory.get("hidden_states")

                if hidden_states:
                    stacked = torch.stack(hidden_states, dim=0)  # [layers, mem_len, batch, hidden]
                    env_tensor = stacked.permute(2, 0, 1, 3).contiguous()  # [batch, layers, mem_len, hidden]
                    env_tensor = env_tensor[:, :, -self.memory_len :, :]
                    if env_tensor.size(2) < self.memory_len:
                        pad_len = self.memory_len - env_tensor.size(2)
                        pad = torch.zeros(
                            env_tensor.size(0),
                            env_tensor.size(1),
                            pad_len,
                            env_tensor.size(3),
                            device=env_tensor.device,
                            dtype=env_tensor.dtype,
                        )
                        env_tensor = torch.cat([pad, env_tensor], dim=2)
                    self._ensure_memory_capacity(max(env_ids) + 1, env_tensor.device, env_tensor.dtype)
                    if self._memory_tensor is not None:
                        index = torch.tensor(env_ids, dtype=torch.long, device=self._memory_tensor.device)
                        self._memory_tensor.index_copy_(
                            0,
                            index,
                            env_tensor.to(device=self._memory_tensor.device, dtype=self._memory_tensor.dtype),
                        )

                if reset_mask is not None and reset_mask.any() and self._memory_tensor is not None:
                    reset_indices = [env_ids[i] for i, flag in enumerate(reset_mask.cpu().tolist()) if flag]
                    if reset_indices:
                        index = torch.tensor(reset_indices, dtype=torch.long, device=self._memory_tensor.device)
                        self._memory_tensor.index_fill_(0, index, 0.0)

                if self._diag_enabled and self._diag_counter < self._diag_limit:
                    logger.info("[TRANSFORMER_DIAG] memory cached for envs %s", env_ids)
            else:
                if self._memory_tensor is not None:
                    index = torch.tensor(env_ids, dtype=torch.long, device=self._memory_tensor.device)
                    self._memory_tensor.index_fill_(0, index, 0.0)
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
        hidden_states = memory.get("hidden_states") if isinstance(memory, dict) else None
        if not hidden_states:
            return {"hidden_states": None}
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
        self.clear_memory()
        return [log] if log is not None else []

    def reset_memory(self) -> None:
        # Transformer policies rely on cached hidden state across minibatches; clearing
        # here would defeat that behaviour. Keep as a no-op to mirror the sliding
        # transformer fix on main.
        return

    def clear_memory(self) -> None:
        """Explicitly clear cached transformer memory."""

        self._memory_tensor = None
        self._memory.clear()

    def update_memory_len(self, new_len: int) -> None:
        new_len = int(new_len)
        if new_len < 0:
            raise ValueError("memory length must be non-negative")
        if new_len > self._memory_len_initial:
            raise ValueError(
                f"memory length {new_len} exceeds initial allocation {self._memory_len_initial}"
            )
        if new_len == self._memory_len:
            return

        self._memory_len = new_len
        self.transformer_cfg.memory_len = new_len
        if hasattr(self.transformer_module, "memory_len"):
            self.transformer_module.memory_len = new_len  # type: ignore[assignment]
        core = getattr(self.transformer_module, "core", None)
        if core is not None and hasattr(core, "mem_len"):
            core.mem_len = new_len  # type: ignore[assignment]
        self._memory_tensor = None
        self._memory.clear()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def memory_len(self) -> int:
        return getattr(self, "_memory_len", 0)

    @property
    def transformer_layers(self) -> int:
        return getattr(self, "_transformer_layers", 0)

    def get_agent_experience_spec(self) -> Composite:
        spec = {
            "env_obs": UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            "dones": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            "truncateds": UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        }
        if self.memory_len > 0:
            spec["transformer_memory_pre"] = UnboundedContinuous(
                shape=torch.Size([self.transformer_layers, self.memory_len, self.hidden_size]),
                dtype=torch.float32,
            )
        return Composite(**spec)


def gtrxl_policy_config() -> TransformerPolicyConfig:
    """Return a policy config for the GTrXL variant."""

    return TransformerPolicyConfig(variant="gtrxl")


def trxl_policy_config() -> TransformerPolicyConfig:
    """Return a policy config for the vanilla Transformer-XL variant."""

    return TransformerPolicyConfig(variant="trxl")


def trxl_nvidia_policy_config() -> TransformerPolicyConfig:
    """Return a policy config for the NVIDIA Transformer-XL variant."""

    return TransformerPolicyConfig(variant="trxl_nvidia")


__all__ = [
    "TransformerPolicyConfig",
    "TransformerPolicy",
    "gtrxl_policy_config",
    "trxl_policy_config",
    "trxl_nvidia_policy_config",
]
