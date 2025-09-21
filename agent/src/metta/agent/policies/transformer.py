"""Transformer policies that mirror legacy PyTorch agents exactly."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple, Type

import pufferlib.pytorch
import torch
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn
from torchrl.data import Composite, UnboundedDiscrete

from metta.agent.components.action import ActionEmbedding, ActionEmbeddingConfig
from metta.agent.components.actor import (
    ActionProbs,
    ActionProbsConfig,
    ActorKey,
    ActorKeyConfig,
    ActorQuery,
    ActorQueryConfig,
)
from metta.agent.components.transformer_module import TransformerModule
from metta.agent.components.transformer_nvidia_module import NvidiaTransformerModule
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


class TransformerPolicyConfig(PolicyArchitecture):
    """Hyperparameters for the legacy convolutional Transformer policy."""

    class_path: str = "metta.agent.policies.transformer.TransformerPolicy"

    # CNN encoder parameters
    cnn1_out_channels: int = 64
    cnn1_kernel_size: int = 5
    cnn1_stride: int = 3
    cnn2_out_channels: int = 128
    cnn2_kernel_size: int = 3
    cnn2_stride: int = 1
    fc1_features: int = 512

    # Transformer interface dimensions
    latent_size: int = 256
    hidden_size: int = 256

    # Transformer hyperparameters
    transformer_num_layers: int = 6
    transformer_num_heads: int = 8
    transformer_ff_size: int = 512
    transformer_max_seq_len: int = 256
    transformer_memory_len: int = 64
    transformer_dropout: float = 0.1
    transformer_attn_dropout: float = 0.1
    transformer_clamp_len: int = -1
    transformer_module_cls: Type[nn.Module] = TransformerModule

    # Actor / critic head dimensions
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    action_embedding_dim: int = 16
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    # Implementation options
    manual_init: bool = False
    strict_attr_indices: bool = False


class TransformerImprovedConfig(TransformerPolicyConfig):
    """Matches the legacy Transformer-XL agent implementation."""

    class_path: str = "metta.agent.policies.transformer.TransformerImprovedPolicy"
    transformer_ff_size: int = 1024


class TransformerNvidiaConfig(TransformerPolicyConfig):
    """Matches NVIDIA's reference Transformer-XL implementation."""

    class_path: str = "metta.agent.policies.transformer.TransformerNvidiaPolicy"
    transformer_ff_size: int = 512
    transformer_memory_len: int = 32
    transformer_dropout: float = 0.1
    transformer_attn_dropout: float = 0.1
    transformer_clamp_len: int = 256
    transformer_module_cls: Type[nn.Module] = NvidiaTransformerModule
    manual_init: bool = True
    strict_attr_indices: bool = True


class TransformerPolicy(Policy):
    """CNN + Transformer policy as implemented in legacy PyTorch agents."""

    def __init__(self, env, config: Optional[TransformerPolicyConfig] = None) -> None:
        super().__init__()
        self.config = config or TransformerPolicyConfig()

        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.out_width = env.obs_width
        self.out_height = env.obs_height
        self.num_layers = max(env.feature_normalizations.keys()) + 1
        self.dim_per_layer = self.out_width * self.out_height

        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size

        self._build_encoder()
        self._build_heads()
        self._build_transformer()

        self._memory: Dict[int, Optional[Dict[str, Optional[List[torch.Tensor]]]]] = {}
        self.register_buffer("max_vec", self._compute_feature_normalization(env), persistent=True)

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def _build_encoder(self) -> None:
        manual = self.config.manual_init

        if manual:
            self.cnn1 = nn.Conv2d(
                self.num_layers,
                self.config.cnn1_out_channels,
                kernel_size=self.config.cnn1_kernel_size,
                stride=self.config.cnn1_stride,
            )
            self.cnn2 = nn.Conv2d(
                self.config.cnn1_out_channels,
                self.config.cnn2_out_channels,
                kernel_size=self.config.cnn2_kernel_size,
                stride=self.config.cnn2_stride,
            )
            nn.init.orthogonal_(self.cnn1.weight, 1.0)
            nn.init.zeros_(self.cnn1.bias)
            nn.init.orthogonal_(self.cnn2.weight, 1.0)
            nn.init.zeros_(self.cnn2.bias)
        else:
            self.cnn1 = pufferlib.pytorch.layer_init(
                nn.Conv2d(
                    self.num_layers,
                    self.config.cnn1_out_channels,
                    kernel_size=self.config.cnn1_kernel_size,
                    stride=self.config.cnn1_stride,
                ),
                std=1.0,
            )
            self.cnn2 = pufferlib.pytorch.layer_init(
                nn.Conv2d(
                    self.config.cnn1_out_channels,
                    self.config.cnn2_out_channels,
                    kernel_size=self.config.cnn2_kernel_size,
                    stride=self.config.cnn2_stride,
                ),
                std=1.0,
            )

        self.flatten = nn.Flatten()
        flattened_size = self._compute_flattened_size(
            (self.out_height, self.out_width),
            self.config.cnn1_kernel_size,
            self.config.cnn1_stride,
            self.config.cnn2_kernel_size,
            self.config.cnn2_stride,
            self.config.cnn2_out_channels,
        )

        if manual:
            self.fc1 = nn.Linear(flattened_size, self.config.fc1_features)
            nn.init.orthogonal_(self.fc1.weight, math.sqrt(2))
            nn.init.zeros_(self.fc1.bias)
            self.encoded_obs = nn.Linear(self.config.fc1_features, self.latent_size)
            nn.init.orthogonal_(self.encoded_obs.weight, math.sqrt(2))
            nn.init.zeros_(self.encoded_obs.bias)
        else:
            self.fc1 = pufferlib.pytorch.layer_init(nn.Linear(flattened_size, self.config.fc1_features), std=1.0)
            self.encoded_obs = pufferlib.pytorch.layer_init(
                nn.Linear(self.config.fc1_features, self.latent_size), std=1.0
            )

        if self.latent_size != self.hidden_size:
            if manual:
                self.input_projection = nn.Linear(self.latent_size, self.hidden_size)
                nn.init.orthogonal_(self.input_projection.weight, 1.0)
                nn.init.zeros_(self.input_projection.bias)
            else:
                self.input_projection = pufferlib.pytorch.layer_init(
                    nn.Linear(self.latent_size, self.hidden_size), std=1.0
                )
        else:
            self.input_projection = nn.Identity()

    def _build_heads(self) -> None:
        manual = self.config.manual_init

        if manual:
            self.critic_1 = nn.Linear(self.hidden_size, self.config.critic_hidden_dim)
            nn.init.orthogonal_(self.critic_1.weight, math.sqrt(2))
            nn.init.zeros_(self.critic_1.bias)
            self.value_head = nn.Linear(self.config.critic_hidden_dim, 1)
            nn.init.orthogonal_(self.value_head.weight, 1.0)
            nn.init.zeros_(self.value_head.bias)

            self.actor_1 = nn.Linear(self.hidden_size, self.config.actor_hidden_dim)
            nn.init.orthogonal_(self.actor_1.weight, 1.0)
            nn.init.zeros_(self.actor_1.bias)
        else:
            self.critic_1 = pufferlib.pytorch.layer_init(
                nn.Linear(self.hidden_size, self.config.critic_hidden_dim), std=math.sqrt(2)
            )
            self.value_head = pufferlib.pytorch.layer_init(nn.Linear(self.config.critic_hidden_dim, 1), std=1.0)
            self.actor_1 = pufferlib.pytorch.layer_init(
                nn.Linear(self.hidden_size, self.config.actor_hidden_dim), std=1.0
            )

        self.critic_activation = nn.Tanh()

        self.actor_module = TDM(self.actor_1, in_keys=["core"], out_keys=["actor_1"])
        self.critic_module = TDM(self.critic_1, in_keys=["core"], out_keys=["critic_1"])
        self.value_module = TDM(self.value_head, in_keys=["critic_1"], out_keys=["values"])

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
        module_cls = self.config.transformer_module_cls
        clamp_len = self.config.transformer_clamp_len
        if clamp_len < 0 and module_cls is NvidiaTransformerModule:
            clamp_len = self.config.transformer_max_seq_len

        self.transformer_module = module_cls(
            d_model=self.hidden_size,
            n_heads=self.config.transformer_num_heads,
            n_layers=self.config.transformer_num_layers,
            d_ff=self.config.transformer_ff_size,
            max_seq_len=self.config.transformer_max_seq_len,
            memory_len=self.config.transformer_memory_len,
            dropout=self.config.transformer_dropout,
            dropatt=self.config.transformer_attn_dropout,
            pre_lnorm=True,
            clamp_len=clamp_len,
            attn_type=0,
        )

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    @torch._dynamo.disable
    def forward(self, td: TensorDict, state=None, action: Optional[torch.Tensor] = None):
        observations = td["env_obs"]

        if observations.dim() == 4:
            B, TT = observations.shape[:2]
            total_batch = B * TT
            batch_shape: Optional[Tuple[int, int]] = (B, TT)
            if td.batch_dims > 1:
                td = td.reshape(total_batch)
                observations = td["env_obs"]
        else:
            B = observations.shape[0]
            TT = 1
            total_batch = B
            batch_shape = None
            if td.batch_dims > 1:
                td = td.reshape(total_batch)
                observations = td["env_obs"]

        device = observations.device
        td.set("bptt", torch.full((total_batch,), TT, device=device, dtype=torch.long))
        td.set("batch", torch.full((total_batch,), B, device=device, dtype=torch.long))

        latent = self._encode_observations(observations)
        latent = self.input_projection(latent)

        core = self._forward_transformer(td, latent, B, TT)
        td["core"] = core

        self.actor_module(td)
        td["actor_1"] = torch.relu(td["actor_1"])

        self.critic_module(td)
        td["critic_1"] = self.critic_activation(td["critic_1"])
        self.value_module(td)
        td["values"] = td["values"].flatten()

        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td)

        td = self.action_probs(td, action)

        if batch_shape is not None:
            td = td.reshape(batch_shape)
        return td

    def _forward_transformer(self, td: TensorDict, latent: torch.Tensor, batch_size: int, tt: int) -> torch.Tensor:
        latent_seq = latent.view(batch_size, tt, self.hidden_size).transpose(0, 1)

        memory = None
        env_key: Optional[int] = None
        if tt == 1:
            env_key = self._get_env_start(td)
            memory = self._memory.get(env_key)

        core_out, new_memory = self.transformer_module(latent_seq, memory)
        core_flat = core_out.transpose(0, 1).reshape(batch_size * tt, self.hidden_size)

        if tt == 1 and env_key is not None:
            updated_memory = self._detach_memory(new_memory)
            if updated_memory is not None:
                dones = td.get("dones", None)
                truncateds = td.get("truncateds", None)
                if dones is not None and truncateds is not None:
                    reset_mask = self._compute_reset_mask(dones, truncateds, batch_size)
                    if reset_mask is not None and reset_mask.any():
                        hidden_states = updated_memory.get("hidden_states")
                        if hidden_states:
                            for idx, layer_mem in enumerate(hidden_states):
                                if layer_mem is None or layer_mem.numel() == 0:
                                    continue
                                masked_layer = layer_mem.clone()
                                masked_layer[:, reset_mask] = 0
                                hidden_states[idx] = masked_layer
                self._memory[env_key] = updated_memory
        elif tt > 1 and env_key is not None:
            self._memory.pop(env_key, None)

        return core_flat

    # ------------------------------------------------------------------
    # Observation encoding helpers
    # ------------------------------------------------------------------
    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != 3:
            raise ValueError(f"Expected 3-token channels, got {observations.shape}")

        coords_byte = observations[..., 0].to(torch.uint8)
        x_coord = ((coords_byte >> 4) & 0x0F).long()
        y_coord = (coords_byte & 0x0F).long()
        attr_indices = observations[..., 1].long()
        attr_values = observations[..., 2].float()

        valid_tokens = coords_byte != 0xFF
        valid_attr = attr_indices < self.num_layers
        valid_mask = valid_tokens & valid_attr

        invalid_mask = valid_tokens & ~valid_attr
        if invalid_mask.any():
            invalid_indices = attr_indices[invalid_mask].unique()
            if self.config.strict_attr_indices:
                raise ValueError(
                    f"Found observation attribute indices {sorted(map(int, invalid_indices.tolist()))} "
                    f">= num_layers ({self.num_layers})."
                )
            warnings.warn(
                f"Found observation attribute indices {sorted(map(int, invalid_indices.tolist()))} "
                f">= num_layers ({self.num_layers}). These tokens will be ignored.",
                stacklevel=2,
            )

        combined_index = attr_indices * self.dim_per_layer + x_coord * self.out_height + y_coord

        safe_index = torch.where(valid_mask, combined_index, torch.zeros_like(combined_index))
        safe_values = torch.where(valid_mask, attr_values, torch.zeros_like(attr_values))

        box_flat = torch.zeros(
            (observations.shape[0], self.num_layers * self.dim_per_layer),
            dtype=attr_values.dtype,
            device=observations.device,
        )
        # Use additive scatter so padding tokens (value 0) cannot overwrite real data.
        box_flat.scatter_add_(1, safe_index, safe_values)

        box_obs = box_flat.view(observations.shape[0], self.num_layers, self.out_width, self.out_height)
        x = box_obs / self.max_vec

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        encoded = F.relu(self.encoded_obs(x))
        return encoded

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _compute_flattened_size(
        self,
        input_hw: Tuple[int, int],
        conv1_kernel,
        conv1_stride,
        conv2_kernel,
        conv2_stride,
        conv2_channels,
    ) -> int:
        def conv_out(size, kernel, stride):
            return math.floor((size - kernel) / stride + 1)

        h1 = conv_out(input_hw[0], conv1_kernel, conv1_stride)
        w1 = conv_out(input_hw[1], conv1_kernel, conv1_stride)
        h2 = conv_out(h1, conv2_kernel, conv2_stride)
        w2 = conv_out(w1, conv2_kernel, conv2_stride)
        return conv2_channels * h2 * w2

    def _compute_feature_normalization(self, env) -> torch.Tensor:
        max_values = torch.ones(self.num_layers, dtype=torch.float32)
        for feature_id, norm_value in env.feature_normalizations.items():
            if feature_id < self.num_layers:
                max_values[feature_id] = norm_value if norm_value > 0 else 1.0
        return max_values.view(1, self.num_layers, 1, 1)

    def _compute_reset_mask(
        self, dones: torch.Tensor, truncateds: torch.Tensor, batch_size: int
    ) -> Optional[torch.Tensor]:
        if dones.numel() == 0 or truncateds.numel() == 0:
            return None
        try:
            dones = rearrange(dones, "(b t) -> t b", b=batch_size)
            truncateds = rearrange(truncateds, "(b t) -> t b", b=batch_size)
        except ValueError:
            dones = dones.view(1, batch_size)
            truncateds = truncateds.view(1, batch_size)
        reset = dones.bool() | truncateds.bool()
        return reset[-1]

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
            return int(training_env_ids.reshape(-1)[0].item())
        training_env_id = td.get("training_env_id", None)
        if training_env_id is not None and training_env_id.numel() > 0:
            return int(training_env_id.reshape(-1)[0].item())
        return 0

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------
    def initialize_to_environment(self, env, device):
        device = torch.device(device)
        self.to(device)

        max_vec = self._compute_feature_normalization(env).to(device=device)
        if self.max_vec.device != device or self.max_vec.shape != max_vec.shape:
            self.max_vec = max_vec
        else:
            with torch.no_grad():
                self.max_vec.copy_(max_vec)

        self.action_embeddings.initialize_to_environment(env, device)
        self.action_probs.initialize_to_environment(env, device)
        self._memory.clear()
        return []

    def reset_memory(self) -> None:
        self._memory.clear()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )


class TransformerImprovedPolicy(TransformerPolicy):
    def __init__(self, env, config: Optional[TransformerImprovedConfig] = None) -> None:
        super().__init__(env, config or TransformerImprovedConfig())


class TransformerNvidiaPolicy(TransformerPolicy):
    def __init__(self, env, config: Optional[TransformerNvidiaConfig] = None) -> None:
        super().__init__(env, config or TransformerNvidiaConfig())
