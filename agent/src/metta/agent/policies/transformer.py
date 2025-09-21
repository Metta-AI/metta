"""Transformer policies that mirror legacy PyTorch agents exactly."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Type

import pufferlib.pytorch
import torch
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
from metta.agent.components.cnn_encoder import CNNEncoder, CNNEncoderConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimBoxConfig
from metta.agent.components.transformer_module import TransformerModule
from metta.agent.components.transformer_nvidia_module import NvidiaTransformerModule
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


class TransformerPolicyConfig(PolicyArchitecture):
    """Hyperparameters for the legacy convolutional Transformer policy."""

    class_path: str = "metta.agent.policies.transformer.TransformerPolicy"

    # Observation preprocessing
    obs_shim_config: ObsShimBoxConfig = ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig(
        in_key="obs_normalizer",
        out_key="encoded_obs",
        cnn1_cfg={"out_channels": 64, "kernel_size": 5, "stride": 3},
        cnn2_cfg={"out_channels": 128, "kernel_size": 3, "stride": 1},
        fc1_cfg={"out_features": 512},
        encoded_obs_cfg={"out_features": 256},
    )

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


class TransformerPolicy(Policy):
    """CNN + Transformer policy as implemented in legacy PyTorch agents."""

    def __init__(self, env, config: Optional[TransformerPolicyConfig] = None) -> None:
        super().__init__()
        self.config = config or TransformerPolicyConfig()

        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size

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

        if self.latent_size != self.hidden_size:
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
            self.value_head = pufferlib.pytorch.layer_init(
                nn.Linear(self.config.critic_hidden_dim, 1), std=1.0
            )
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
        original_shape: Optional[torch.Size]
        if td.batch_dims > 1:
            original_shape = td.batch_size
            total_batch = original_shape.numel()
            td = td.reshape(total_batch)

            env_obs = td.get("env_obs", None)
            device = env_obs.device if env_obs is not None else torch.device("cpu")
            if "bptt" not in td.keys() and len(original_shape) >= 2:
                td.set(
                    "bptt",
                    torch.full((total_batch,), int(original_shape[1]), device=device, dtype=torch.long),
                )
            if "batch" not in td.keys() and len(original_shape) >= 1:
                td.set(
                    "batch",
                    torch.full((total_batch,), int(original_shape[0]), device=device, dtype=torch.long),
                )
        else:
            original_shape = None

        self.obs_shim(td)
        self.cnn_encoder(td)

        encoded_key = self.config.cnn_encoder_config.out_key
        latent = td[encoded_key]
        latent = self.input_projection(latent)

        core = self._forward_transformer(td, latent, original_shape)
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
        if original_shape is not None:
            td = td.reshape(original_shape)
        return td

    def _forward_transformer(
        self, td: TensorDict, latent: torch.Tensor, original_shape: Optional[torch.Size]
    ) -> torch.Tensor:
        tt: Optional[int] = None
        batch_size: Optional[int] = None

        if "bptt" in td.keys():
            tt = int(td["bptt"].reshape(-1)[0].item())
        elif original_shape is not None and len(original_shape) >= 2:
            tt = int(original_shape[1])
        else:
            tt = 1

        if tt <= 0:
            raise ValueError("bptt entries must be positive")

        total_batch = latent.shape[0]
        if total_batch % tt != 0:
            raise ValueError("encoded_obs batch dimension must be divisible by bptt")

        if "batch" in td.keys():
            batch_size = int(td["batch"].reshape(-1)[0].item())
        elif original_shape is not None:
            batch_size = int(total_batch // tt)
        else:
            batch_size = total_batch // tt

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

        log = self.obs_shim.initialize_to_environment(env, device)
        self.cnn_encoder.initialize_to_environment(env, device)
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
