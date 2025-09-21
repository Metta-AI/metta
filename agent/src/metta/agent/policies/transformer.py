import logging
from typing import Dict, List, Optional

import numpy as np
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
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


class TransformerPolicyConfig(PolicyArchitecture):
    """Transformer-based policy using the refactored component stack."""

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

    # Transformer hyperparameters
    transformer_latent_size: int = 256
    transformer_hidden_size: int = 256
    transformer_num_layers: int = 6
    transformer_num_heads: int = 8
    transformer_ff_size: int = 512
    transformer_max_seq_len: int = 256
    transformer_memory_len: int = 64
    transformer_dropout: float = 0.1
    transformer_attn_dropout: float = 0.1

    # Actor / critic heads
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig(out_key="action_embedding")
    actor_query_config: ActorQueryConfig = ActorQueryConfig(in_key="actor_1", out_key="actor_query")
    actor_key_config: ActorKeyConfig = ActorKeyConfig(
        query_key="actor_query",
        embedding_key="action_embedding",
        out_key="logits",
    )
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class TransformerImprovedConfig(TransformerPolicyConfig):
    """Variant mirroring the richer Transformer-XL configuration."""

    transformer_ff_size: int = 1024
    transformer_memory_len: int = 96


class TransformerNvidiaConfig(TransformerPolicyConfig):
    """Variant inspired by NVIDIA's reference implementation settings."""

    transformer_ff_size: int = 1536
    transformer_dropout: float = 0.0
    transformer_attn_dropout: float = 0.0
    transformer_memory_len: int = 128


class TransformerPolicy(Policy):
    def __init__(self, env, config: Optional[TransformerPolicyConfig] = None):
        super().__init__()
        self.config = config or TransformerPolicyConfig()
        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        # Observation preprocessing stack
        self.obs_shim = ObsShimBox(env=env, config=self.config.obs_shim_config)

        # Ensure encoder output matches transformer latent expectations before instantiation
        encoder_out = self.config.cnn_encoder_config.encoded_obs_cfg.get("out_features")
        if encoder_out != self.config.transformer_latent_size:
            logger.info(
                "Adjusting CNN encoder output from %s to match transformer latent size %s.",
                encoder_out,
                self.config.transformer_latent_size,
            )
            self.config.cnn_encoder_config.encoded_obs_cfg["out_features"] = self.config.transformer_latent_size

        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, env=env)

        # Sequence model
        self.transformer_module = TransformerModule(
            d_model=self.config.transformer_hidden_size,
            n_heads=self.config.transformer_num_heads,
            n_layers=self.config.transformer_num_layers,
            d_ff=self.config.transformer_ff_size,
            max_seq_len=self.config.transformer_max_seq_len,
            memory_len=self.config.transformer_memory_len,
            dropout=self.config.transformer_dropout,
            dropatt=self.config.transformer_attn_dropout,
            pre_lnorm=True,
        )

        self.hidden_size = self.config.transformer_hidden_size
        self.latent_size = self.config.transformer_latent_size

        if self.latent_size != self.hidden_size:
            self.input_projection = pufferlib.pytorch.layer_init(
                nn.Linear(self.latent_size, self.hidden_size), std=1.0
            )
        else:
            self.input_projection = nn.Identity()

        # Critic branch
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.hidden_size, self.config.critic_hidden_dim), std=np.sqrt(2)
        )
        self.critic_1 = TDM(module, in_keys=["core"], out_keys=["critic_1"])
        self.critic_activation = nn.Tanh()
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.critic_hidden_dim, 1), std=1.0
        )
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor branch
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.hidden_size, self.config.actor_hidden_dim), std=1.0
        )
        self.actor_1 = TDM(module, in_keys=["core"], out_keys=["actor_1"])

        self.action_embeddings = ActionEmbedding(config=self.config.action_embedding_config)
        self.config.actor_query_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.config.actor_query_config.hidden_size = self.config.actor_hidden_dim
        self.actor_query = ActorQuery(config=self.config.actor_query_config)
        self.config.actor_key_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=self.config.actor_key_config)
        self.action_probs = ActionProbs(config=self.config.action_probs_config)

        self._memory: Dict[int, Optional[Dict[str, List[torch.Tensor]]]] = {}

    @torch._dynamo.disable
    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        self.obs_shim(td)
        self.cnn_encoder(td)

        core = self._forward_transformer(td)
        td["core"] = core

        self.actor_1(td)
        td["actor_1"] = torch.relu(td["actor_1"])

        self.critic_1(td)
        td["critic_1"] = self.critic_activation(td["critic_1"])
        self.value_head(td)

        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td)
        self.action_probs(td, action)

        td["values"] = td["values"].flatten()
        return td

    def _forward_transformer(self, td: TensorDict) -> torch.Tensor:
        latent = td[self.config.cnn_encoder_config.out_key]

        if "bptt" not in td.keys():
            raise KeyError("TensorDict is missing required 'bptt' metadata")

        TT = int(td["bptt"][0].item())
        if TT <= 0:
            raise ValueError("bptt entries must be positive")

        total_batch = latent.shape[0]
        if total_batch % TT != 0:
            raise ValueError("encoded_obs batch dimension must be divisible by bptt")

        if "batch" in td.keys():
            B = int(td["batch"][0].item())
        else:
            B = total_batch // TT

        latent = rearrange(latent, "(b t) h -> t b h", b=B, t=TT)
        latent = self.input_projection(latent)

        memory = None
        env_key: Optional[int] = None
        if TT == 1:
            env_key = self._get_env_start(td)
            memory = self._memory.get(env_key)

        core_out, new_memory = self.transformer_module(latent, memory)
        core_flat = rearrange(core_out, "t b h -> (b t) h")

        if TT == 1 and env_key is not None:
            updated_memory = self._detach_memory(new_memory)
            if updated_memory is not None:
                dones = td.get("dones", None)
                truncateds = td.get("truncateds", None)
                if dones is not None and truncateds is not None:
                    reset_mask = self._compute_reset_mask(dones, truncateds, B)
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
        elif TT > 1 and env_key is not None:
            # Do not carry sequence memory across training batches
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
            # Already shape [B]; treat as single timestep
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

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> List[str]:
        device = torch.device(device)
        self.to(device)

        log = self.obs_shim.initialize_to_environment(env, device)
        self.action_embeddings.initialize_to_environment(env, device)
        self.action_probs.initialize_to_environment(env, device)
        self._memory.clear()
        return [log]

    def reset_memory(self):
        self._memory.clear()

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
