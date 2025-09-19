import logging
from collections.abc import Sequence
from typing import List, Optional

import numpy as np
import pufferlib.pytorch
import torch
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
from metta.agent.components.lstm import LSTM, LSTMConfig
from metta.agent.components.obs_shim import ObsShimBox, ObsShimBoxConfig
from metta.agent.policy import Policy, PolicyArchitecture

logger = logging.getLogger(__name__)


class FastConfig(PolicyArchitecture):
    """
    Fast uses a CNN encoder so is not flexible to changing observation features but it runs faster than the ViT encoder.

    This particular class is also set up without using PolicyAutoBuilder to demonstrate an alternative way to build a
    policy, affording more control over the process at the expense of more code. It demonstrates that we can use config
    objects (ie LSTMConfig) for classes as layers (self.lstm) or attributes (ie actor_hidden_dim) for simple torch
    classes as layers (ie self.critic_1), wrap them in a TensorDictModule, and intermix."""

    class_path: str = "metta.agent.policies.fast.FastPolicy"

    obs_shim_config: ObsShimBoxConfig = ObsShimBoxConfig(in_key="env_obs", out_key="obs_normalizer")
    cnn_encoder_config: CNNEncoderConfig = CNNEncoderConfig(in_key="obs_normalizer", out_key="encoded_obs")
    lstm_config: LSTMConfig = LSTMConfig(
        in_key="encoded_obs", out_key="core", latent_size=128, hidden_size=128, num_layers=2
    )
    critic_hidden_dim: int = 1024
    actor_hidden_dim: int = 512
    action_embedding_config: ActionEmbeddingConfig = ActionEmbeddingConfig(out_key="action_embedding")
    actor_query_config: ActorQueryConfig = ActorQueryConfig(in_key="actor_1", out_key="actor_query")
    actor_key_config: ActorKeyConfig = ActorKeyConfig(
        query_key="actor_query", embedding_key="action_embedding", out_key="logits"
    )
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")


class FastPolicy(Policy):
    def __init__(self, env, config: Optional[FastConfig] = None):
        super().__init__()
        self.config = config or FastConfig()
        self.env = env
        self.env_metadata = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        self.obs_shim = ObsShimBox(env=env, config=self.config.obs_shim_config)

        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, env=env)

        self.lstm = LSTM(config=self.config.lstm_config)

        self._components_with_metrics: list[nn.Module] = []
        self.action_index_tensor: torch.Tensor | None = None
        self.cum_action_max_params: torch.Tensor | None = None

        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.lstm_config.hidden_size, self.config.actor_hidden_dim), std=1.0
        )
        self.actor_1 = TDM(module, in_keys=["core"], out_keys=["actor_1"])

        # Critic branch
        # critic_1 uses gain=sqrt(2) because it's followed by tanh (YAML: nonlinearity: nn.Tanh)
        module = pufferlib.pytorch.layer_init(
            nn.Linear(self.config.lstm_config.hidden_size, self.config.critic_hidden_dim), std=np.sqrt(2)
        )
        self.critic_1 = TDM(module, in_keys=["core"], out_keys=["critic_1"])
        self.critic_activation = nn.Tanh()
        module = pufferlib.pytorch.layer_init(nn.Linear(self.config.critic_hidden_dim, 1), std=1.0)
        self.value_head = TDM(module, in_keys=["critic_1"], out_keys=["values"])

        # Actor branch
        self.action_embeddings = ActionEmbedding(config=self.config.action_embedding_config)
        self.config.actor_query_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.config.actor_query_config.hidden_size = self.config.actor_hidden_dim
        self.actor_query = ActorQuery(config=self.config.actor_query_config)
        self.config.actor_key_config.embed_dim = self.config.action_embedding_config.embedding_dim
        self.actor_key = ActorKey(config=self.config.actor_key_config)
        self.action_probs = ActionProbs(config=self.config.action_probs_config)
        self._components_with_metrics = [
            getattr(self.cnn_encoder, "cnn1", None),
            getattr(self.cnn_encoder, "cnn2", None),
        ]

    def forward(self, td: TensorDict, state=None, action: torch.Tensor = None):
        needs_unflatten = td.batch_dims > 1
        if needs_unflatten:
            batch_size, time_steps = td.batch_size
            td = td.reshape(td.batch_size.numel())
            td.set(
                "bptt",
                torch.full((batch_size * time_steps,), time_steps, device=td.device, dtype=torch.long),
            )
            td.set(
                "batch",
                torch.full((batch_size * time_steps,), batch_size, device=td.device, dtype=torch.long),
            )
        else:
            batch_elems = td.batch_size.numel()
            td.set("bptt", torch.ones((batch_elems,), device=td.device, dtype=torch.long))
            td.set("batch", torch.full((batch_elems,), batch_elems, device=td.device, dtype=torch.long))

        self.obs_shim(td)
        self.cnn_encoder(td)
        self.lstm(td)
        self.actor_1(td)
        td["actor_1"] = torch.relu(td["actor_1"])
        self.critic_1(td)
        td["critic_1"] = self.critic_activation(td["critic_1"])
        self.value_head(td)
        self.action_embeddings(td)
        self.actor_query(td)
        self.actor_key(td)
        self.action_probs(td, action)
        value_tensor = td["values"]
        td["value"] = value_tensor
        td["values"] = value_tensor.flatten()

        if needs_unflatten and action is not None:
            td = td.reshape(batch_size, time_steps)

        return td

    def initialize_to_environment(
        self,
        env_or_full_action_names,
        device,
    ) -> List[str]:
        if isinstance(env_or_full_action_names, Sequence) and not isinstance(env_or_full_action_names, str):
            full_action_names = list(env_or_full_action_names)
            env_metadata = self.env_metadata
        else:
            env_metadata = env_or_full_action_names
            full_action_names = [
                f"{name}_{i}"
                for name, max_param in zip(env_metadata.action_names, env_metadata.max_action_args, strict=False)
                for i in range(max_param + 1)
            ]
            self.env_metadata = env_metadata

        log = self.obs_shim.initialize_to_environment(env_metadata, device)
        self.action_embeddings.initialize_to_environment(full_action_names, device)
        self.action_probs.initialize_to_environment(env_metadata.max_action_args, device)
        self.action_index_tensor = self.action_probs.action_index_tensor
        self.cum_action_max_params = self.action_probs.cum_action_max_params
        return [log]

    def reset_memory(self):
        self.lstm.reset_memory()

    def clip_weights(self) -> None:
        for module in self._iter_component_modules():
            if hasattr(module, "clip_weights"):
                module.clip_weights()

    def l2_init_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for module in self._iter_component_modules():
            if hasattr(module, "l2_init_loss"):
                total_loss = total_loss + module.l2_init_loss()
        return total_loss

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        metrics: List[dict] = []
        for module in self._components_with_metrics:
            if module is not None and hasattr(module, "compute_weight_metrics"):
                result = module.compute_weight_metrics(delta)
                if result:
                    metrics.append(result)
        return metrics

    def _iter_component_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = [
            self.cnn_encoder,
            self.lstm,
            self.actor_query,
            self.actor_key,
            self.action_probs,
            self.action_embeddings,
        ]

        # Include TensorDictModule-wrapped modules if available
        td_modules = [
            getattr(self.actor_1, "module", None),
            getattr(self.critic_1, "module", None),
            getattr(self.value_head, "module", None),
        ]
        modules.extend(td_modules)

        return [module for module in modules if module is not None]

    def _convert_action_to_logit_index(self, flattened_action: torch.Tensor) -> torch.Tensor:
        if self.cum_action_max_params is None:
            raise RuntimeError("FastPolicy has not been initialized with action metadata yet")
        if flattened_action.size(0) == 0:
            raise ValueError("flattened_action must have non-zero batch dimension")
        action_type_numbers = flattened_action[:, 0].long()
        action_params = flattened_action[:, 1].long()
        cumulative_sum = self.cum_action_max_params[action_type_numbers]
        return action_type_numbers + cumulative_sum + action_params

    def _convert_logit_index_to_action(self, action_logit_index: torch.Tensor) -> torch.Tensor:
        if self.action_index_tensor is None:
            raise RuntimeError("FastPolicy has not been initialized with action metadata yet")
        return self.action_index_tensor[action_logit_index]

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
