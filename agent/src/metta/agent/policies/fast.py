import logging
from typing import Dict, List, Optional

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
from metta.agent.util.weights_analysis import analyze_weights

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
    clip_range: float = 0.0
    clip_scale: float = 1.0
    l2_init_scale: float = 1.0
    analyze_weights_interval: int = 300


class FastPolicy(Policy):
    def __init__(self, env, config: Optional[FastConfig] = None):
        super().__init__()
        self.config = config or FastConfig()
        self.env = env
        self.is_continuous = False
        self.action_space = env.action_space

        self.active_action_names = []
        self.num_active_actions = 100  # Default
        self.action_index_tensor = None
        self.cum_action_max_params = None

        self.out_width = env.obs_width
        self.out_height = env.obs_height

        self.obs_shim = ObsShimBox(env=env, config=self.config.obs_shim_config)

        self.cnn_encoder = CNNEncoder(config=self.config.cnn_encoder_config, env=env)

        self.lstm = LSTM(config=self.config.lstm_config)

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

        # Regularisation/monitoring state (mirrors PyTorchAgentMixin behaviour)
        self.clip_range = self.config.clip_range
        self.clip_scale = self.config.clip_scale
        self.l2_init_scale = self.config.l2_init_scale
        self.analyze_weights_interval = self.config.analyze_weights_interval
        self._initial_weights: Dict[str, torch.Tensor] = {}
        self._store_initial_weights()

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
        td["values"] = td["values"].flatten()

        if needs_unflatten and action is not None:
            td = td.reshape(batch_size, time_steps)

        return td

    def initialize_to_environment(
        self,
        env,
        device,
    ) -> List[str]:
        log = self.obs_shim.initialize_to_environment(env, device)
        self.action_embeddings.initialize_to_environment(env, device)
        self.action_probs.initialize_to_environment(env, device)
        self.action_index_tensor = self.action_probs.action_index_tensor
        self.cum_action_max_params = self.action_probs.cum_action_max_params
        return [log]

    def reset_memory(self):
        self.lstm.reset_memory()

    def get_agent_experience_spec(self) -> Composite:
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Weight regularisation and monitoring helpers (parity with PyTorch mixin)
    # ------------------------------------------------------------------
    def clip_weights(self) -> None:
        if self.clip_range <= 0:
            return

        clip_value = self.clip_range * self.clip_scale
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                module.weight.data.clamp_(-clip_value, clip_value)
                if module.bias is not None:
                    module.bias.data.clamp_(-clip_value, clip_value)

    def l2_init_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, dtype=torch.float32)
        if not self._initial_weights:
            return total_loss

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)) and hasattr(module, "weight"):
                key = name if name else "root"
                if key in self._initial_weights:
                    weight_diff = module.weight - self._initial_weights[key].to(module.weight.device)
                    total_loss = total_loss.to(module.weight.device)
                    total_loss += torch.sum(weight_diff**2) * self.l2_init_scale
        return total_loss

    def update_l2_init_weight_copy(self) -> None:
        self._store_initial_weights()

    def compute_weight_metrics(self, delta: float = 0.01) -> List[dict]:
        metrics: List[dict] = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and module.weight.data.dim() == 2:
                result = analyze_weights(module.weight.data, delta)
                result["name"] = name if name else "root"
                metrics.append(result)
        return metrics

    def _store_initial_weights(self) -> None:
        self._initial_weights.clear()
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)) and hasattr(module, "weight"):
                key = name if name else "root"
                self._initial_weights[key] = module.weight.data.clone()

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
