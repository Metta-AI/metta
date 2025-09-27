import logging
import types
from typing import List

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import (
    ObsAttrEmbedFourierConfig,
)
from metta.agent.components.sliding_transformer import SlidingTransformerConfig
from metta.agent.policy import Policy, PolicyArchitecture
from softmax.training.rl.training import EnvironmentMetaData
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


def forward(self, td: TensorDict, action: torch.Tensor = None) -> TensorDict:
    """Forward pass for the FastDynamics policy."""
    self.network(td)
    self.action_probs(td, action)

    td["pred_input"] = torch.cat([td["core"], td["actor_query"]], dim=-1)
    self.returns_pred(td)
    self.reward_pred(td)
    td["values"] = td["values"].flatten()
    return td


class FastDynamicsConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _hidden_size = 32
    _embedding_dim = 16

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 64
    _token_embed_dim = 8
    _fourier_freqs = 3
    _embed_dim = 16
    _core_out_dim = 32
    _memory_num_layers = 2

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
        ObsAttrEmbedFourierConfig(
            in_key="obs_shim_tokens",
            out_key="obs_attr_embed",
            attr_embed_dim=_token_embed_dim,
            num_freqs=_fourier_freqs,
        ),
        ObsPerceiverLatentConfig(
            in_key="obs_attr_embed",
            out_key="encoded_obs",
            feat_dim=_token_embed_dim + (4 * _fourier_freqs) + 1,
            latent_dim=_latent_dim,
            num_latents=12,
            num_heads=4,
            num_layers=2,
        ),
        SlidingTransformerConfig(
            in_key="encoded_obs", out_key="core", output_dim=_core_out_dim, num_layers=_memory_num_layers
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[1024],
        ),
        ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=_embed_dim),
        ActorQueryConfig(in_key="core", out_key="actor_query", hidden_size=_core_out_dim, embed_dim=_embed_dim),
        ActorKeyConfig(
            query_key="actor_query",
            embedding_key="action_embedding",
            out_key="logits",
            embed_dim=_embed_dim,
        ),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, env_metadata: EnvironmentMetaData) -> Policy:
        AgentClass = load_symbol(self.class_path)
        policy = AgentClass(env_metadata, self)

        pred_input_dim = self._hidden_size + self._embedding_dim
        returns_module = nn.Linear(pred_input_dim, 1)
        reward_module = nn.Linear(pred_input_dim, 1)
        policy.returns_pred = TDM(returns_module, in_keys=["pred_input"], out_keys=["returns_pred"])
        policy.reward_pred = TDM(reward_module, in_keys=["pred_input"], out_keys=["reward_pred"])

        policy.forward = types.MethodType(forward, policy)

        return policy
