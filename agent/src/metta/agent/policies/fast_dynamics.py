import logging
from typing import List

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm_reset import LSTMResetConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsLatentAttnConfig, ObsSelfAttnConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import (
    ObsAttrEmbedFourierConfig,
)
from metta.agent.policy import Policy, PolicyArchitecture
from metta.rl.training.training_environment import EnvironmentMetaData
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


def forward(self, td: TensorDict, action: torch.Tensor = None) -> TensorDict:
    """Forward pass for the FastDynamics policy."""
    self.network(td)
    self.action_probs(td, action)

    td["pred_input"] = torch.cat([td["core"], td["actor_query"]], dim=-1)
    td["returns_pred"] = self.layers["returns_pred"](td)
    td["reward_pred"] = self.layers["reward_pred"](td)
    td["values"] = td["values"].flatten()
    return td


class FastDynamicsConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    lstm_cfg: LSTMResetConfig = LSTMResetConfig(
        in_key="obs_self_attn",
        out_key="core",
        latent_size=128,
        hidden_size=128,
        num_layers=2,
    )
    action_embedding_cfg: ActionEmbeddingConfig = ActionEmbeddingConfig(out_key="action_embedding")
    actor_query_cfg: ActorQueryConfig = ActorQueryConfig(
        in_key="core",
        out_key="actor_query",
        hidden_size=lstm_cfg.hidden_size,
        embed_dim=action_embedding_cfg.embedding_dim,
    )

    components: List[ComponentConfig] = [
        ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens"),
        ObsAttrEmbedFourierConfig(in_key="obs_shim_tokens", out_key="obs_attr_embed_fourier"),
        ObsLatentAttnConfig(in_key="obs_attr_embed_fourier", out_key="obs_latent_attn", feat_dim=37, out_dim=48),
        ObsSelfAttnConfig(in_key="obs_latent_attn", out_key="obs_self_attn", feat_dim=48, out_dim=128),
        lstm_cfg,
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=lstm_cfg.hidden_size,
            out_features=1,
            hidden_features=[1024],
        ),
        action_embedding_cfg,
        actor_query_cfg,
        ActorKeyConfig(query_key="actor_query", embedding_key="action_embedding", out_key="logits"),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, env_metadata: EnvironmentMetaData) -> Policy:
        AgentClass = load_symbol(self.class_path)
        policy = AgentClass(env_metadata, self)

        pred_input_dim = self.lstm_cfg.hidden_size + self.actor_query_cfg.embed_dim
        returns_module = nn.Linear(pred_input_dim, 1)
        reward_module = nn.Linear(pred_input_dim, 1)
        policy.layers["returns_pred"] = TDM(returns_module, in_keys=["pred_input"], out_keys=["returns_pred"])
        policy.layers["reward_pred"] = TDM(reward_module, in_keys=["pred_input"], out_keys=["reward_pred"])

        policy.forward = forward

        return policy
