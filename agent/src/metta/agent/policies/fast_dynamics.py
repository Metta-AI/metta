import logging
import types
from typing import List

import torch
from cortex.stacks import build_cortex_auto_config
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as TDM
from torch import nn

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import (
    ObsAttrEmbedFourierConfig,
)
from metta.agent.policy import Policy
from metta.agent.policy_architecture import PolicyArchitecture
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)


def forward(self, td: TensorDict, action: torch.Tensor = None) -> TensorDict:
    """Forward pass for the FastDynamics policy."""
    self.network(td)
    self.action_probs(td, action)

    td["pred_input"] = torch.cat([td["core"], td["logits"]], dim=-1)
    self.returns_pred(td)
    self.reward_pred(td)
    self.future_latent_pred(td)
    td["values"] = td["values"].flatten()
    if "h_values" in td.keys():
        td["h_values"] = td["h_values"].flatten()
    return td


class FastDynamicsConfig(PolicyArchitecture):
    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _latent_dim = 64
    _token_embed_dim = 8
    _fourier_freqs = 3
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
        CortexTDConfig(
            in_key="encoded_obs",
            out_key="core",
            d_hidden=_latent_dim,
            out_features=_core_out_dim,
            key_prefix="fastdyn_cortex_state",
            stack_cfg=build_cortex_auto_config(
                d_hidden=_latent_dim,
                num_layers=_memory_num_layers,
                pattern="X",
                post_norm=True,
            ),
        ),
        MLPConfig(
            in_key="core",
            out_key="values",
            name="critic",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[1024],
        ),
        MLPConfig(
            in_key="core",
            out_key="h_values",
            name="gtd_aux",
            in_features=_core_out_dim,
            out_features=1,
            hidden_features=[1024],
        ),
        ActorHeadConfig(in_key="core", out_key="logits", input_dim=_core_out_dim),
    ]

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, policy_env_info: PolicyEnvInterface) -> Policy:
        AgentClass = load_symbol(self.class_path)
        policy = AgentClass(policy_env_info, self)

        num_actions = policy_env_info.action_space.n
        pred_input_dim = self._core_out_dim + num_actions
        returns_module = nn.Linear(pred_input_dim, 1)
        reward_module = nn.Linear(pred_input_dim, 1)
        future_latent_module = nn.Linear(pred_input_dim, self._core_out_dim)
        policy.returns_pred = TDM(returns_module, in_keys=["pred_input"], out_keys=["returns_pred"])
        policy.reward_pred = TDM(reward_module, in_keys=["pred_input"], out_keys=["reward_pred"])
        policy.future_latent_pred = TDM(
            future_latent_module,
            in_keys=["pred_input"],
            out_keys=["future_latent_pred"],
        )

        policy.forward = types.MethodType(forward, policy)

        return policy
