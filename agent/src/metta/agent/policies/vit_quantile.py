from typing import List, Optional

import torch
from cortex.stacks import build_cortex_auto_config
from tensordict import TensorDict

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy_architecture import PolicyArchitecture
from metta.agent.policy_auto_builder import PolicyAutoBuilder


class QuantilePolicyAutoBuilder(PolicyAutoBuilder):
    """Policy builder for quantile critic."""

    def forward(self, td: TensorDict, action: Optional[torch.Tensor] = None) -> TensorDict:
        td = self._sequential_network(td)
        self.action_probs(td, action)
        # Do NOT flatten values for quantile critic
        # if "values" in td.keys():
        #    td["values"] = td["values"].flatten()
        return td

    @property
    def critic_quantiles(self) -> int:
        return self.config.critic_quantiles


class ViTQuantileConfig(PolicyArchitecture):
    """Speed-optimized ViT variant with lighter token embeddings and attention stack, using Quantile Critic."""

    class_path: str = "metta.agent.policies.vit_quantile.QuantilePolicyAutoBuilder"

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _actor_hidden = 256

    # Whether training passes cached pre-state to the Cortex core
    pass_state_during_training: bool = False
    # Whether to torch.compile the trunk (Cortex stack)
    core_compile: bool = False
    _critic_hidden = 512

    critic_quantiles: int = 25

    # Define components as a method or property or just default, but we need to update it based on critic_quantiles
    components: List[ComponentConfig] = []

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def __init__(self, **data):
        super().__init__(**data)

        # If components not passed (using default empty list or whatever), build them
        # Actually, we should define the default list here to ensure it uses instance values

        if not self.components:
            self.components = [
                ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
                ObsAttrEmbedFourierConfig(
                    in_key="obs_shim_tokens",
                    out_key="obs_attr_embed",
                    attr_embed_dim=self._token_embed_dim,
                    num_freqs=self._fourier_freqs,
                ),
                ObsPerceiverLatentConfig(
                    in_key="obs_attr_embed",
                    out_key="obs_latent_attn",
                    feat_dim=self._token_embed_dim + (4 * self._fourier_freqs) + 1,
                    latent_dim=self._latent_dim,
                    num_latents=12,
                    num_heads=4,
                    num_layers=2,
                ),
                CortexTDConfig(
                    in_key="obs_latent_attn",
                    out_key="core",
                    d_hidden=self._latent_dim,
                    out_features=self._latent_dim,
                    key_prefix="vit_cortex_state",
                    stack_cfg=build_cortex_auto_config(
                        d_hidden=self._latent_dim,
                        num_layers=1,
                        pattern="L",
                        post_norm=False,
                        compile_blocks=self.core_compile,
                    ),
                    pass_state_during_training=self.pass_state_during_training,
                ),
                MLPConfig(
                    in_key="core",
                    out_key="actor_hidden",
                    name="actor_mlp",
                    in_features=self._latent_dim,
                    hidden_features=[self._actor_hidden],
                    out_features=self._actor_hidden,
                ),
                MLPConfig(
                    in_key="core",
                    out_key="values",
                    name="critic",
                    in_features=self._latent_dim,
                    out_features=self.critic_quantiles,  # Output N quantiles
                    hidden_features=[self._critic_hidden],
                ),
                ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=self._actor_hidden),
            ]
