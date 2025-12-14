from typing import List

from cortex.stacks import build_cortex_auto_config

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import Policy, PolicyArchitecture
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class ViTDefaultConfig(PolicyArchitecture):
    """Speed-optimized ViT variant with lighter token embeddings and attention stack.

    The trunk uses Axon blocks (post-up experts with residual connections) for efficient
    feature processing. Configure trunk depth, layer normalization, and hidden dimension
    scaling independently.
    """

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    _token_embed_dim = 8
    _fourier_freqs = 3
    _latent_dim = 64
    _actor_hidden = 256

    # Whether training passes cached pre-state to the Cortex core
    pass_state_during_training: bool = False
    _critic_hidden = 512

    # Trunk configuration
    # Number of Axon layers in the trunk
    core_resnet_layers: int = 2
    # Pattern for trunk layers (e.g., "A" for Axon blocks, "L" for linear)
    core_resnet_pattern: str = "A"
    # Enable layer normalization after each trunk layer
    core_use_layer_norm: bool = True
    # Whether to torch.compile the trunk (Cortex stack)
    core_compile: bool = False

    components: List[ComponentConfig] = []

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, policy_env_info: PolicyEnvInterface) -> Policy:
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
                    num_layers=self.core_resnet_layers,
                    pattern=self.core_resnet_pattern,
                    post_norm=self.core_use_layer_norm,
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
                out_features=1,
                hidden_features=[self._critic_hidden],
            ),
            ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=self._actor_hidden),
        ]

        return super().make_policy(policy_env_info)
