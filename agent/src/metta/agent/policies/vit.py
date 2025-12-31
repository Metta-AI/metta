from typing import List

from cortex.stacks import build_cortex_auto_config
from pydantic import ConfigDict, Field

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
    model_config = ConfigDict(populate_by_name=True)

    _token_embed_dim = 8
    _fourier_freqs = 3
    # Defaults aligned with legacy CvC baseline (keep max_tokens from newer runs)
    latent_dim: int = Field(default=64)
    actor_hidden: int = Field(default=256)
    core_num_heads: int = Field(default=4)
    max_tokens: int = Field(default=128)
    core_num_latents: int = Field(default=12)

    # Whether training passes cached pre-state to the Cortex core
    pass_state_during_training: bool = False
    critic_hidden: int = Field(default=512)

    # Trunk configuration
    # Number of Axon layers in the trunk
    core_resnet_layers: int = 2
    # Pattern for trunk layers (e.g., "A" for Axon blocks, "L" for linear)
    core_resnet_pattern: str = "A"
    # Enable layer normalization after each trunk layer
    core_use_layer_norm: bool = False
    # Whether to torch.compile the trunk (Cortex stack)
    core_compile: bool = False

    components: List[ComponentConfig] = []

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def make_policy(self, policy_env_info: PolicyEnvInterface) -> Policy:
        # If the architecture spec already bundled a component list (common for saved
        # .mpt checkpoints), reuse it instead of regenerating with current defaults.
        # This keeps restored policies aligned with the shapes they were trained with.
        if self.components:
            return super().make_policy(policy_env_info)

        self.components = [
            ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=self.max_tokens),
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
                latent_dim=self.latent_dim,
                num_latents=self.core_num_latents,
                num_heads=self.core_num_heads,
                num_layers=2,
            ),
            CortexTDConfig(
                in_key="obs_latent_attn",
                out_key="core",
                d_hidden=self.latent_dim,
                out_features=self.latent_dim,
                key_prefix="vit_cortex_state",
                stack_cfg=build_cortex_auto_config(
                    d_hidden=self.latent_dim,
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
                in_features=self.latent_dim,
                hidden_features=[self.actor_hidden],
                out_features=self.actor_hidden,
            ),
            MLPConfig(
                in_key="core",
                out_key="values",
                name="critic",
                in_features=self.latent_dim,
                out_features=1,
                hidden_features=[self.critic_hidden],
            ),
            MLPConfig(
                in_key="core",
                out_key="h_values",
                name="gtd_aux",
                in_features=self.latent_dim,
                out_features=1,
                hidden_features=[self.critic_hidden],
            ),
            ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=self.actor_hidden),
        ]

        return super().make_policy(policy_env_info)
