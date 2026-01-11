from typing import List, Optional

from cortex.config import CortexStackConfig
from cortex.stacks import build_cortex_auto_config
from pydantic import ConfigDict

from metta.agent.components.actor import ActionProbsConfig, ActorHeadConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.cortex import CortexTDConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture


class CortexBaseConfig(PolicyArchitecture):
    """ViT-style policy with Cortex stack (xLSTM) replacing LSTM core."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    # Default geometry; stack_cfg may override d_hidden
    _token_embed_dim: int = 8
    _fourier_freqs: int = 3
    _latent_dim: int = 64
    _core_out: int = 32  # align with ViTReset LSTM latent
    _actor_hidden: int = 256
    _critic_hidden: int = 512

    # Storage dtype for CortexTDConfig ("float32", "float16", "bfloat16", etc.).
    dtype: str = "float32"

    # Optional explicit Cortex stack configuration. If not provided, a default
    # AXMS-based stack is built in model_post_init.
    stack_cfg: Optional[CortexStackConfig] = None

    components: List[ComponentConfig] = []
    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def model_post_init(self, __context: object) -> None:  # type: ignore[override]
        self.components = self.build_components()

    def build_components(self) -> List[ComponentConfig]:
        feat_dim = self._token_embed_dim + (4 * self._fourier_freqs) + 1

        stack_cfg = self.stack_cfg
        if stack_cfg is None:
            stack_cfg = build_cortex_auto_config(d_hidden=self._latent_dim, post_norm=True)
        stack_hidden = int(stack_cfg.d_hidden)

        components: List[ComponentConfig] = [
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
                feat_dim=feat_dim,
                latent_dim=stack_hidden,
                num_latents=12,
                num_heads=4,
                num_layers=2,
            ),
            CortexTDConfig(
                in_key="obs_latent_attn",
                out_key="core",
                d_hidden=stack_hidden,
                out_features=self._core_out,
                # Default to the mixed Cortex auto stack (Axon/mLSTM/sLSTM) via config.
                stack_cfg=stack_cfg,
                key_prefix="cortex_state",
                dtype=self.dtype,
            ),
            MLPConfig(
                in_key="core",
                out_key="actor_hidden",
                name="actor_mlp",
                in_features=self._core_out,
                hidden_features=[self._actor_hidden],
                out_features=self._actor_hidden,
            ),
            MLPConfig(
                in_key="core",
                out_key="values",
                name="critic",
                in_features=self._core_out,
                out_features=1,
                hidden_features=[self._critic_hidden],
            ),
            MLPConfig(
                in_key="core",
                out_key="h_values",
                name="gtd_aux",
                in_features=self._core_out,
                out_features=1,
                hidden_features=[self._critic_hidden],
            ),
            ActorHeadConfig(in_key="actor_hidden", out_key="logits", input_dim=self._actor_hidden),
        ]

        return components
