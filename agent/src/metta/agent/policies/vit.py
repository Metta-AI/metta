from typing import List

from pydantic import Field

from metta.agent.components.action import ActionEmbeddingConfig
from metta.agent.components.actor import ActionProbsConfig, ActorKeyConfig, ActorQueryConfig
from metta.agent.components.component_config import ComponentConfig
from metta.agent.components.lstm import LSTMConfig
from metta.agent.components.misc import MLPConfig
from metta.agent.components.obs_enc import ObsPerceiverLatentConfig
from metta.agent.components.obs_shim import ObsShimTokensConfig
from metta.agent.components.obs_tokenizers import ObsAttrEmbedFourierConfig
from metta.agent.policy import PolicyArchitecture
from metta.rl.training import EnvironmentMetaData


class ViTDefaultConfig(PolicyArchitecture):
    """Speed-optimized ViT variant with lighter token embeddings and attention stack."""

    class_path: str = "metta.agent.policy_auto_builder.PolicyAutoBuilder"

    embedding_dim: int = Field(default=16, gt=0, description="Action embedding width")
    token_embed_dim: int = Field(default=8, gt=0, description="Observation token embedding width")
    fourier_freqs: int = Field(default=3, ge=1, le=8, description="Number of Fourier frequency pairs")
    latent_dim: int = Field(default=64, gt=0, description="Perceiver latent dimensionality")
    lstm_latent: int = Field(default=32, gt=0, description="LSTM hidden size")
    actor_hidden: int = Field(default=256, gt=0, description="Actor MLP hidden features")
    critic_hidden: int = Field(default=512, gt=0, description="Critic MLP hidden features")

    components: List[ComponentConfig] = Field(default_factory=list)

    action_probs_config: ActionProbsConfig = ActionProbsConfig(in_key="logits")

    def _make_components(self) -> List[ComponentConfig]:
        feat_dim = self.token_embed_dim + (4 * self.fourier_freqs) + 1

        return [
            ObsShimTokensConfig(in_key="env_obs", out_key="obs_shim_tokens", max_tokens=48),
            ObsAttrEmbedFourierConfig(
                in_key="obs_shim_tokens",
                out_key="obs_attr_embed",
                attr_embed_dim=self.token_embed_dim,
                num_freqs=self.fourier_freqs,
            ),
            ObsPerceiverLatentConfig(
                in_key="obs_attr_embed",
                out_key="obs_latent_attn",
                feat_dim=feat_dim,
                latent_dim=self.latent_dim,
                num_latents=12,
                num_heads=4,
                num_layers=2,
            ),
            LSTMConfig(
                in_key="obs_latent_attn",
                out_key="core",
                latent_size=self.latent_dim,
                hidden_size=self.lstm_latent,
                num_layers=1,
            ),
            MLPConfig(
                in_key="core",
                out_key="actor_hidden",
                name="actor_mlp",
                in_features=self.lstm_latent,
                hidden_features=[self.actor_hidden],
                out_features=self.actor_hidden,
            ),
            MLPConfig(
                in_key="core",
                out_key="values",
                name="critic",
                in_features=self.lstm_latent,
                out_features=1,
                hidden_features=[self.critic_hidden],
            ),
            ActionEmbeddingConfig(out_key="action_embedding", embedding_dim=self.embedding_dim),
            ActorQueryConfig(
                in_key="actor_hidden",
                out_key="actor_query",
                hidden_size=self.actor_hidden,
                embed_dim=self.embedding_dim,
            ),
            ActorKeyConfig(
                query_key="actor_query",
                embedding_key="action_embedding",
                out_key="logits",
                embed_dim=self.embedding_dim,
            ),
        ]

    def build_components(self, env_metadata: EnvironmentMetaData | None = None) -> List[ComponentConfig]:
        return self._make_components()
