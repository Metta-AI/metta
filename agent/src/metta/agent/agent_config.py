from typing import Any, ClassVar, Literal, Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, Field, model_validator

from metta.common.util.config import Config


class ObservationConfig(Config):
    """Configuration for observation processing."""

    obs_key: str = Field(default="grid_obs", description="Key for observation data")

    # Observation encoder type
    encoder_type: Literal["cnn", "attention", "latent_attention"] = Field(
        default="cnn", description="Type of observation encoder"
    )

    # CNN-specific parameters
    cnn_channels: list[int] = Field(
        default=[64, 64], description="Number of channels for each CNN layer"
    )
    cnn_kernel_sizes: list[int] = Field(
        default=[5, 3], description="Kernel sizes for CNN layers"
    )
    cnn_strides: list[int] = Field(
        default=[3, 1], description="Strides for CNN layers"
    )

    # Attention-specific parameters
    attention_dim: int = Field(
        default=128, description="Dimension for attention layers"
    )
    attention_heads: int = Field(
        default=4, description="Number of attention heads"
    )
    attention_layers: int = Field(
        default=2, description="Number of attention layers"
    )

    # Latent attention parameters
    latent_query_tokens: int = Field(
        default=10, description="Number of latent query tokens"
    )
    latent_query_dim: int = Field(
        default=32, description="Dimension of latent query tokens"
    )

    # Fourier feature parameters
    fourier_freqs: int = Field(
        default=8, description="Number of Fourier frequencies"
    )
    attr_embed_dim: int = Field(
        default=8, description="Attribute embedding dimension"
    )


class CoreConfig(Config):
    """Configuration for the core network (LSTM)."""

    hidden_size: int = Field(
        default=128, description="LSTM hidden size"
    )
    num_layers: int = Field(
        default=2, description="Number of LSTM layers"
    )


class CriticConfig(Config):
    """Configuration for the critic network."""

    hidden_size: int = Field(
        default=1024, description="Critic hidden layer size"
    )
    nonlinearity: Literal["nn.Tanh", "nn.ReLU", "nn.GELU", None] = Field(
        default="nn.Tanh", description="Nonlinearity for critic"
    )
    effective_rank: bool = Field(
        default=True, description="Use effective rank regularization"
    )


class ActorConfig(Config):
    """Configuration for the actor network."""

    hidden_size: int = Field(
        default=512, description="Actor hidden layer size"
    )


class ActionEmbeddingConfig(Config):
    """Configuration for action embeddings."""

    num_embeddings: int = Field(
        default=100, description="Number of action embeddings"
    )
    embedding_dim: int = Field(
        default=16, description="Action embedding dimension"
    )


class AgentConfig(Config):
    """Compact agent configuration that abstracts away detailed component specifications."""

    # Core configuration
    observations: ObservationConfig = Field(default_factory=ObservationConfig)
    core: CoreConfig = Field(default_factory=CoreConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    actor: ActorConfig = Field(default_factory=ActorConfig)
    action_embeddings: ActionEmbeddingConfig = Field(default_factory=ActionEmbeddingConfig)

    # Training parameters
    clip_range: float = Field(
        default=0.0, description="Weight clipping range (0 to disable)"
    )
    analyze_weights_interval: int = Field(
        default=300, description="Interval for weight analysis"
    )

    # Policy selector (for loading pre-trained policies)
    policy_selector: Optional[dict[str, Any]] = Field(
        default=None, description="Policy selector configuration"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "AgentConfig":
        """Validate configuration fields."""
        # Validate CNN parameters have matching lengths
        if self.observations.encoder_type == "cnn":
            cnn_params = [
                self.observations.cnn_channels,
                self.observations.cnn_kernel_sizes,
                self.observations.cnn_strides
            ]
            if not all(len(param) == len(cnn_params[0]) for param in cnn_params):
                raise ValueError("CNN parameters must have matching lengths")

        return self

    def to_component_config(self) -> dict[str, Any]:
        """Convert compact config to detailed component configuration."""
        components = {}

        # Observation components
        if self.observations.encoder_type == "cnn":
            components.update(self._build_cnn_components())
        elif self.observations.encoder_type == "attention":
            components.update(self._build_attention_components())
        elif self.observations.encoder_type == "latent_attention":
            components.update(self._build_latent_attention_components())

        # Core LSTM
        components["_core_"] = {
            "_target_": "metta.agent.lib.lstm.LSTM",
            "sources": [{"name": "encoded_obs"}],
            "nn_params": {
                "hidden_size": self.core.hidden_size,
                "num_layers": self.core.num_layers
            }
        }

        # Critic network
        components["critic_1"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "_core_"}],
            "nn_params": {
                "out_features": self.critic.hidden_size
            },
            "nonlinearity": self.critic.nonlinearity,
            "effective_rank": self.critic.effective_rank
        }

        components["_value_"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "critic_1"}],
            "nn_params": {
                "out_features": 1
            },
            "nonlinearity": None
        }

        # Actor network
        components["actor_1"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "_core_"}],
            "nn_params": {
                "out_features": self.actor.hidden_size
            }
        }

        # Action embeddings
        components["_action_embeds_"] = {
            "_target_": "metta.agent.lib.action.ActionEmbedding",
            "sources": None,
            "nn_params": {
                "num_embeddings": self.action_embeddings.num_embeddings,
                "embedding_dim": self.action_embeddings.embedding_dim
            }
        }

        # Action head
        components["_action_"] = {
            "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
            "sources": [
                {"name": "actor_1"},
                {"name": "_action_embeds_"}
            ]
        }

        return components

    def _build_cnn_components(self) -> dict[str, Any]:
        """Build CNN-based observation components."""
        components = {}

        # Observation token to box shaper
        components["_obs_"] = {
            "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
            "sources": None
        }

        # Observation normalizer
        components["obs_normalizer"] = {
            "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
            "sources": [{"name": "_obs_"}]
        }

        # CNN layers
        prev_layer = "obs_normalizer"
        for i, (channels, kernel_size, stride) in enumerate(zip(
            self.observations.cnn_channels,
            self.observations.cnn_kernel_sizes,
            self.observations.cnn_strides
        )):
            layer_name = f"cnn{i+1}"
            components[layer_name] = {
                "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                "sources": [{"name": prev_layer}],
                "nn_params": {
                    "out_channels": channels,
                    "kernel_size": kernel_size,
                    "stride": stride
                }
            }
            prev_layer = layer_name

        # Flatten and final encoding
        components["obs_flattener"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Flatten",
            "sources": [{"name": prev_layer}]
        }


        components["fc1"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "obs_flattener"}],
            "nn_params": {
                "out_features": self.core.hidden_size
            }
        }

        components["encoded_obs"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "fc1"}],
            "nn_params": {
                "out_features": self.core.hidden_size
            }
        }

        return components

    def _build_attention_components(self) -> dict[str, Any]:
        """Build attention-based observation components."""
        components = {}

        # Observation tokenizer
        components["_obs_"] = {
            "_target_": "metta.agent.lib.obs_tokenizers.ObsTokenPadStrip",
            "sources": None
        }

        # Normalizer
        components["obs_normalizer"] = {
            "_target_": "metta.agent.lib.obs_tokenizers.ObsAttrValNorm",
            "sources": [{"name": "_obs_"}]
        }

        # Fourier features
        components["obs_fourier"] = {
            "_target_": "metta.agent.lib.obs_tokenizers.ObsAttrEmbedFourier",
            "num_freqs": self.observations.fourier_freqs,
            "attr_embed_dim": self.observations.attr_embed_dim,
            "sources": [{"name": "obs_normalizer"}]
        }

        # Attention layers
        components["obs_attention"] = {
            "_target_": "metta.agent.lib.obs_enc.ObsSelfAttn",
            "out_dim": self.observations.attention_dim,
            "num_heads": self.observations.attention_heads,
            "num_layers": self.observations.attention_layers,
            "qk_dim": self.observations.attention_dim // self.observations.attention_heads,
            "use_mask": True,
            "use_cls_token": True,
            "sources": [{"name": "obs_fourier"}]
        }

        # Final encoding
        components["encoded_obs"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "obs_attention"}],
            "nn_params": {
                "out_features": self.core.hidden_size
            }
        }

        return components

    def _build_latent_attention_components(self) -> dict[str, Any]:
        """Build latent attention-based observation components."""
        components = {}

        # Observation tokenizer
        components["_obs_"] = {
            "_target_": "metta.agent.lib.obs_tokenizers.ObsTokenPadStrip",
            "sources": None
        }

        # Normalizer
        components["obs_normalizer"] = {
            "_target_": "metta.agent.lib.obs_tokenizers.ObsAttrValNorm",
            "sources": [{"name": "_obs_"}]
        }

        # Fourier features
        components["obs_fourier"] = {
            "_target_": "metta.agent.lib.obs_tokenizers.ObsAttrEmbedFourier",
            "num_freqs": self.observations.fourier_freqs,
            "attr_embed_dim": self.observations.attr_embed_dim,
            "sources": [{"name": "obs_normalizer"}]
        }

        # Latent query attention
        components["obs_latent_query_attn"] = {
            "_target_": "metta.agent.lib.obs_enc.ObsLatentAttn",
            "out_dim": self.observations.latent_query_dim,
            "use_mask": True,
            "num_query_tokens": self.observations.latent_query_tokens,
            "query_token_dim": self.observations.latent_query_dim,
            "num_heads": self.observations.attention_heads,
            "num_layers": 1,
            "sources": [{"name": "obs_fourier"}]
        }

        # Latent self attention
        components["obs_latent_self_attn"] = {
            "_target_": "metta.agent.lib.obs_enc.ObsSelfAttn",
            "out_dim": self.observations.attention_dim,
            "num_heads": self.observations.attention_heads,
            "num_layers": self.observations.attention_layers,
            "qk_dim": self.observations.latent_query_dim,
            "use_mask": False,
            "use_cls_token": True,
            "sources": [{"name": "obs_latent_query_attn"}]
        }

        # Final encoding
        components["encoded_obs"] = {
            "_target_": "metta.agent.lib.nn_layer_library.Linear",
            "sources": [{"name": "obs_latent_self_attn"}],
            "nn_params": {
                "out_features": self.core.hidden_size
            }
        }

        return components


def create_agent_config(cfg: DictConfig) -> AgentConfig:
    """Create AgentConfig from DictConfig, similar to create_trainer_config."""

    if not isinstance(cfg, DictConfig):
        raise ValueError("Agent config must be a DictConfig")

    # Convert to dict and let OmegaConf handle all interpolations
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise ValueError("Agent config must be a dict")

    # Extract only the fields that belong to AgentConfig
    # Skip _target_, components, and other fields that don't belong to AgentConfig
    agent_config_fields = {
        "observations",
        "core",
        "critic",
        "actor",
        "action_embeddings",
        "clip_range",
        "analyze_weights_interval",
        "policy_selector"
    }

    filtered_config = {}
    for key, value in config_dict.items():
        if key in agent_config_fields:
            filtered_config[key] = value

    return AgentConfig.model_validate(filtered_config)
