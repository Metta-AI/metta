import logging
from typing import Literal

from omegaconf import DictConfig
from pydantic import Field

from metta.agent.component_policy import ComponentPolicy
from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorKeySingleHead, MettaActorQuerySingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.nn_layer_library import Conv2d, Flatten, Linear
from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib.observation_normalizer import ObservationNormalizer
from metta.common.config import Config

logger = logging.getLogger(__name__)


class PPOConfig(Config):
    # PPO hyperparameters
    # Clip coefficient: 0.1 is conservative, common range 0.1-0.3 from PPO paper (Schulman et al., 2017)
    clip_coef: float = Field(default=0.1, gt=0, le=1.0)
    # Entropy coefficient: Type 2 default chosen from sweep
    ent_coef: float = Field(default=0.0021, ge=0)
    # GAE lambda: Type 2 default chosen from sweep, deviates from typical 0.95, bias/variance tradeoff
    gae_lambda: float = Field(default=0.916, ge=0, le=1.0)
    # Gamma: Type 2 default chosen from sweep, deviates from typical 0.99, suggests shorter
    # effective horizon for multi-agent
    gamma: float = Field(default=0.977, ge=0, le=1.0)

    # Training parameters
    # Gradient clipping: 0.5 is standard PPO default to prevent instability
    max_grad_norm: float = Field(default=0.5, gt=0)
    # Value function clipping: Matches policy clip for consistency
    vf_clip_coef: float = Field(default=0.1, ge=0)
    # Value coefficient: Type 2 default chosen from sweep, balances policy vs value loss
    vf_coef: float = Field(default=0.44, ge=0)
    # L2 regularization: Disabled by default, common in RL
    l2_reg_loss_coef: float = Field(default=0, ge=0)
    l2_init_loss_coef: float = Field(default=0, ge=0)

    # Normalization and clipping
    # Advantage normalization: Standard PPO practice for stability
    norm_adv: bool = True
    # Value loss clipping: PPO best practice from implementation details
    clip_vloss: bool = True
    # Target KL: None allows unlimited updates, common for stable environments
    target_kl: float | None = None


class OptimizerConfig(Config):
    type: Literal["adam", "muon"] = "adam"
    # Learning rate: Type 2 default chosen by sweep
    learning_rate: float = Field(default=0.000457, gt=0, le=1.0)
    # Beta1: Standard Adam default from Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
    beta1: float = Field(default=0.9, ge=0, le=1.0)
    # Beta2: Standard Adam default from Kingma & Ba (2014)
    beta2: float = Field(default=0.999, ge=0, le=1.0)
    # Epsilon: Type 2 default chosen arbitrarily
    eps: float = Field(default=1e-12, gt=0)
    # Weight decay: Disabled by default, common practice for RL to avoid over-regularization
    weight_decay: float = Field(default=0, ge=0)


class PrioritizedExperienceReplayConfig(Config):
    # Alpha=0 disables prioritization (uniform sampling), Type 2 default to be updated by sweep
    prio_alpha: float = Field(default=0.0, ge=0, le=1.0)
    # Beta0=0.6: From Schaul et al. (2016) "Prioritized Experience Replay" paper
    prio_beta0: float = Field(default=0.6, ge=0, le=1.0)


class VTraceConfig(Config):
    # V-trace rho clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    vtrace_rho_clip: float = Field(default=1.0, gt=0)
    # V-trace c clipping at 1.0: From IMPALA paper (Espeholt et al., 2018), standard for on-policy
    vtrace_c_clip: float = Field(default=1.0, gt=0)


class Fast(ComponentPolicy):
    """
    Fast CNN-based component policy - fastest but least robust to feature changes.
    """

    def _build_components(self) -> dict:
        """Build components for Fast CNN architecture."""
        return {
            "_obs_": ObsTokenToBoxShaper(
                name="_obs_",
                obs_shape=self.agent_attributes["obs_shape"],
                obs_width=self.agent_attributes["obs_width"],
                obs_height=self.agent_attributes["obs_height"],
                feature_normalizations=self.agent_attributes["feature_normalizations"],
                sources=None,
            ),
            "obs_normalizer": ObservationNormalizer(
                name="obs_normalizer",
                feature_normalizations=self.agent_attributes["feature_normalizations"],
                sources=[{"name": "_obs_"}],
            ),
            "cnn1": Conv2d(
                name="cnn1",
                nn_params=DictConfig({"out_channels": 64, "kernel_size": 5, "stride": 3, "padding": 0}),
                sources=[{"name": "obs_normalizer"}],
                **self.agent_attributes,
            ),
            "cnn2": Conv2d(
                name="cnn2",
                nn_params=DictConfig({"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 0}),
                sources=[{"name": "cnn1"}],
                **self.agent_attributes,
            ),
            "obs_flattener": Flatten(
                name="obs_flattener",
                sources=[{"name": "cnn2"}],
            ),
            "fc1": Linear(
                name="fc1",
                nn_params=DictConfig({"out_features": 128}),
                sources=[{"name": "obs_flattener"}],
                **self.agent_attributes,
            ),
            "encoded_obs": Linear(
                name="encoded_obs",
                nn_params=DictConfig({"out_features": 128}),
                sources=[{"name": "fc1"}],
                **self.agent_attributes,
            ),
            "_core_": LSTM(
                name="_core_",
                nn_params=DictConfig({"hidden_size": 128, "num_layers": 2}),
                sources=[{"name": "encoded_obs"}],
            ),
            "critic_1": Linear(
                name="critic_1",
                nn_params=DictConfig({"out_features": 1024}),
                sources=[{"name": "_core_"}],
                nonlinearity="nn.Tanh",
                effective_rank=True,
                **self.agent_attributes,
            ),
            "_value_": Linear(
                name="_value_",
                nn_params=DictConfig({"out_features": 1}),
                sources=[{"name": "critic_1"}],
                nonlinearity=None,
                **self.agent_attributes,
            ),
            "actor_1": Linear(
                name="actor_1",
                nn_params=DictConfig({"out_features": 512}),
                sources=[{"name": "_core_"}],
                **self.agent_attributes,
            ),
            "_action_embeds_": ActionEmbedding(
                name="_action_embeds_",
                nn_params=DictConfig({"num_embeddings": 100, "embedding_dim": 16}),
                sources=None,
            ),
            "actor_query": MettaActorQuerySingleHead(
                name="actor_query",
                sources=[{"name": "actor_1"}],
            ),
            "_action_": MettaActorKeySingleHead(
                name="_action_",
                sources=[{"name": "actor_query"}, {"name": "_action_embeds_"}],
            ),
        }
