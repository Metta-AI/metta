import logging

from omegaconf import DictConfig

from metta.agent.component_policy import ComponentPolicy
from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorKeySingleHead, MettaActorQuerySingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.nn_layer_library import Conv2d, Flatten, Linear
from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib.observation_normalizer import ObservationNormalizer

logger = logging.getLogger(__name__)


class FastDynamics(ComponentPolicy):
    """
    Fast CNN-based component policy - fastest but least robust to feature changes.
    This one has dynamics prediction heads for use with the dynamics.py loss (Muesli).
    """

    def _get_output_heads(self) -> list[str]:
        return ["_action_", "_value_", "returns_pred", "reward_pred"]

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
                sources=[{"name": "actor_1"}, {"name": "_action_embeds_"}],
            ),
            "_action_": MettaActorKeySingleHead(
                name="_action_",
                sources=[{"name": "actor_query"}, {"name": "_action_embeds_"}],
            ),
            "returns_pred": Linear(
                name="returns_pred",
                nn_params=DictConfig({"out_features": 1}),
                sources=[{"name": "critic_1"}],
                nonlinearity=None,
                **self.agent_attributes,
            ),
            "reward_pred": Linear(
                name="reward_pred",
                nn_params=DictConfig({"out_features": 1}),
                sources=[{"name": "critic_1"}],
                nonlinearity=None,
                **self.agent_attributes,
            ),
        }
