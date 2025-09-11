import logging

from metta.agent.component_policy import ComponentPolicy
from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorKeySingleHead, MettaActorQuerySingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.metta_layer import NNParams
from metta.agent.lib.nn_layer_library import Conv2d, Flatten, Linear
from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib.observation_normalizer import ObservationNormalizer
from metta.rl.training.training_environment import EnvironmentMetaData

logger = logging.getLogger(__name__)


class Fast(ComponentPolicy):
    """
    Fast CNN-based component policy - fastest but least robust to feature changes.
    """

    def _output_heads(self) -> list[str]:
        return ["_action_", "_value_"]

    def _build_components(self, env_metadata: EnvironmentMetaData) -> dict:
        feature_normalizations = {feature.id: feature.normalization for feature in env_metadata.obs_features.values()}
        """Build components for Fast CNN architecture."""
        return {
            "_obs_": ObsTokenToBoxShaper(
                name="_obs_",
                obs_shape=self._obs_shape,
                obs_width=env_metadata.obs_width,
                obs_height=env_metadata.obs_height,
                feature_normalizations=feature_normalizations,
            ),
            "obs_normalizer": ObservationNormalizer(
                name="obs_normalizer",
                feature_normalizations=feature_normalizations,
                sources=["_obs_"],
            ),
            "cnn1": Conv2d(
                name="cnn1",
                nn_params=NNParams(out_channels=64, kernel_size=5, stride=3, padding=0),
                sources=["obs_normalizer"],
            ),
            "cnn2": Conv2d(
                name="cnn2",
                nn_params=NNParams(out_channels=64, kernel_size=3, stride=1, padding=0),
                sources=["cnn1"],
            ),
            "obs_flattener": Flatten(
                name="obs_flattener",
                sources=["cnn2"],
            ),
            "fc1": Linear(
                name="fc1",
                nn_params=NNParams(out_features=128),
                sources=["obs_flattener"],
            ),
            "encoded_obs": Linear(
                name="encoded_obs",
                nn_params=NNParams(out_features=128),
                sources=["fc1"],
            ),
            "_core_": LSTM(
                name="_core_",
                nn_params=NNParams(hidden_size=128, num_layers=2),
                sources=["encoded_obs"],
            ),
            "critic_1": Linear(
                name="critic_1",
                nn_params=NNParams(out_features=1024),
                sources=["_core_"],
                nonlinearity="nn.Tanh",
            ),
            "_value_": Linear(
                name="_value_",
                nn_params=NNParams(out_features=1),
                sources=["critic_1"],
                nonlinearity=None
            ),
            "actor_1": Linear(
                name="actor_1",
                nn_params=NNParams(out_features=512),
                sources=["_core_"],
            ),
            "_action_embeds_": ActionEmbedding(
                name="_action_embeds_",
                num_embeddings=100,
                embedding_dim=16,
            ),
            "actor_query": MettaActorQuerySingleHead(
                name="actor_query",
                sources=["actor_1", "_action_embeds_"],
            ),
            "_action_": MettaActorKeySingleHead(
                name="_action_",
                sources=["actor_query", "_action_embeds_"],
            ),
        }
