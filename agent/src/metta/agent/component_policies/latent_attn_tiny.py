import logging

from omegaconf import DictConfig

from metta.agent.component_policy import ComponentPolicy
from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorKeySingleHead, MettaActorQuerySingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.nn_layer_library import Linear
from metta.agent.lib.obs_enc import ObsLatentAttn
from metta.agent.lib.obs_tokenizers import ObsAttrEmbedFourier, ObsAttrValNorm, ObsTokenPadStrip

logger = logging.getLogger(__name__)


class LatentAttnTiny(ComponentPolicy):
    """
    Latent attention tiny model - fastest attention-based model with minimal expressivity.
    """

    def _get_output_heads(self) -> list[str]:
        return ["_action_", "_value_"]

    def _build_components(self) -> dict:
        """Build components for LatentAttnTiny architecture."""
        return {
            "_obs_": ObsTokenPadStrip(
                name="_obs_",
                obs_shape=self.agent_attributes["obs_shape"],
                sources=None,
            ),
            "obs_normalizer": ObsAttrValNorm(
                name="obs_normalizer",
                feature_normalizations=self.agent_attributes["feature_normalizations"],
                sources=[{"name": "_obs_"}],
            ),
            "obs_fourier": ObsAttrEmbedFourier(
                name="obs_fourier",
                attr_embed_dim=10,
                num_freqs=4,
                sources=[{"name": "obs_normalizer"}],
            ),
            "obs_cross_attn": ObsLatentAttn(
                name="obs_cross_attn",
                out_dim=128,
                use_mask=True,
                num_layers=2,
                num_heads=4,
                num_query_tokens=1,
                use_cls_token=True,
                query_token_dim=32,
                qk_dim=32,
                mlp_out_hidden_dim=256,
                sources=[{"name": "obs_fourier"}],
            ),
            "_core_": LSTM(
                name="_core_",
                nn_params=DictConfig({"hidden_size": 128, "num_layers": 2}),
                sources=[{"name": "obs_cross_attn"}],
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
        }
