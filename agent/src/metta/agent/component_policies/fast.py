import logging

import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from metta.agent.component_policy import ComponentPolicy
from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorSingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.nn_layer_library import Conv2d, Flatten, Linear
from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib.observation_normalizer import ObservationNormalizer

logger = logging.getLogger(__name__)


class Fast(ComponentPolicy):
    """
    Fast CNN-based component policy - fastest but least robust to feature changes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            "_action_": MettaActorSingleHead(
                name="_action_",
                sources=[{"name": "actor_1"}, {"name": "_action_embeds_"}],
            ),
        }

    def forward(self, td: TensorDict, state=None, action=None) -> TensorDict:
        """Forward pass with support for both token observations and latent observations."""

        # Check if we have latent observations
        if "latent_obs" in td:
            return self._forward_latent(td, state, action)
        else:
            # Use the regular ComponentPolicy forward method for token observations
            return super().forward(td, state, action)

    def _forward_latent(self, td: TensorDict, state=None, action=None) -> TensorDict:
        """Forward pass for latent observations (bypass CNN pipeline)."""
        # Handle BPTT reshaping
        if td.batch_dims > 1:
            B = td.batch_size[0]
            TT = td.batch_size[1]
            td = td.reshape(td.batch_size.numel())  # flatten to BT
            td.set("bptt", torch.full((B * TT,), TT, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B * TT,), B, device=td.device, dtype=torch.long))
        else:
            B = td.batch_size.numel()
            td.set("bptt", torch.full((B,), 1, device=td.device, dtype=torch.long))
            td.set("batch", torch.full((B,), B, device=td.device, dtype=torch.long))

        # Use latent observations directly (already 128-dimensional)
        latent_obs = td["latent_obs"]
        td["encoded_obs"] = latent_obs

        # Run LSTM on encoded observations
        self.components["_core_"](td)

        # Run value and action components
        self.components["_value_"](td)
        self.components["_action_"](td)

        if action is None:
            output_td = self.forward_inference(td)
        else:
            output_td = self.forward_training(td, action)
            # Reshape back for training mode
            batch_size = td["batch"][0].item()
            bptt_size = td["bptt"][0].item()
            output_td = output_td.reshape(batch_size, bptt_size)

        return output_td
