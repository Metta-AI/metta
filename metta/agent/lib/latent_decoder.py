import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class LatentDecoder(LayerBase):
    """Latent prediction decoder that predicts future latent states from current representations.

    This component takes the current latent representation and attempts to predict
    the next latent state. This encourages the representations to encode temporal
    dynamics and predictive information.
    """

    def __init__(self, decoder_hidden_dim: int = 256, **cfg):
        super().__init__(**cfg)
        self.decoder_hidden_dim = decoder_hidden_dim

    def _make_net(self):
        """Create the latent decoder network."""
        input_size = self._in_tensor_shapes[0][0]

        # Decoder network: current latent -> hidden -> next latent prediction
        self._net = nn.Sequential(
            nn.Linear(input_size, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self._out_tensor_shape[0])
        )

        return self._net

    def _forward(self, td: TensorDict):
        """Forward pass for latent prediction."""
        current_latent = td[self._sources[0]["name"]]
        next_latent_prediction = self._net(current_latent)
        td[self._name] = next_latent_prediction
        return td

    def compute_auxiliary_loss(self, td: TensorDict, next_latent: torch.Tensor) -> torch.Tensor:
        """Compute latent prediction loss.

        Args:
            td: TensorDict containing the latent prediction
            next_latent: Actual next latent state

        Returns:
            Prediction loss
        """
        prediction = td[self._name]
        # MSE loss between prediction and actual next latent state
        loss = nn.functional.mse_loss(prediction, next_latent)
        return loss
