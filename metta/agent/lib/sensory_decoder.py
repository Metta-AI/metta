import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class SensoryDecoder(LayerBase):
    """Sensory prediction decoder that reconstructs sensory inputs from latent representations.

    This component takes the encoded representation from the core network and attempts
    to reconstruct the original sensory inputs. This acts as a regularization signal
    to ensure the representations retain sensory information.
    """

    def __init__(self, decoder_hidden_dim: int = 256, **cfg):
        super().__init__(**cfg)
        self.decoder_hidden_dim = decoder_hidden_dim

    def _make_net(self):
        """Create the sensory decoder network."""
        input_size = self._in_tensor_shapes[0][0]

        # Decoder network: latent -> hidden -> sensory reconstruction
        self._net = nn.Sequential(
            nn.Linear(input_size, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self._out_tensor_shape[0])
        )

        return self._net

    def _forward(self, td: TensorDict):
        """Forward pass for sensory reconstruction."""
        latent = td[self._sources[0]["name"]]
        sensory_reconstruction = self._net(latent)
        td[self._name] = sensory_reconstruction
        return td

    def compute_auxiliary_loss(self, td: TensorDict, original_sensory: torch.Tensor) -> torch.Tensor:
        """Compute sensory reconstruction loss.

        Args:
            td: TensorDict containing the sensory reconstruction
            original_sensory: Original sensory inputs to reconstruct

        Returns:
            Reconstruction loss
        """
        reconstruction = td[self._name]
        # MSE loss between reconstruction and original sensory inputs
        loss = nn.functional.mse_loss(reconstruction, original_sensory)
        return loss
