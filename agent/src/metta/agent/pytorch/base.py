"""Base class for PyTorch agents with TensorDict interface."""

import pufferlib.models
import torch
from torchrl.data import Composite, UnboundedDiscrete


class PytorchAgentBase(pufferlib.models.LSTMWrapper):
    """Base class for PyTorch agents that provides MettaAgent compatibility."""

    def get_agent_experience_spec(self):
        """Provide experience spec for compatibility with trainer."""
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
        )

    def reset_memory(self):
        """Reset LSTM memory if needed."""
        # LSTM state is managed through TensorDict, no explicit reset needed
        pass
