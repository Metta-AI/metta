"""Latent-variable dynamics model component for model-based RL.

Based on "Learning Dynamics Model in Reinforcement Learning by Incorporating
the Long Term Future" (Ke et al., ICLR 2019).
"""

from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite

from metta.rl.training import EnvironmentMetaData

from .config import LatentDynamicsConfig


class LatentDynamicsModelComponent(nn.Module):
    """Latent-variable autoregressive dynamics model.

    Encodes (state, action, next_state) into stochastic latent variables using
    variational inference. An auxiliary task forces latents to carry long-term
    future information (returns, rewards, or future observations).
    """

    def __init__(self, config: LatentDynamicsConfig, env: Optional[EnvironmentMetaData] = None):
        super().__init__()
        self.config = config

        # Input/output keys
        self._in_key = config.in_key
        self._out_key = config.out_key
        self._action_key = config.action_key

        # Determine input dimensions
        # We'll infer obs_dim from the first forward pass using lazy linear
        self._latent_dim = config.latent_dim
        self._action_dim = 1  # Will be resolved from env
        self._obs_dim = None  # Will be inferred from first forward pass

        if env is not None:
            env_action_space = getattr(env, "action_space", None)
            if env_action_space is not None:
                resolved_action_dim = getattr(env_action_space, "n", None)
                if resolved_action_dim is not None:
                    self._action_dim = int(resolved_action_dim)

        # Build networks
        self._build_encoder()
        self._build_decoder()
        if config.use_auxiliary:
            self._build_auxiliary()

        # Flag to track if output layer is initialized
        self._decoder_output_initialized = False

    def _build_encoder(self):
        """Build encoder network: q(z | s_t, a_t, s_{t+1})."""
        layers = []
        # We'll use LazyLinear for the first layer to infer dimensions
        for i, hidden_dim in enumerate(self.config.encoder_hidden):
            if i == 0:
                layers.append(nn.LazyLinear(hidden_dim))
            else:
                layers.append(nn.Linear(layers[-2].out_features if i > 1 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self._encoder_base = nn.Sequential(*layers)

        # Output layers for mean and log variance
        if self.config.encoder_hidden:
            last_dim = self.config.encoder_hidden[-1]
        else:
            last_dim = None  # Will use LazyLinear

        if last_dim is not None:
            self._encoder_mean = nn.Linear(last_dim, self._latent_dim)
            self._encoder_logvar = nn.Linear(last_dim, self._latent_dim)
        else:
            self._encoder_mean = nn.LazyLinear(self._latent_dim)
            self._encoder_logvar = nn.LazyLinear(self._latent_dim)

    def _build_decoder(self):
        """Build decoder network: p(s_{t+1} | s_t, a_t, z)."""
        layers = []
        for i, hidden_dim in enumerate(self.config.decoder_hidden):
            if i == 0:
                layers.append(nn.LazyLinear(hidden_dim))
            else:
                layers.append(nn.Linear(self.config.decoder_hidden[i - 1], hidden_dim))
            layers.append(nn.ReLU())

        self._decoder_base = nn.Sequential(*layers)

        # Output layer will predict next observation
        # Use Identity to return last hidden layer, then we'll add proper output after first forward
        self._decoder_output = nn.Identity()

    def _build_auxiliary(self):
        """Build auxiliary network: p(future | z)."""
        layers = []
        for i, hidden_dim in enumerate(self.config.auxiliary_hidden):
            if i == 0:
                layers.append(nn.Linear(self._latent_dim, hidden_dim))
            else:
                layers.append(nn.Linear(self.config.auxiliary_hidden[i - 1], hidden_dim))
            layers.append(nn.ReLU())

        # Output dimension depends on future_type
        if self.config.future_type in ["returns", "rewards"]:
            out_dim = 1
        else:
            out_dim = None  # Will use LazyLinear for observations

        if self.config.auxiliary_hidden:
            if out_dim is not None:
                layers.append(nn.Linear(self.config.auxiliary_hidden[-1], out_dim))
            else:
                layers.append(nn.LazyLinear(out_dim))
        else:
            if out_dim is not None:
                layers.append(nn.Linear(self._latent_dim, out_dim))
            else:
                layers.append(nn.LazyLinear(out_dim))

        self._auxiliary_net = nn.Sequential(*layers)

    def encode(self, obs_t: torch.Tensor, action_t: torch.Tensor, obs_next: torch.Tensor):
        """Encode transition into latent distribution.

        Args:
            obs_t: Current observation (B, obs_dim)
            action_t: Action taken (B,) or (B, action_dim)
            obs_next: Next observation (B, obs_dim)

        Returns:
            z_mean: Mean of latent distribution (B, latent_dim)
            z_logvar: Log variance of latent distribution (B, latent_dim)
        """
        # Convert action to one-hot if needed
        if action_t.dim() == 1:
            action_onehot = torch.nn.functional.one_hot(action_t.long(), num_classes=self._action_dim).float()
        else:
            action_onehot = action_t.float()

        # Concatenate inputs
        encoder_input = torch.cat([obs_t, action_onehot, obs_next], dim=-1)

        # Pass through encoder
        hidden = self._encoder_base(encoder_input)
        z_mean = self._encoder_mean(hidden)
        z_logvar = self._encoder_logvar(hidden)

        return z_mean, z_logvar

    def reparameterize(self, z_mean: torch.Tensor, z_logvar: torch.Tensor):
        """Sample from latent distribution using reparameterization trick.

        Args:
            z_mean: Mean of latent distribution
            z_logvar: Log variance of latent distribution

        Returns:
            z: Sampled latent variable
        """
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, obs_t: torch.Tensor, action_t: torch.Tensor, z: torch.Tensor):
        """Decode latent to predict next observation.

        Args:
            obs_t: Current observation (B, obs_dim)
            action_t: Action taken (B,) or (B, action_dim)
            z: Latent variable (B, latent_dim)

        Returns:
            obs_next_pred: Predicted next observation (B, obs_dim)
        """
        # Infer obs_dim on first call
        if self._obs_dim is None:
            self._obs_dim = obs_t.shape[-1]

        # Convert action to one-hot if needed
        if action_t.dim() == 1:
            action_onehot = torch.nn.functional.one_hot(action_t.long(), num_classes=self._action_dim).float()
        else:
            action_onehot = action_t.float()

        # Concatenate inputs
        decoder_input = torch.cat([obs_t, action_onehot, z], dim=-1)

        # Pass through decoder
        hidden = self._decoder_base(decoder_input)

        # Initialize output layer on first use
        if not self._decoder_output_initialized:
            if self.config.decoder_hidden:
                last_hidden_dim = self.config.decoder_hidden[-1]
            else:
                # No hidden layers, use input dim
                last_hidden_dim = decoder_input.shape[-1]

            # Create and register the output layer as a submodule
            output_layer = nn.Linear(last_hidden_dim, self._obs_dim).to(hidden.device)
            # Use add_module to properly register it
            self.add_module("_decoder_output_layer", output_layer)
            self._decoder_output = output_layer
            self._decoder_output_initialized = True

        obs_next_pred = self._decoder_output(hidden)

        return obs_next_pred

    def predict_future(self, z: torch.Tensor):
        """Predict future information from latent (auxiliary task).

        Args:
            z: Latent variable (B, latent_dim)

        Returns:
            future_pred: Predicted future information
        """
        if not self.config.use_auxiliary:
            return None

        return self._auxiliary_net(z)

    def forward(self, td: TensorDict) -> TensorDict:
        """Forward pass through the dynamics model.

        During training, expects:
        - obs_t: Current observation
        - action_t: Action taken
        - obs_next: Next observation (for encoding)

        During inference, can generate predictions from obs_t and action_t.
        """
        obs_t = td[self._in_key]

        # Get actions
        action_t = td.get(self._action_key)
        if action_t is None:
            # Default to zero action if not provided
            batch_size = obs_t.shape[0]
            action_t = torch.zeros(batch_size, dtype=torch.long, device=obs_t.device)

        # Check if we have next observation for encoding
        obs_next = td.get(f"{self._in_key}_next")

        if obs_next is not None:
            # Training mode: encode the transition
            z_mean, z_logvar = self.encode(obs_t, action_t, obs_next)
            z = self.reparameterize(z_mean, z_logvar)

            # Store latent distribution parameters
            td.set("latent_mean", z_mean)
            td.set("latent_logvar", z_logvar)
            td.set("latent", z)

            # Decode to reconstruct next observation
            obs_next_pred = self.decode(obs_t, action_t, z)
            td.set("obs_next_pred", obs_next_pred)

            # Predict future information (auxiliary task)
            if self.config.use_auxiliary:
                future_pred = self.predict_future(z)
                td.set("future_pred", future_pred)
        else:
            # Inference mode: sample from prior
            batch_size = obs_t.shape[0]
            device = obs_t.device
            z_mean = torch.zeros(batch_size, self._latent_dim, device=device)
            z_logvar = torch.zeros(batch_size, self._latent_dim, device=device)
            z = self.reparameterize(z_mean, z_logvar)

            td.set("latent", z)

            # Decode to predict next observation
            obs_next_pred = self.decode(obs_t, action_t, z)
            td.set("obs_next_pred", obs_next_pred)

        # Output latent representation
        td.set(self._out_key, z)

        return td

    def get_agent_experience_spec(self) -> Composite:
        """Return empty spec since this is an internal component."""
        return Composite({})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        """Initialize the component for a specific environment."""
        # Update action dimension if available
        if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
            self._action_dim = int(env.action_space.n)

        self.to(device)
        return None

    def reset_memory(self) -> None:
        """Reset any recurrent state (none for this component)."""
        pass

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        return next(self.parameters()).device
