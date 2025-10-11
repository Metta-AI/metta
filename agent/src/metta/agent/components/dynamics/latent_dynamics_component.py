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
from .triton_kernels import reparameterize as triton_reparameterize


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

        # Per-environment latent state for multi-agent gridworld rollouts
        # Each environment maintains its own latent mean and logvar during inference
        self.register_buffer("latent_mean_state", torch.empty(0, self._latent_dim))
        self.register_buffer("latent_logvar_state", torch.empty(0, self._latent_dim))
        self.max_num_envs = 0

    def __setstate__(self, state):
        """Ensure latent states are properly initialized after loading from checkpoint.

        This is important for multi-agent gridworld environments to avoid batch size
        mismatches when loading a saved model.
        """
        self.__dict__.update(state)
        # Reset hidden states when loading from checkpoint
        if not hasattr(self, "latent_mean_state"):
            self.latent_mean_state = torch.empty(0, self._latent_dim)
        if not hasattr(self, "latent_logvar_state"):
            self.latent_logvar_state = torch.empty(0, self._latent_dim)
        if not hasattr(self, "max_num_envs"):
            self.max_num_envs = 0

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
            obs_t: Current observation (batch, obs_dim)
            action_t: Action taken (batch,) or (batch, action_dim)
            obs_next: Next observation (batch, obs_dim)

        Returns:
            z_mean: Mean of latent distribution (batch, latent_dim)
            z_logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # Convert action to one-hot if needed
        if action_t.dim() == 1:
            action_onehot = torch.nn.functional.one_hot(action_t.long(), num_classes=self._action_dim).float()
        else:
            action_onehot = action_t.float()

        # Concatenate inputs: [obs_t, action, obs_next]
        encoder_input = torch.cat([obs_t, action_onehot, obs_next], dim=-1)

        # Pass through encoder
        hidden = self._encoder_base(encoder_input)
        z_mean = self._encoder_mean(hidden)
        z_logvar = self._encoder_logvar(hidden)

        return z_mean, z_logvar

    def reparameterize(self, z_mean: torch.Tensor, z_logvar: torch.Tensor):
        """Sample from latent distribution using reparameterization trick.

        Uses Triton-optimized kernel when available and enabled for better performance.

        Args:
            z_mean: Mean of latent distribution (batch, latent_dim)
            z_logvar: Log variance of latent distribution (batch, latent_dim)

        Returns:
            z: Sampled latent variable (batch, latent_dim)
        """
        # Use Triton kernel if enabled and available
        return triton_reparameterize(z_mean, z_logvar, use_triton=self.config.use_triton)

    def decode(self, obs_t: torch.Tensor, action_t: torch.Tensor, z: torch.Tensor):
        """Decode latent to predict next observation.

        Args:
            obs_t: Current observation (batch, obs_dim)
            action_t: Action taken (batch,) or (batch, action_dim)
            z: Latent variable (batch, latent_dim)

        Returns:
            obs_next_pred: Predicted next observation (batch, obs_dim)
        """
        # Infer obs_dim on first call
        if self._obs_dim is None:
            self._obs_dim = obs_t.shape[-1]

        # Convert action to one-hot if needed
        if action_t.dim() == 1:
            action_onehot = torch.nn.functional.one_hot(action_t.long(), num_classes=self._action_dim).float()
        else:
            action_onehot = action_t.float()

        # Concatenate inputs: [obs_t, action, z]
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

        Handles multi-agent gridworld environments by maintaining separate latent
        states for each environment during rollouts.

        During training, expects:
        - obs_t: Current observation
        - action_t: Action taken
        - obs_next: Next observation (for encoding)

        During inference (rollout), maintains per-environment latent states.
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

        # Get environment IDs for per-environment state management
        training_env_ids = td.get("training_env_ids", None)
        batch_size = obs_t.shape[0]

        if training_env_ids is None:
            # No env IDs provided - use sequential IDs
            training_env_ids = torch.arange(batch_size, device=obs_t.device)
        else:
            training_env_ids = training_env_ids.reshape(batch_size)

        # Ensure we have allocated state for all environments
        if training_env_ids.numel() > 0:
            max_env_id = int(training_env_ids.max().item()) + 1
            if max_env_id > self.max_num_envs:
                self._ensure_capacity(max_env_id, obs_t.device)

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
            # Inference mode: use per-environment latent states for rollouts
            # This maintains separate latent dynamics for each parallel environment
            z_mean = self.latent_mean_state[training_env_ids]
            z_logvar = self.latent_logvar_state[training_env_ids]

            # Sample latent variable
            z = self.reparameterize(z_mean, z_logvar)

            # Update the latent state for next step (simple persistence model)
            # In a full implementation, this could predict the next latent state
            self.latent_mean_state[training_env_ids] = z_mean.detach()
            self.latent_logvar_state[training_env_ids] = z_logvar.detach()

            td.set("latent", z)

            # Decode to predict next observation
            obs_next_pred = self.decode(obs_t, action_t, z)
            td.set("obs_next_pred", obs_next_pred)

        # Output latent representation
        td.set(self._out_key, z)

        return td

    def _ensure_capacity(self, num_envs: int, device: torch.device) -> None:
        """Ensure we have allocated latent states for the specified number of environments.

        This is critical for multi-agent gridworld rollouts where each environment
        maintains its own hidden state.
        """
        if num_envs > self.max_num_envs:
            num_to_add = num_envs - self.max_num_envs

            # Allocate new states (initialized from prior N(0, 1))
            new_mean = torch.zeros(num_to_add, self._latent_dim, device=device)
            new_logvar = torch.zeros(num_to_add, self._latent_dim, device=device)

            # Concatenate with existing states
            self.latent_mean_state = torch.cat([self.latent_mean_state, new_mean.detach()], dim=0).to(device)
            self.latent_logvar_state = torch.cat([self.latent_logvar_state, new_logvar.detach()], dim=0).to(device)

            self.max_num_envs = num_envs

    def get_agent_experience_spec(self) -> Composite:
        """Return empty spec since this is an internal component."""
        return Composite({})

    def initialize_to_environment(self, env: EnvironmentMetaData, device: torch.device) -> Optional[str]:
        """Initialize the component for a specific multi-agent gridworld environment."""
        # Update action dimension if available
        if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
            self._action_dim = int(env.action_space.n)

        # Reset memory for new environment
        self.reset_memory()

        # Allocate states for parallel environments
        num_envs = getattr(env, "num_envs", None)
        if num_envs is None:
            num_envs = getattr(env, "num_agents", 0)

        if isinstance(num_envs, int) and num_envs > 0:
            self._ensure_capacity(num_envs, device)

        self.to(device)
        return None

    def reset_memory(self) -> None:
        """Reset per-environment latent states.

        This is called when starting a new training run or evaluation,
        ensuring each environment starts with a clean latent state.
        """
        if self.latent_mean_state.numel() > 0:
            self.latent_mean_state.fill_(0.0)
            self.latent_logvar_state.fill_(0.0)

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        return next(self.parameters()).device
