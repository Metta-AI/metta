"""Neural network models for meta-analysis training curve prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EnvironmentVAE(nn.Module):
    """Variational Autoencoder for environment configurations."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with reconstruction and KL loss."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class AgentVAE(nn.Module):
    """Variational Autoencoder for agent hyperparameters."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with reconstruction and KL loss."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class RewardPredictor(nn.Module):
    """Predicts reward curves from environment and agent latent representations."""

    def __init__(
        self,
        env_latent_dim: int = 32,
        agent_latent_dim: int = 32,
        curve_length: int = 100,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.curve_length = curve_length

        # Combine latent representations
        combined_dim = env_latent_dim + agent_latent_dim

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, curve_length),
        )

    def forward(
        self,
        env_latent: torch.Tensor,
        agent_latent: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward curve from latent representations."""

        # Concatenate latent representations
        combined = torch.cat([env_latent, agent_latent], dim=1)

        # Predict reward curve
        curve = self.predictor(combined)

        return curve


class MetaAnalysisModel(nn.Module):
    """Combined model for end-to-end training."""

    def __init__(
        self,
        env_input_dim: int,
        agent_input_dim: int,
        env_latent_dim: int = 32,
        agent_latent_dim: int = 32,
        curve_length: int = 100,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.env_vae = EnvironmentVAE(env_input_dim, env_latent_dim, hidden_dim)
        self.agent_vae = AgentVAE(agent_input_dim, agent_latent_dim, hidden_dim)
        self.reward_predictor = RewardPredictor(
            env_latent_dim, agent_latent_dim, curve_length, hidden_dim * 2
        )

    def forward(
        self,
        env_config: torch.Tensor,
        agent_config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all components."""

        # Encode environment and agent configs
        env_recon, env_mu, env_logvar = self.env_vae(env_config)
        agent_recon, agent_mu, agent_logvar = self.agent_vae(agent_config)

        # Get latent representations
        env_latent = self.env_vae.reparameterize(env_mu, env_logvar)
        agent_latent = self.agent_vae.reparameterize(agent_mu, agent_logvar)

        # Predict reward curve
        predicted_curve = self.reward_predictor(env_latent, agent_latent)

        return (
            env_recon, env_mu, env_logvar,
            agent_recon, agent_mu, agent_logvar,
            predicted_curve
        )

    def encode_only(
        self,
        env_config: torch.Tensor,
        agent_config: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode configs to latent space without sampling."""

        env_mu, env_logvar = self.env_vae.encode(env_config)
        agent_mu, agent_logvar = self.agent_vae.encode(agent_config)

        return env_mu, agent_mu

    def predict_curve(
        self,
        env_latent: torch.Tensor,
        agent_latent: torch.Tensor
    ) -> torch.Tensor:
        """Predict reward curve from latent representations."""

        return self.reward_predictor(env_latent, agent_latent)
