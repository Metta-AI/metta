"""Configuration for latent-variable dynamics model component."""

from typing import Any, Optional

from metta.agent.components.component_config import ComponentConfig


class LatentDynamicsConfig(ComponentConfig):
    """Configuration for latent-variable dynamics model.

    This model learns to predict future states by encoding transitions into
    stochastic latent variables. An auxiliary task forces the latents to
    carry long-term future information.
    """

    name: str = "latent_dynamics"

    # Input/output keys
    in_key: str = "encoded_obs"
    out_key: str = "latent_dynamics_hidden"
    action_key: str = "last_actions"

    # Architecture
    latent_dim: int = 32
    encoder_hidden: list[int] = [64, 64]
    decoder_hidden: list[int] = [64, 64]
    auxiliary_hidden: list[int] = [32]

    # Loss weights
    beta_kl: float = 0.01
    gamma_auxiliary: float = 1.0

    # Future prediction
    future_horizon: int = 5
    future_type: str = "returns"  # "returns", "rewards", or "observations"

    # Training
    use_auxiliary: bool = True

    # Performance
    use_triton: bool = True  # Use Triton kernels if available for faster computation

    def make_component(self, env: Optional[Any] = None):  # type: ignore[override]
        """Create LatentDynamicsModelComponent instance."""
        from .latent_dynamics_component import LatentDynamicsModelComponent

        return LatentDynamicsModelComponent(config=self, env=env)
