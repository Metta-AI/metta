"""Muesli configuration and hyperparameters."""

from pydantic import Field
from metta.rl.trainer_config import BaseModelWithForbidExtra


class CMPOConfig(BaseModelWithForbidExtra):
    """CMPO (Clipped Maximum a Posteriori Policy Optimization) configuration."""
    # Advantage clipping bound (controls max TV distance: max D_TV = tanh(c/2))
    clip_bound: float = Field(default=1.0, gt=0, le=5.0,
                             description="Advantage clipping bound for CMPO")
    # Weight for CMPO regularization loss
    cmpo_weight: float = Field(default=1.0, ge=0,
                              description="Weight for CMPO KL regularization")
    # Variance normalization decay rate
    variance_decay: float = Field(default=0.99, ge=0, le=1.0,
                                 description="Running variance estimate decay rate")


class ModelLearningConfig(BaseModelWithForbidExtra):
    """Model learning configuration."""
    # Number of unroll steps for model supervision
    unroll_steps: int = Field(default=5, ge=1, le=10,
                             description="Number of steps to unroll the model")
    # Weight for model loss
    model_weight: float = Field(default=1.0, ge=0,
                               description="Weight for model learning loss")
    # Weight for reward prediction loss
    reward_weight: float = Field(default=1.0, ge=0,
                                description="Weight for reward prediction loss")
    # Weight for value prediction loss  
    value_weight: float = Field(default=1.0, ge=0,
                               description="Weight for value prediction loss")
    # Weight for policy model loss
    policy_model_weight: float = Field(default=1.0, ge=0,
                                      description="Weight for policy model loss")


class CategoricalConfig(BaseModelWithForbidExtra):
    """Categorical value/reward representation configuration."""
    # Number of bins for categorical representation
    support_size: int = Field(default=601, ge=101, le=1001,
                             description="Number of bins for categorical representation")
    # Value range for categorical representation
    value_min: float = Field(default=-300.0,
                            description="Minimum value for categorical support")
    value_max: float = Field(default=300.0,
                            description="Maximum value for categorical support")


class ReplayConfig(BaseModelWithForbidExtra):
    """Replay buffer configuration."""
    # Replay buffer capacity
    capacity: int = Field(default=100000, gt=0,
                         description="Maximum number of transitions to store")
    # Fraction of replay data in each batch
    replay_fraction: float = Field(default=0.75, ge=0, le=1.0,
                                  description="Fraction of replay data in each batch")
    # Priority exponent (alpha=0 for uniform sampling)
    priority_alpha: float = Field(default=0.0, ge=0, le=1.0,
                                 description="Priority exponent for replay sampling")
    # Importance sampling correction (beta)
    priority_beta: float = Field(default=0.6, ge=0, le=1.0,
                                description="Importance sampling correction")


class TargetNetworkConfig(BaseModelWithForbidExtra):
    """Target network configuration."""
    # EMA update rate for target network
    tau: float = Field(default=0.1, gt=0, le=1.0,
                      description="Exponential moving average update rate")
    # Update frequency (in training steps)
    update_freq: int = Field(default=1, ge=1,
                           description="Target network update frequency")


class RetraceConfig(BaseModelWithForbidExtra):
    """Retrace configuration for off-policy correction."""
    # Lambda parameter for Retrace
    lambda_: float = Field(default=0.95, ge=0, le=1.0,
                          description="Lambda parameter for Retrace")
    # Maximum importance sampling ratio
    rho_max: float = Field(default=1.0, ge=1.0,
                          description="Maximum importance sampling ratio")


class NetworkConfig(BaseModelWithForbidExtra):
    """Neural network architecture configuration."""
    # Hidden size for LSTM and linear layers
    hidden_size: int = Field(default=512, gt=0,
                           description="Hidden size for LSTM and linear layers")
    # Number of channels in conv layers
    conv_channels: int = Field(default=64, gt=0,
                             description="Number of channels in conv layers")
    # LSTM hidden size for dynamics network
    dynamics_hidden_size: int = Field(default=1024, gt=0,
                                    description="LSTM hidden size for dynamics network")
    # Number of LSTM layers
    num_lstm_layers: int = Field(default=1, ge=1,
                               description="Number of LSTM layers")
    # Policy head initialization gain
    policy_init_gain: float = Field(default=0.01, gt=0,
                                  description="Initialization gain for policy head")


class MuesliConfig(BaseModelWithForbidExtra):
    """Main Muesli configuration."""
    # CMPO configuration
    cmpo: CMPOConfig = Field(default_factory=CMPOConfig)
    
    # Model learning configuration
    model_learning: ModelLearningConfig = Field(default_factory=ModelLearningConfig)
    
    # Categorical representation configuration
    categorical: CategoricalConfig = Field(default_factory=CategoricalConfig)
    
    # Replay buffer configuration
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    
    # Target network configuration
    target_network: TargetNetworkConfig = Field(default_factory=TargetNetworkConfig)
    
    # Retrace configuration
    retrace: RetraceConfig = Field(default_factory=RetraceConfig)
    
    # Network architecture configuration
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    
    # Whether to use mixed on-policy/off-policy training
    mixed_training: bool = Field(default=True,
                               description="Use mixed on-policy/off-policy training")
    
    # Maximum gradient norm for clipping
    max_grad_norm: float = Field(default=40.0, gt=0,
                               description="Maximum gradient norm for clipping")