"""
Full-context transformer agent for Metta.

This agent uses a transformer with GTrXL-style stabilization that views the entire BPTT
trajectory at once, using all observations as context to predict actions.
Optimized for parallel processing across thousands of environments/agents.
"""

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from torch import nn

from metta.agent.modules.full_context_transformer import FullContextTransformer
from metta.agent.modules.transformer_wrapper import TransformerWrapper
from metta.agent.pytorch.pytorch_agent_mixin import PyTorchAgentMixin

logger = logging.getLogger(__name__)


class Policy(nn.Module):
    """Policy network using full-context transformer."""
    
    def __init__(
        self,
        env,
        input_size: int = 256,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
    ):
        """Initialize the policy network.
        
        Args:
            env: Environment
            input_size: Input embedding size  
            hidden_size: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking
            use_gating: Whether to use GRU gating
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.is_continuous = False
        
        # Get action space dimensions
        if hasattr(env, 'single_action_space'):
            action_space = env.single_action_space
        else:
            action_space = env.action_space
        
        if hasattr(action_space, 'nvec'):
            # Multi-discrete action space
            self.action_dims = list(action_space.nvec)
        elif hasattr(action_space, 'n'):
            # Discrete action space
            self.action_dims = [action_space.n]
        else:
            raise NotImplementedError(f"Action space {type(action_space)} not supported")
        
        # Get observation shape
        if hasattr(env, 'single_observation_space'):
            obs_space = env.single_observation_space
        else:
            obs_space = env.observation_space
        self.obs_shape = obs_space.shape
        
        # Calculate observation dimension (handle various shapes)
        if len(self.obs_shape) == 3:
            # Image observations (H, W, C)
            obs_dim = self.obs_shape[0] * self.obs_shape[1] * self.obs_shape[2]
        elif len(self.obs_shape) == 2:
            # Matrix observations (M, F)
            obs_dim = self.obs_shape[0] * self.obs_shape[1]
        elif len(self.obs_shape) == 1:
            # Vector observations
            obs_dim = self.obs_shape[0]
        else:
            raise ValueError(f"Unsupported observation shape: {self.obs_shape}")
        
        # Simple observation encoder with layer norm for stability
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        
        # Full-context transformer core (optimized for parallel processing)
        logger.info(f"Creating FullContextTransformer with hidden_size={hidden_size}, "
                   f"n_heads={n_heads}, n_layers={n_layers}")
        self._transformer = FullContextTransformer(
            d_model=hidden_size,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
            use_gating=use_gating,
        )
        logger.info("FullContextTransformer created successfully")
        
        # Action heads for multi-discrete actions
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_size, dim) for dim in self.action_dims
        ])
        
        # Value head
        self.value_head = nn.Linear(hidden_size, 1)
        
    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Encode observations to hidden representation.
        
        Args:
            observations: Raw observations
            state: Optional state dict
            
        Returns:
            Encoded observations
        """
        # Convert to float32 if needed (observations might be uint8)
        if observations.dtype != torch.float32:
            observations = observations.float() / 255.0  # Normalize to [0, 1] if uint8
        
        # Flatten observations if needed
        original_shape = observations.shape
        if len(original_shape) > 2:
            # Flatten all but batch dimension
            batch_size = original_shape[0]
            observations = observations.reshape(batch_size, -1)
        
        # Encode through MLP
        hidden = self.obs_encoder(observations)
        
        return hidden
    
    def decode_actions(self, hidden: torch.Tensor) -> tuple:
        """Decode hidden states to action logits and values.
        
        Args:
            hidden: Hidden state tensor
            
        Returns:
            logits: Concatenated action logits
            values: Value estimates
        """
        # Get action logits for each action dimension and concatenate
        logits_list = [head(hidden) for head in self.action_heads]
        logits = torch.cat(logits_list, dim=-1)
        
        # Get value estimates
        values = self.value_head(hidden).squeeze(-1)
        
        return logits, values
    
    def transformer(self, hidden: torch.Tensor, terminations: torch.Tensor = None, memory: dict = None):
        """Forward pass through transformer.
        
        This method is expected by TransformerWrapper.
        
        Args:
            hidden: Input tensor of shape (T, B, hidden_dim) where B = num_envs * num_agents
            terminations: Termination flags (unused for full-context)
            memory: Previous memory state (unused for full-context)
            
        Returns:
            output: Transformer output
            new_memory: Updated memory (None for full-context)
        """
        # Full-context transformer doesn't need memory
        # Memory could be None or empty dict - both are fine
        output = self._transformer(hidden)
        # Return None for memory since we don't use it
        return output, None
    
    def initialize_memory(self, batch_size: int) -> dict:
        """Initialize memory for the transformer.
        
        This method is expected by TransformerWrapper.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Empty dict (no memory needed for full-context transformer)
        """
        return {}


class FullContext(PyTorchAgentMixin, TransformerWrapper):
    """Full-context transformer agent.
    
    This agent uses a transformer with GTrXL-style stabilization that processes the entire
    BPTT trajectory at once, using all observations as context. Optimized for parallel
    processing across thousands of environments and agents.
    """
    
    def __init__(
        self,
        env,
        policy: Optional[nn.Module] = None,
        input_size: int = 256,
        hidden_size: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_gating: bool = True,
        **kwargs,
    ):
        """Initialize the full-context transformer agent.
        
        Args:
            env: Environment
            policy: Optional policy network
            input_size: Input embedding size
            hidden_size: Hidden dimension for transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length (should be >= BPTT horizon)
            dropout: Dropout rate
            use_causal_mask: Whether to use causal masking
            use_gating: Whether to use GRU gating
            **kwargs: Configuration parameters handled by mixin
        """
        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)
        
        # Log initialization
        logger.info("Initializing FullContext transformer agent...")
        
        # Log batch size information if available
        if hasattr(env, 'num_envs'):
            num_envs = getattr(env, 'num_envs', 1)
            num_agents = getattr(env, 'num_agents', 1)
            total_batch = num_envs * num_agents
            logger.info(f"Batch size: {total_batch} ({num_envs} envs × {num_agents} agents)")
        
        if policy is None:
            logger.info("Creating Policy network...")
            try:
                policy = Policy(
                    env,
                    input_size=input_size,
                    hidden_size=hidden_size,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    use_causal_mask=use_causal_mask,
                    use_gating=use_gating,
                )
                logger.info("Policy network created successfully")
            except Exception as e:
                logger.error(f"Failed to create Policy network: {e}")
                raise
        
        # Initialize transformer wrapper
        logger.info("Initializing TransformerWrapper...")
        super().__init__(env, policy, hidden_size)
        logger.info("TransformerWrapper initialized successfully")
        
        # Initialize mixin with configuration parameters
        logger.info("Initializing PyTorchAgentMixin...")
        self.init_mixin(**mixin_params)
        logger.info("FullContext agent initialization complete")
    
    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass through the agent.
        
        Processes observations from multiple environments and agents in parallel.
        
        Args:
            td: TensorDict with observations and other data
            state: Optional state dict
            action: Optional action tensor (for training)
            
        Returns:
            Updated TensorDict with predictions
        """
        observations = td["env_obs"]
        
        if state is None:
            state = {"transformer_memory": None, "hidden": None}
        
        # Determine dimensions from observations
        if observations.dim() == 4:  # Training (B, T, ...)
            B = observations.shape[0]
            TT = observations.shape[1]
            # Reshape TD for training if needed
            if td.batch_dims > 1:
                td = td.reshape(B * TT)
        else:  # Inference (B, ...)
            B = observations.shape[0]
            TT = 1
        
        # Set TensorDict fields with mixin
        self.set_tensordict_fields(td, observations)
        
        # Use parent class methods for forward pass
        if action is None:
            # Inference mode - use forward_eval from parent
            logits, values = super().forward_eval(observations, state)
            # Convert logits to list (mixin expects a list)
            logits_list = [logits]
            # Mixin handles inference mode
            td = self.forward_inference(td, logits_list, values)
        else:
            # Training mode - use forward from parent
            logits, values = super().forward(observations, state)
            # Convert logits to list (mixin expects a list)
            logits_list = [logits]
            # Mixin handles training mode with proper reshaping
            td = self.forward_training(
                td=td,
                action=action,
                logits_list=logits_list,
                value=values,
            )
        
        return td