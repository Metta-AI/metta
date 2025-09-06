"""Muesli agent implementation with MuZero-inspired architecture."""

import copy
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from metta.rl.muesli.categorical import scalar_to_support, support_to_scalar
from metta.rl.muesli.config import MuesliConfig


class ResidualBlock(nn.Module):
    """Residual block for the representation network."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class MuesliAgent(nn.Module):
    """Muesli agent with representation, dynamics, and prediction networks.
    
    Architecture:
    1. Representation network: Encodes observations into hidden states
    2. Dynamics network: Predicts next hidden state and reward given action
    3. Prediction networks: Output policy and value from hidden states
    4. Target network: EMA copy for stable value targets
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_space: Any,
        config: MuesliConfig,
        device: torch.device
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Get dimensions
        self.obs_channels = obs_shape[0] if len(obs_shape) == 3 else 1
        self.obs_height = obs_shape[-2] if len(obs_shape) >= 2 else obs_shape[0]
        self.obs_width = obs_shape[-1] if len(obs_shape) >= 2 else 1
        
        # Action space
        if hasattr(action_space, 'n'):
            self.num_actions = action_space.n
            self.discrete_actions = True
        else:
            self.num_actions = action_space.shape[0]
            self.discrete_actions = False
            
        # Network dimensions
        self.hidden_size = config.network.hidden_size
        self.conv_channels = config.network.conv_channels
        self.dynamics_hidden = config.network.dynamics_hidden_size
        self.support_size = config.categorical.support_size
        self.num_lstm_layers = config.network.num_lstm_layers
        
        # Build networks
        self._build_representation_network()
        self._build_dynamics_network()
        self._build_prediction_networks()
        
        # Initialize target network
        self.target_network = None
        self._init_target_network()
        
        # Running variance estimate for advantage normalization
        self.register_buffer('variance_estimate', torch.tensor(1.0))
        
        # Apply initialization
        self._initialize_weights()
        
    def _build_representation_network(self):
        """Build the representation network (observation -> hidden state)."""
        # For image observations, use convolutional layers
        if self.obs_height > 1 and self.obs_width > 1:
            # Convolutional stem
            self.conv_stem = nn.Sequential(
                nn.Conv2d(self.obs_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, self.conv_channels, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            
            # Calculate conv output size
            def conv_out_size(size, kernel, stride):
                return (size - kernel) // stride + 1
            
            h = conv_out_size(self.obs_height, 8, 4)
            h = conv_out_size(h, 4, 2)
            h = conv_out_size(h, 3, 1)
            
            w = conv_out_size(self.obs_width, 8, 4)
            w = conv_out_size(w, 4, 2)
            w = conv_out_size(w, 3, 1)
            
            conv_out_dim = h * w * self.conv_channels
            
            # Fully connected layers
            self.representation_fc = nn.Sequential(
                nn.Linear(conv_out_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
            )
        else:
            # For vector observations, use only fully connected layers
            input_dim = self.obs_height * self.obs_width * self.obs_channels
            self.conv_stem = None
            self.representation_fc = nn.Sequential(
                nn.Linear(input_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
            )
        
        # LSTM for temporal dependencies
        self.representation_lstm = nn.LSTM(
            self.hidden_size, 
            self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )
        
    def _build_dynamics_network(self):
        """Build the dynamics network (hidden, action -> next_hidden, reward)."""
        # Input: concatenated hidden state and action
        if self.discrete_actions:
            # One-hot encode discrete actions
            dynamics_input_dim = self.hidden_size + self.num_actions
        else:
            dynamics_input_dim = self.hidden_size + self.num_actions
            
        # LSTM for dynamics
        self.dynamics_lstm = nn.LSTM(
            dynamics_input_dim,
            self.dynamics_hidden,
            num_layers=self.num_lstm_layers,
            batch_first=True
        )
        
        # Projection to next hidden state
        self.dynamics_projection = nn.Sequential(
            nn.Linear(self.dynamics_hidden, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        # Reward prediction head (categorical)
        self.reward_head = nn.Sequential(
            nn.Linear(self.dynamics_hidden, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.support_size)
        )
        
    def _build_prediction_networks(self):
        """Build the prediction networks (hidden -> policy, value)."""
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_actions)
        )
        
        # Value head (categorical)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.support_size)
        )
        
    def _init_target_network(self):
        """Initialize target network as a copy of the main network."""
        self.target_network = copy.deepcopy(self)
        for param in self.target_network.parameters():
            param.requires_grad = False
            
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
                        
        # Small initialization for policy head to encourage exploration
        if hasattr(self, 'policy_head'):
            with torch.no_grad():
                self.policy_head[-1].weight.mul_(self.config.network.policy_init_gain)
                
    def representation(self, obs: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Encode observation into hidden state.
        
        Args:
            obs: Observation tensor
            
        Returns:
            hidden_state: Encoded representation
            lstm_state: LSTM hidden and cell states
        """
        batch_size = obs.shape[0]
        
        # Process through conv layers if applicable
        if self.conv_stem is not None:
            x = self.conv_stem(obs)
            x = x.flatten(1)
        else:
            x = obs.flatten(1)
            
        # Process through FC layers
        x = self.representation_fc(x)
        
        # Add sequence dimension for LSTM
        x = x.unsqueeze(1)
        
        # Initialize LSTM state
        h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_size, device=self.device)
        
        # Process through LSTM
        x, (hn, cn) = self.representation_lstm(x, (h0, c0))
        
        # Remove sequence dimension
        hidden_state = x.squeeze(1)
        
        return hidden_state, (hn, cn)
        
    def dynamics(
        self, 
        hidden_state: Tensor, 
        action: Tensor,
        lstm_state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """Predict next hidden state and reward given current state and action.
        
        Args:
            hidden_state: Current hidden state
            action: Action to take
            lstm_state: Optional LSTM state
            
        Returns:
            next_hidden: Predicted next hidden state
            reward_logits: Predicted reward (categorical logits)
            next_lstm_state: Updated LSTM state
        """
        batch_size = hidden_state.shape[0]
        
        # Prepare action
        if self.discrete_actions:
            # One-hot encode discrete actions
            action_one_hot = F.one_hot(action.long(), num_classes=self.num_actions).float()
            dynamics_input = torch.cat([hidden_state, action_one_hot], dim=-1)
        else:
            dynamics_input = torch.cat([hidden_state, action], dim=-1)
            
        # Add sequence dimension
        dynamics_input = dynamics_input.unsqueeze(1)
        
        # Initialize LSTM state if not provided
        if lstm_state is None:
            h0 = torch.zeros(self.num_lstm_layers, batch_size, self.dynamics_hidden, device=self.device)
            c0 = torch.zeros(self.num_lstm_layers, batch_size, self.dynamics_hidden, device=self.device)
            lstm_state = (h0, c0)
            
        # Process through dynamics LSTM
        dynamics_out, next_lstm_state = self.dynamics_lstm(dynamics_input, lstm_state)
        
        # Remove sequence dimension
        dynamics_out = dynamics_out.squeeze(1)
        
        # Predict next hidden state
        next_hidden = self.dynamics_projection(dynamics_out)
        
        # Predict reward
        reward_logits = self.reward_head(dynamics_out)
        
        return next_hidden, reward_logits, next_lstm_state
        
    def prediction(self, hidden_state: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict policy and value from hidden state.
        
        Args:
            hidden_state: Hidden state representation
            
        Returns:
            policy_logits: Action logits
            value_logits: Value prediction (categorical logits)
        """
        policy_logits = self.policy_head(hidden_state)
        value_logits = self.value_head(hidden_state)
        
        return policy_logits, value_logits
        
    def forward(
        self, 
        obs: Tensor,
        lstm_state: Optional[Tuple[Tensor, Tensor]] = None,
        action: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Forward pass through the network.
        
        Args:
            obs: Observation tensor
            lstm_state: Optional LSTM state
            action: Optional action for computing log probabilities
            
        Returns:
            Dictionary containing:
                - policy_logits: Action logits
                - value: Scalar value prediction
                - value_logits: Categorical value logits
                - action: Sampled action (if action not provided)
                - log_prob: Log probability of action
                - entropy: Policy entropy
        """
        # Encode observation
        hidden_state, lstm_state = self.representation(obs)
        
        # Get predictions
        policy_logits, value_logits = self.prediction(hidden_state)
        
        # Convert categorical value to scalar
        value_probs = F.softmax(value_logits, dim=-1)
        value = support_to_scalar(
            value_probs,
            self.config.categorical.support_size,
            self.config.categorical.value_min,
            self.config.categorical.value_max
        )
        
        # Compute policy distribution
        if self.discrete_actions:
            policy_dist = torch.distributions.Categorical(logits=policy_logits)
        else:
            # For continuous actions, assume Gaussian with learned std
            mean = policy_logits
            std = torch.ones_like(mean) * 0.5  # Fixed std for now
            policy_dist = torch.distributions.Normal(mean, std)
            
        # Sample action if not provided
        if action is None:
            action = policy_dist.sample()
            
        # Compute log probability and entropy
        log_prob = policy_dist.log_prob(action)
        entropy = policy_dist.entropy()
        
        return {
            'policy_logits': policy_logits,
            'value': value,
            'value_logits': value_logits,
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'lstm_state': lstm_state,
            'hidden_state': hidden_state
        }
        
    def update_target_network(self, tau: float = 0.1):
        """Update target network using exponential moving average.
        
        Args:
            tau: EMA update rate (0 = no update, 1 = full copy)
        """
        with torch.no_grad():
            for param, target_param in zip(
                self.parameters(), 
                self.target_network.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
                
    def update_variance_estimate(self, advantages: Tensor, decay: float = 0.99):
        """Update running variance estimate for advantage normalization.
        
        Args:
            advantages: Batch of advantages
            decay: Decay rate for running average
        """
        with torch.no_grad():
            batch_var = advantages.var()
            self.variance_estimate = decay * self.variance_estimate + (1 - decay) * batch_var