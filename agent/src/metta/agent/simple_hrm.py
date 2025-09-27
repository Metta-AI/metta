
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from tensordict import TensorDict
from metta.agent.policy import Policy


class SimpleHRMMemory:
    """Minimal memory management for HRM."""
    
    def __init__(self):
        self.carry = {}
    
    def has_memory(self):
        return True
    
    def get_memory(self):
        return self.carry
    
    def set_memory(self, memory):
        self.carry = memory
    
    def reset_memory(self):
        self.carry = {}
    
    def reset_env_memory(self, env_id):
        if env_id in self.carry:
            del self.carry[env_id]


class SimpleAttention(nn.Module):
    """Simplified single-head attention."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = hidden_size ** -0.5
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v
        return self.proj(out)


class SimpleReasoningBlock(nn.Module):
    """Minimal reasoning block with attention and MLP."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = SimpleAttention(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleHRMCore(nn.Module):
    """Simplified HRM core with hierarchical reasoning."""
    
    def __init__(self, hidden_size: int = 64, max_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        
        # Observation encoder
        self.obs_encoder = nn.Linear(3, hidden_size)  # For (x, y, attr) tokens
        
        # Two-level reasoning
        self.low_level = SimpleReasoningBlock(hidden_size)
        self.high_level = SimpleReasoningBlock(hidden_size)
        
        # Halt predictor (simple binary classifier)
        self.halt_head = nn.Linear(hidden_size, 1)
        
        # Initial states
        self.register_buffer('h_init', torch.randn(hidden_size) * 0.1)
        self.register_buffer('l_init', torch.randn(hidden_size) * 0.1)
    
    def encode_observations(self, obs):
        """Encode observations to hidden representations."""
        # obs shape: [batch, seq_len, 3] where 3 = (x_coord, y_coord, attribute)
        return self.obs_encoder(obs.float())
    
    def forward(self, obs, h_state=None, l_state=None):
        """Forward pass with adaptive computation."""
        batch_size, seq_len = obs.shape[:2]
        device = obs.device
        
        # Initialize states if not provided
        if h_state is None:
            h_state = self.h_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        if l_state is None:
            l_state = self.l_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Encode input
        input_emb = self.encode_observations(obs)
        
        # Adaptive computation loop
        total_halt_prob = 0.0
        outputs = []
        
        for step in range(self.max_steps):
            # Low-level reasoning
            l_state = self.low_level(l_state + input_emb)
            
            # High-level reasoning
            h_state = self.high_level(h_state + l_state)
            
            # Compute halt probability
            halt_logits = self.halt_head(h_state.mean(dim=1))  # Pool over sequence
            halt_prob = torch.sigmoid(halt_logits)
            
            outputs.append({
                'hidden': h_state.mean(dim=1),  # Global representation
                'halt_prob': halt_prob,
                'step': step
            })
            
            total_halt_prob += halt_prob
            
            # Stop if we should halt (during inference)
            if not self.training and halt_prob.mean() > 0.5:
                break
        
        # Return final state and all intermediate outputs
        return {
            'h_state': h_state.detach(),
            'l_state': l_state.detach(),
            'outputs': outputs,
            'final_hidden': h_state.mean(dim=1)
        }


class SimpleHRM(Policy, SimpleHRMMemory):
    """Simplified HRM policy inheriting from Policy and SimpleHRMMemory."""
    
    def __init__(self, env, config=None):
        """Initialize SimpleHRM policy following Metta policy pattern."""
        super().__init__()
        SimpleHRMMemory.__init__(self)
        
        self.env = env
        self.config = config
        self.is_continuous = False
        
        # Store action space info
        self.action_space = getattr(env, "action_space", None)
        self.active_action_names = []
        self.num_active_actions = 10  # Default
        
        # Get environment dimensions
        self.out_width = getattr(env, "obs_width", 11)
        self.out_height = getattr(env, "obs_height", 11)
        self.num_layers = 25  # Default for mettagrid
        
        # Model parameters
        self.obs_dim = 200  # Default observation dimension
        self.hidden_size = 64
        
        # Create the internal policy components
        self.policy = SimpleHRMPolicyInner(env, hidden_size=self.hidden_size)
        
        # Set up device tracking
        self._device = torch.device("cpu")
    
    def initialize_to_environment(self, env_metadata, device: torch.device):
        """Initialize policy to environment - required by training framework."""
        # Extract action names from environment metadata
        action_names = getattr(env_metadata, "action_names", ["noop", "move", "attack"])
        
        # Initialize action embeddings in the policy
        self.policy.initialize_to_environment(action_names, device)
        
        # Store environment metadata
        self.env_metadata = env_metadata
        self._device = device
        
        # Move model to device
        self.to(device)
    
    @property
    def device(self) -> torch.device:
        """Device property required by the Policy interface."""
        return getattr(self, "_device", torch.device("cpu"))
    
    def reset_memory(self):
        """Reset memory state - required by Policy interface."""
        SimpleHRMMemory.reset_memory(self)
    
    def get_agent_experience_spec(self):
        """Get experience spec - required by Policy interface."""
        from torchrl.data import Composite, UnboundedDiscrete
        
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )
    
    @torch._dynamo.disable
    def forward(self, td: TensorDict, state=None, action=None):
        """Forward pass following Metta policy pattern."""
        observations = td["env_obs"]
        
        # Encode observations using SimpleHRM core
        hidden = self.policy.encode_observations(observations, td, memory_manager=self)
        
        # Decode actions and values
        logits, value = self.policy.decode_actions(hidden.to(torch.float32))
        
        # Set outputs in TensorDict
        td["logits"] = logits
        td["values"] = value.flatten()
        
        return td


class SimpleHRMPolicyInner(nn.Module):
    """Inner policy component matching the original HRMPolicyInner interface."""
    
    def __init__(self, env, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        
        # Handle both EnvironmentMetaData and legacy env objects
        if hasattr(env, "single_action_space"):
            self.action_space = env.single_action_space
        else:
            # Default action space for EnvironmentMetaData
            from gymnasium.spaces import MultiDiscrete
            self.action_space = MultiDiscrete([9, 10])
        
        self.out_width = getattr(env, "obs_width", 11)
        self.out_height = getattr(env, "obs_height", 11)
        self.num_layers = 25
        
        # SimpleHRM Core
        self.core = SimpleHRMCore(hidden_size)
        
        # Policy heads (matching original interface)
        self.critic_1 = nn.Linear(hidden_size, 128)
        self.value_head = nn.Linear(128, 1)
        
        self.actor_1 = nn.Linear(hidden_size, 64)
        
        # Action embeddings
        self.action_embeddings = nn.Embedding(100, 16)
        self._initialize_action_embeddings()
        
        # Bilinear layer
        self.actor_W = nn.Parameter(torch.Tensor(1, 64, 16))
        self.actor_bias = nn.Parameter(torch.Tensor(1))
        self._init_bilinear_actor()
        
        # Track active actions
        self.active_action_names = []
        self.num_active_actions = 100
    
    def _initialize_action_embeddings(self):
        """Initialize action embeddings."""
        nn.init.orthogonal_(self.action_embeddings.weight)
        with torch.no_grad():
            max_abs_value = torch.max(torch.abs(self.action_embeddings.weight))
            self.action_embeddings.weight.mul_(0.1 / max_abs_value)
    
    def _init_bilinear_actor(self):
        """Initialize bilinear actor head."""
        import math
        bound = 1 / math.sqrt(64)
        nn.init.uniform_(self.actor_W, -bound, bound)
        nn.init.uniform_(self.actor_bias, -bound, bound)
    
    def initialize_to_environment(self, full_action_names: list[str], device):
        """Initialize to environment."""
        self.active_action_names = full_action_names
        self.num_active_actions = len(full_action_names)
    
    def encode_observations(self, observations, td=None, memory_manager=None):
        """Encode observations using SimpleHRM core."""
        if len(observations.shape) == 4:
            observations = observations.reshape(-1, 200, 3)
        
        # Get environment ID
        if td is not None:
            training_env_id_start = td.get("training_env_id_start", None)
            if training_env_id_start is not None:
                env_id = training_env_id_start[0].item()
            else:
                env_id = 0
        else:
            env_id = 0
        
        # Get previous states from memory
        if memory_manager is not None:
            prev_memory = memory_manager.get_memory()
        else:
            prev_memory = {}
        
        if prev_memory is None or f"{env_id}" not in prev_memory:
            h_state, l_state = None, None
        else:
            h_state, l_state = prev_memory[f"{env_id}"]
        
        # Forward through core
        result = self.core(observations, h_state, l_state)
        
        # Update memory
        if memory_manager is not None:
            current_memory = memory_manager.get_memory() or {}
            current_memory[f"{env_id}"] = (result['h_state'], result['l_state'])
            memory_manager.set_memory(current_memory)
        
        return result['final_hidden']
    
    def decode_actions(self, hidden):
        """Decode actions using bilinear interaction."""
        # Critic branch
        critic_features = torch.tanh(self.critic_1(hidden))
        value = self.value_head(critic_features)
        
        # Actor branch
        actor_features = F.relu(self.actor_1(hidden))
        actual_batch_size = hidden.shape[0]
        
        # Get action embeddings
        action_embeds = self.action_embeddings.weight[:self.num_active_actions]
        action_embeds = action_embeds.unsqueeze(0).expand(actual_batch_size, -1, -1)
        
        # Bilinear interaction
        num_actions = action_embeds.shape[1]
        actor_repeated = actor_features.unsqueeze(1).expand(-1, num_actions, -1)
        
        # Compute scores using bilinear operation
        query = torch.einsum('bna,kae->bne', actor_repeated, self.actor_W)
        query = torch.tanh(query)
        scores = torch.einsum('bne,bne->bn', query, action_embeds)
        logits = scores + self.actor_bias
        
        return logits, value

