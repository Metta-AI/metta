import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


def rms_norm(x: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + variance_epsilon)


def trunc_normal_init_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    """Truncated normal initialization."""
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(a, b)
    return tensor


class SwiGLU(nn.Module):
    """SwiGLU activation from official implementation."""

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        ffn_hidden = int(hidden_size * expansion)
        self.w1 = nn.Linear(hidden_size, ffn_hidden, bias=False)
        self.w2 = nn.Linear(ffn_hidden, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class HRMBlock(nn.Module):
    """Transformer block matching official HRM implementation (post-norm, RMSNorm, SwiGLU)."""

    def __init__(self, hidden_size: int, num_heads: int, expansion: float, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, bias=False)
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = rms_norm_eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm: x + sublayer, then norm
        # Self-attention
        attn_out, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)
        # MLP
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HRMReasoningModule(nn.Module):
    """Reasoning module with input injection, matching official implementation."""

    def __init__(self, layers: list[HRMBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor) -> torch.Tensor:
        # Input injection (add input to hidden state)
        hidden_states = hidden_states + input_injection

        # Apply layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class HRMReasoningConfig(ComponentConfig):
    """HRM reasoning component config."""

    class_path: str = "metta.agent.components.hrm.HRMReasoning"
    in_key: str = "hrm_obs_encoded"
    out_key: str = "hrm_reasoning"
    name: str = "hrm_reasoning"
    embed_dim: int = 256
    num_layers: int = 4  # Number of transformer blocks per module
    num_heads: int = 8
    ffn_expansion: float = 4.0  # SwiGLU expansion factor
    rms_norm_eps: float = 1e-5
    # ACT parameters
    H_cycles: int = 3  # Number of high-level reasoning cycles
    L_cycles: int = 5  # Number of low-level processing cycles per H-cycle
    Mmin: int = 1  # Minimum number of reasoning segments
    Mmax: int = 4  # Maximum number of reasoning segments
    # Monitoring
    track_gradients: bool = True  # Enable gradient tracking for debugging

    def make_component(self, env=None) -> "HRMReasoning":
        return HRMReasoning(self)


class HRMReasoning(nn.Module):
    """HRM reasoning component matching official implementation."""

    def __init__(self, config: HRMReasoningConfig):
        super().__init__()
        self.config = config

        # Initialize memory dictionary (env_id -> {z_l, z_h})
        self.carry: dict[int, dict[str, torch.Tensor]] = {}

        # Gradient tracking storage
        self._grad_norms: dict[str, list[float]] = {}

        # Reasoning modules (H and L levels)
        self.H_level = HRMReasoningModule(
            layers=[
                HRMBlock(config.embed_dim, config.num_heads, config.ffn_expansion, config.rms_norm_eps)
                for _ in range(config.num_layers)
            ]
        )
        self.L_level = HRMReasoningModule(
            layers=[
                HRMBlock(config.embed_dim, config.num_heads, config.ffn_expansion, config.rms_norm_eps)
                for _ in range(config.num_layers)
            ]
        )

        # Q-head for ACT halting (2 outputs: halt, continue)
        self.q_head = nn.Linear(config.embed_dim, 2, bias=True)

        # Initialize Q-head with less conservative bias for better exploration
        # Original paper used -5.0, but this prevents multi-step reasoning
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-1.0)  # Reduced from -5.0 to allow more reasoning steps

        # Initial states (truncated normal like official implementation)
        self.register_buffer(
            "H_init", trunc_normal_init_(torch.empty(config.embed_dim), std=1.0, a=-2.0, b=2.0), persistent=True
        )
        self.register_buffer(
            "L_init", trunc_normal_init_(torch.empty(config.embed_dim), std=1.0, a=-2.0, b=2.0), persistent=True
        )

        # Register gradient hooks for all layers if tracking enabled
        if self.config.track_gradients:
            self._register_gradient_hooks()

    def __setstate__(self, state):
        """Ensure HRM hidden states are properly initialized after loading from checkpoint."""
        self.__dict__.update(state)
        # Reset hidden states when loading from checkpoint to avoid batch size mismatch
        if not hasattr(self, "carry"):
            self.carry = {}
        # Clear any existing states to handle batch size mismatches
        self.carry.clear()
        # Reset gradient tracking
        if not hasattr(self, "_grad_norms"):
            self._grad_norms = {}

    def _register_gradient_hooks(self):
        """Register gradient hooks on all layers to track gradient norms."""
        # Track H-level layers
        for i, layer in enumerate(self.H_level.layers):
            layer.self_attn.out_proj.weight.register_hook(
                lambda grad, idx=i: self._track_grad_norm(f"H_level.layer{idx}.attn.out_proj", grad)
            )
            layer.mlp.w2.weight.register_hook(
                lambda grad, idx=i: self._track_grad_norm(f"H_level.layer{idx}.mlp.w2", grad)
            )

        # Track L-level layers
        for i, layer in enumerate(self.L_level.layers):
            layer.self_attn.out_proj.weight.register_hook(
                lambda grad, idx=i: self._track_grad_norm(f"L_level.layer{idx}.attn.out_proj", grad)
            )
            layer.mlp.w2.weight.register_hook(
                lambda grad, idx=i: self._track_grad_norm(f"L_level.layer{idx}.mlp.w2", grad)
            )

        # Track Q-head
        self.q_head.weight.register_hook(lambda grad: self._track_grad_norm("q_head.weight", grad))

    def _track_grad_norm(self, name: str, grad: torch.Tensor) -> None:
        """Track gradient norm for a specific layer."""
        if grad is not None:
            grad_norm = grad.norm().item()
            if name not in self._grad_norms:
                self._grad_norms[name] = []
            self._grad_norms[name].append(grad_norm)

    def get_grad_norms(self) -> dict[str, float]:
        """Get average gradient norms since last call and reset."""
        result = {}
        for name, norms in self._grad_norms.items():
            if norms:
                result[f"grad_norm/{name}"] = sum(norms) / len(norms)

        # Debug: Always return at least one metric to verify this is being called
        if not result:
            result["grad_norm/debug_called"] = 1.0

        self._grad_norms.clear()
        return result

    @torch._dynamo.disable  # Exclude HRM forward from Dynamo to avoid graph breaks with stateful memory
    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.config.in_key]

        # Handle input shapes - flatten to (batch, embed_dim)
        if x.ndim == 4:
            # Spatial: (B, H, W, D) -> flatten and average
            batch_size = x.shape[0]
            x = x.view(batch_size, -1, x.shape[-1]).mean(dim=1)
        elif x.ndim == 3:
            # Sequence: (B, seq, D) -> average over sequence
            x = x.mean(dim=1)
        # else: x.ndim == 2, already (batch, embed_dim)

        batch_size = x.shape[0]
        device = x.device

        # Track gradient statistics if enabled (stored in _grad_norms, not TensorDict)
        if self.config.track_gradients and x.requires_grad:

            def input_grad_hook(grad):
                if grad is not None:
                    if "input" not in self._grad_norms:
                        self._grad_norms["input"] = []
                    self._grad_norms["input"].append(grad.norm().item())

            x.register_hook(input_grad_hook)

        # Unsqueeze to add sequence dimension for transformer: (B, D) -> (B, 1, D)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Get environment ID (single ID per batch, like LSTM)
        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is not None:
            flat_env_ids = training_env_ids.reshape(-1)
        else:
            flat_env_ids = torch.arange(batch_size, device=device)

        training_env_id_start = int(flat_env_ids[0].item()) if flat_env_ids.numel() else 0

        # Retrieve or initialize hidden states for this environment
        if training_env_id_start in self.carry:
            z_l_stored = self.carry[training_env_id_start]["z_l"]
            z_h_stored = self.carry[training_env_id_start]["z_h"]

            # Check if batch size matches - if not, reinitialize
            if z_l_stored.shape[0] == batch_size:
                z_l = z_l_stored.clone()
                z_h = z_h_stored.clone()

                # Reset the hidden state if the episode is done or truncated
                dones = td.get("dones", None)
                truncateds = td.get("truncateds", None)
                if dones is not None and truncateds is not None:
                    reset_mask = (dones.bool() | truncateds.bool()).view(-1, 1)
                    z_l = z_l.masked_fill(reset_mask, 0)
                    z_h = z_h.masked_fill(reset_mask, 0)
                    # Re-initialize with L_init and H_init for reset envs
                    z_l = torch.where(reset_mask, self.L_init.unsqueeze(0).expand(batch_size, -1), z_l)
                    z_h = torch.where(reset_mask, self.H_init.unsqueeze(0).expand(batch_size, -1), z_h)
            else:
                # Batch size mismatch - reinitialize
                z_l = self.L_init.unsqueeze(0).expand(batch_size, -1).clone()
                z_h = self.H_init.unsqueeze(0).expand(batch_size, -1).clone()
        else:
            # Initialize new environment with init values
            z_l = self.L_init.unsqueeze(0).expand(batch_size, -1).clone()
            z_h = self.H_init.unsqueeze(0).expand(batch_size, -1).clone()

        # Add sequence dimension: (B, D) -> (B, 1, D)
        z_l = z_l.unsqueeze(1)
        z_h = z_h.unsqueeze(1)

        # Track number of segments used
        num_segments = 0

        for m_step in range(self.config.Mmax):
            num_segments = m_step + 1

            # Run H_cycles - 1 iterations without gradients (exploration)
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        z_l = self.L_level(z_l, z_h + x)
                    z_h = self.H_level(z_h, z_l)

            # Final iteration WITH gradients (one-step approximation)
            for _ in range(self.config.L_cycles):
                z_l = self.L_level(z_l, z_h + x)
            z_h = self.H_level(z_h, z_l)

            # Track gradients on z_h if enabled (stored in _grad_norms, not TensorDict)
            if self.config.track_gradients and z_h.requires_grad:

                def grad_hook(grad, step=m_step):
                    if grad is not None:
                        grad_norm = grad.norm().item()
                        key = f"z_h_m{step}"
                        if key not in self._grad_norms:
                            self._grad_norms[key] = []
                        self._grad_norms[key].append(grad_norm)

                z_h.register_hook(grad_hook)

        # Store hidden states (squeeze seq dim and detach)
        self.carry[training_env_id_start] = {
            "z_l": z_l.squeeze(1).detach(),
            "z_h": z_h.squeeze(1).detach(),
        }

        # Output final state (squeeze seq dimension)
        td[self.config.out_key] = z_h.squeeze(1)
        return td
