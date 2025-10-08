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
    Mmax: int = 10  # Maximum number of reasoning segments

    def make_component(self, env=None) -> "HRMReasoning":
        return HRMReasoning(self)


class HRMReasoning(nn.Module):
    """HRM reasoning component matching official implementation."""

    def __init__(self, config: HRMReasoningConfig):
        super().__init__()
        self.config = config

        self.memory = HRMMemory()
        self.carry = self.memory.get_memory()

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

        # Initialize Q-head for fast learning (official implementation)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)

        # Initial states (truncated normal like official implementation)
        self.register_buffer(
            "H_init", trunc_normal_init_(torch.empty(config.embed_dim), std=1.0, a=-2.0, b=2.0), persistent=True
        )
        self.register_buffer(
            "L_init", trunc_normal_init_(torch.empty(config.embed_dim), std=1.0, a=-2.0, b=2.0), persistent=True
        )

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

        # Unsqueeze to add sequence dimension for transformer: (B, D) -> (B, 1, D)
        if x.ndim == 2:
            x = x.unsqueeze(1)

        # Get reset mask
        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is not None and truncateds is not None:
            reset_mask = (dones.bool() | truncateds.bool()).view(-1, 1)
        else:
            reset_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # Get environment IDs
        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            training_env_ids = torch.arange(batch_size, device=device)
        else:
            training_env_ids = training_env_ids.reshape(batch_size)

        # Allocate/expand memory
        max_num_envs = int(training_env_ids.max().item()) + 1
        if "z_l" not in self.carry or self.carry["z_l"].shape[0] < max_num_envs:
            num_new = max_num_envs - (self.carry["z_l"].shape[0] if "z_l" in self.carry else 0)
            # Initialize with same init values as buffers
            z_l_new = self.L_init.unsqueeze(0).expand(num_new, -1).clone()
            z_h_new = self.H_init.unsqueeze(0).expand(num_new, -1).clone()

            if "z_l" in self.carry:
                self.carry["z_l"] = torch.cat([self.carry["z_l"], z_l_new], dim=0).to(device)
                self.carry["z_h"] = torch.cat([self.carry["z_h"], z_h_new], dim=0).to(device)
            else:
                self.carry["z_l"] = z_l_new
                self.carry["z_h"] = z_h_new

        # Retrieve and reset states
        z_l = self.carry["z_l"][training_env_ids].clone()
        z_h = self.carry["z_h"][training_env_ids].clone()

        # Reset on episode end (use init values)
        z_l = torch.where(reset_mask, self.L_init.unsqueeze(0), z_l)
        z_h = torch.where(reset_mask, self.H_init.unsqueeze(0), z_h)

        # Add sequence dimension: (B, D) -> (B, 1, D)
        z_l = z_l.unsqueeze(1)
        z_h = z_h.unsqueeze(1)

        # Main HRM forward pass (matching official implementation)
        with torch.no_grad():
            for H_step in range(self.config.H_cycles):
                for L_step in range(self.config.L_cycles):
                    # Skip last iteration (will be done with gradients)
                    if not ((H_step == self.config.H_cycles - 1) and (L_step == self.config.L_cycles - 1)):
                        z_l = self.L_level(z_l, z_h + x)

                # Skip last H update (will be done with gradients)
                if not (H_step == self.config.H_cycles - 1):
                    z_h = self.H_level(z_h, z_l)

        # Final 1-step with gradients (for learning)
        z_l = self.L_level(z_l, z_h + x)
        z_h = self.H_level(z_h, z_l)

        # Q-head for halting decision (use first token)
        q_logits = self.q_head(z_h[:, 0])  # (batch, 2)

        # Store Q-logits in tensordict for ACT loss computation
        td["q_halt_logits"] = q_logits[:, 0]  # (batch,)
        td["q_continue_logits"] = q_logits[:, 1]  # (batch,)

        # Store hidden states (squeeze seq dim and detach)
        self.carry["z_l"][training_env_ids] = z_l.squeeze(1).detach()
        self.carry["z_h"][training_env_ids] = z_h.squeeze(1).detach()

        # Output final state (squeeze seq dimension)
        td[self.config.out_key] = z_h.squeeze(1)
        return td
