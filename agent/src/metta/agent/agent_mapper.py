"""
Unified agent mapper for both PyTorch and ComponentPolicy implementations.

This module provides a centralized mapping of agent names to their implementations,
supporting both vanilla PyTorch models and ComponentPolicy architectures.
"""

# PyTorch implementations (vanilla models)
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast as PyTorchFast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed as PyTorchLatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall as PyTorchLatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny as PyTorchLatentAttnTiny

# ComponentPolicy implementations (modular architecture)
from metta.agent.component_policies.fast import Fast as ComponentFast
from metta.agent.component_policies.latent_attn_med import LatentAttnMed as ComponentLatentAttnMed
from metta.agent.component_policies.latent_attn_small import LatentAttnSmall as ComponentLatentAttnSmall
from metta.agent.component_policies.latent_attn_tiny import LatentAttnTiny as ComponentLatentAttnTiny

# PyTorch models - vanilla implementations
pytorch_agents = {
    "example": Example,
    "fast": PyTorchFast,
    "latent_attn_small": PyTorchLatentAttnSmall,
    "latent_attn_med": PyTorchLatentAttnMed,
    "latent_attn_tiny": PyTorchLatentAttnTiny,
}

# ComponentPolicy models - modular implementations (default)
component_agents = {
    "fast": ComponentFast,
    "latent_attn_med": ComponentLatentAttnMed,
    "latent_attn_small": ComponentLatentAttnSmall,
    "latent_attn_tiny": ComponentLatentAttnTiny,
}

# Combined mapping for backward compatibility
# ComponentPolicies take precedence as they are the preferred implementation
all_agents = {
    **pytorch_agents,  # Add PyTorch agents first
    **component_agents,  # ComponentPolicies override PyTorch ones with same name
}