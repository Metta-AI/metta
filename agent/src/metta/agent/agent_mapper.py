"""
Unified agent mapper for both PyTorch and ComponentPolicy implementations.

This module provides a centralized mapping of agent names to their implementations,
supporting both vanilla PyTorch models and ComponentPolicy architectures.
"""

# PyTorch implementations (vanilla models)
# ComponentPolicy implementations (modular architecture)
from metta.agent.component_policies.fast import Fast as ComponentFast
from metta.agent.component_policies.latent_attn_med import LatentAttnMed as ComponentLatentAttnMed
from metta.agent.component_policies.latent_attn_small import LatentAttnSmall as ComponentLatentAttnSmall
from metta.agent.component_policies.latent_attn_tiny import LatentAttnTiny as ComponentLatentAttnTiny
from metta.agent.pytorch.example import Example
from metta.agent.pytorch.fast import Fast as PyTorchFast
from metta.agent.pytorch.latent_attn_med import LatentAttnMed as PyTorchLatentAttnMed
from metta.agent.pytorch.latent_attn_small import LatentAttnSmall as PyTorchLatentAttnSmall
from metta.agent.pytorch.latent_attn_tiny import LatentAttnTiny as PyTorchLatentAttnTiny

# Unified agent mapping - includes both PyTorch and ComponentPolicy implementations
agents = {
    # ComponentPolicy implementations (default, no prefix)
    "fast": ComponentFast,
    "latent_attn_med": ComponentLatentAttnMed,
    "latent_attn_small": ComponentLatentAttnSmall,
    "latent_attn_tiny": ComponentLatentAttnTiny,
    # PyTorch implementations (with explicit prefix)
    "pytorch/example": Example,
    "pytorch/fast": PyTorchFast,
    "pytorch/latent_attn_small": PyTorchLatentAttnSmall,
    "pytorch/latent_attn_med": PyTorchLatentAttnMed,
    "pytorch/latent_attn_tiny": PyTorchLatentAttnTiny,
}
