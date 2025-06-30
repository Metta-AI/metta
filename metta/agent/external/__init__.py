# Import external policy modules for easier access
from . import example, lstm_transformer, torch
from .pytorch_adapter import PytorchAdapter, load_pytorch_policy

# ExternalPolicyAdapter is kept as an alias for backwards compatibility
ExternalPolicyAdapter = PytorchAdapter

__all__ = ["PytorchAdapter", "ExternalPolicyAdapter", "load_pytorch_policy", "torch", "lstm_transformer", "example"]
