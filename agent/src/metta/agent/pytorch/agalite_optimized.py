"""
Optimized AGaLiTe variant - a beefed up version of fast mode.
Uses FastAGaLiTeLayer architecture with enhanced parameters for better metrics.
"""

import logging
from typing import Dict, Tuple

import torch
from torch import nn

from metta.agent.pytorch.agalite import AGaLiTe, AGaLiTeCore, AGaLiTePolicy

logger = logging.getLogger(__name__)

# Import fast implementation to verify it's available
try:
    from metta.agent.modules.agalite_fast import FastAGaLiTeLayer

    FAST_MODE_AVAILABLE = True
except ImportError as e:
    FAST_MODE_AVAILABLE = False
    raise ImportError("FastAGaLiTeLayer required for AGaLiTeOptimized") from e


class AGaLiTeOptimizedCore(AGaLiTeCore):
    """
    Optimized AGaLiTe core that uses FastAGaLiTeLayer with custom eta/r.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_head: int,
        d_ffc: int,
        n_heads: int,
        eta: int,
        r: int,
        reset_on_terminate: bool = True,
        dropout: float = 0.0,
        use_fast_mode: bool = True,  # Always true for optimized
    ):
        # Initialize nn.Module directly, skipping AGaLiTeCore's init
        nn.Module.__init__(self)

        self.n_layers = n_layers
        self.d_model = d_model
        self.d_head = d_head
        self.d_ffc = d_ffc
        self.n_heads = n_heads
        self.use_fast_mode = True

        # Use the provided eta/r without reduction
        self.eta = eta
        self.r = r

        # Build encoders with FastAGaLiTeLayer
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            encoder = FastAGaLiTeLayer(
                d_model=d_model,
                head_num=n_heads,
                head_dim=d_head,
                eta=eta,
                r=r,
                reset_hidden_on_terminate=reset_on_terminate,
                dropout=dropout,
            )
            self.encoders.append(encoder)

        logger.info(f"AGaLiTeOptimizedCore using FastAGaLiTeLayer with eta={eta}, r={r}")

    def initialize_memory(self, batch_size: int, device: torch.device = None) -> Dict[str, Tuple]:
        """Initialize memory with custom eta/r values."""
        memory_dict = {}
        for layer in range(1, self.n_layers + 1):
            memory_dict[f"layer_{layer}"] = FastAGaLiTeLayer.initialize_memory(
                batch_size, self.n_heads, self.d_head, self.eta, self.r, device
            )
        return memory_dict


class AGaLiTeOptimizedPolicy(AGaLiTePolicy):
    """
    Optimized AGaLiTe policy that uses AGaLiTeOptimizedCore.
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 2,
        eta: int = 3,
        r: int = 6,
        reset_on_terminate: bool = True,
        dropout: float = 0.05,
        use_fast_mode: bool = True,  # Always true for optimized
    ):
        # Initialize parent class
        super().__init__(
            env=env,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            n_layers=n_layers,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
            use_fast_mode=False,  # We'll replace the transformer
        )

        # Store the actual eta/r we're using
        self.eta = eta
        self.r = r

        # Replace the transformer with our optimized version
        self.transformer = AGaLiTeOptimizedCore(
            n_layers=n_layers,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
            use_fast_mode=True,
        )

        # Apply standard initialization matching working implementations
        # This is handled by the base classes already, so we don't need
        # to override unless we encounter stability issues

    def initialize_memory(self, batch_size: int) -> Dict:
        """Initialize memory with custom eta/r values."""
        device = next(self.parameters()).device
        return self.transformer.initialize_memory(batch_size, device)


class AGaLiTeOptimized(AGaLiTe):
    """
    Optimized AGaLiTe - a beefed up version of fast mode.

    Uses FastAGaLiTeLayer architecture (for speed) with:
    - Higher eta/r for better representation capacity
    - Configurable layers for better learning
    - Small dropout for generalization

    Expected performance:
    - Speed: ~100k SPS (between fast mode 200k+ and standard 30k)
    - Metrics: Better than fast mode due to increased capacity
    """

    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 2,  # Keep at 2 for reasonable speed
        eta: int = 3,  # Higher than fast mode (2) but not full (4)
        r: int = 6,  # Higher than fast mode (4) but not full (8)
        reset_on_terminate: bool = True,
        dropout: float = 0.05,  # Small dropout for generalization
        **kwargs,
    ):
        """Initialize optimized AGaLiTe.

        This variant uses the fast architecture with enhanced parameters.
        For maximum speed, use regular AGaLiTe with use_fast_mode=True (eta=2, r=4).
        For maximum metrics, increase eta/r further (at cost of speed).

        Args:
            env: Environment
            d_model: Model dimension
            d_head: Head dimension
            d_ffc: Feedforward dimension (not used in fast architecture)
            n_heads: Number of attention heads
            n_layers: Number of layers (2 recommended for speed)
            eta: Feature map expansion (3 = balanced, 2 = fast, 4 = full)
            r: Approximation order (6 = balanced, 4 = fast, 8 = full)
            reset_on_terminate: Whether to reset memory on termination
            dropout: Dropout rate for generalization
            **kwargs: Configuration parameters handled by mixin
        """
        logger.info(f"Creating AGaLiTeOptimized with eta={eta}, r={r}, layers={n_layers}")

        # Extract mixin parameters before passing to parent
        mixin_params = self.extract_mixin_params(kwargs)

        # Create the optimized policy
        policy = AGaLiTeOptimizedPolicy(
            env=env,
            d_model=d_model,
            d_head=d_head,
            d_ffc=d_ffc,
            n_heads=n_heads,
            n_layers=n_layers,
            eta=eta,
            r=r,
            reset_on_terminate=reset_on_terminate,
            dropout=dropout,
        )

        # Initialize with TransformerWrapper (from parent)
        # Note: We bypass AGaLiTe's __init__ and call TransformerWrapper directly
        from metta.agent.modules.transformer_wrapper import TransformerWrapper

        TransformerWrapper.__init__(self, env, policy, hidden_size=d_model)

        # Initialize mixin with configuration parameters
        self.init_mixin(**mixin_params)
