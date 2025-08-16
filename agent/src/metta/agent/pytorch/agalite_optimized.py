"""
Optimized AGaLiTe variant for improved metrics.
This is separate from the main agalite.py to preserve the exact fast mode performance.
"""

from metta.agent.pytorch.agalite import AGaLiTe as BaseAGaLiTe


class AGaLiTeOptimized(BaseAGaLiTe):
    """
    Optimized AGaLiTe for better metrics while being slower than fast mode.
    
    Changes from base AGaLiTe:
    - 3 layers instead of 2 for better learning capacity
    - Small dropout (0.1) for better generalization
    - Can use eta=3, r=6 for balanced speed/performance (optional)
    """
    
    def __init__(
        self,
        env,
        d_model: int = 256,
        d_head: int = 64,
        d_ffc: int = 1024,
        n_heads: int = 4,
        n_layers: int = 3,  # Increased for better learning
        eta: int = 3,  # Reduced for better balance
        r: int = 6,    # Reduced for better balance
        reset_on_terminate: bool = True,
        dropout: float = 0.1,  # Small dropout for generalization
        use_fast_mode: bool = False,  # Should not use fast mode
        **kwargs,
    ):
        """Initialize optimized AGaLiTe.
        
        This variant is designed for better metrics, not speed.
        For maximum speed, use the regular AGaLiTe with use_fast_mode=True.
        """
        if use_fast_mode:
            raise ValueError(
                "AGaLiTeOptimized is for better metrics, not speed. "
                "Use regular AGaLiTe with use_fast_mode=True for maximum speed."
            )
        
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
            use_fast_mode=False,
            **kwargs,
        )