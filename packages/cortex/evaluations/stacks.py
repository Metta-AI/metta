"""Template Cortex stack builders for the synthetic evaluation harness.

This module centralizes a handful of small, readable "stack recipes" that are
useful for quick comparisons on synthetic tasks. They showcase how to compose
cells and blocks via the configuration layer, and how to expose higher‑level
architectures like xLSTM behind a simple callable.

Add new templates by:
1) Writing a `build_*` function that returns a `CortexStack` (see examples).
2) Registering it in `STACKS` with a `StackSpec` so it becomes available via
   the `--stack` CLI flag in `run.py`.
"""

import dataclasses
import typing

import cortex.config
import cortex.factory
import cortex.stacks
import cortex.stacks.xlstm

# cortex_auto_stack is implemented in core (cortex.stacks.auto);
# this module simply imports and registers it below.


@dataclasses.dataclass
class StackSpec:
    name: str
    builder: typing.Callable[[], cortex.stacks.CortexStack]
    d_hidden: int


def build_slstm_postup(
    *, d_hidden: int = 128, proj_factor: float = 1.5, num_heads: int = 4
) -> cortex.stacks.CortexStack:
    """sLSTM cell in a PostUp block; cell size equals external hidden size."""
    cfg = cortex.config.CortexStackConfig(
        d_hidden=d_hidden,
        post_norm=True,
        blocks=[
            cortex.config.PostUpBlockConfig(
                proj_factor=proj_factor,
                # hidden_size may be None here; the stack builder sets it to d_hidden for PostUp
                cell=cortex.config.sLSTMCellConfig(
                    hidden_size=None, num_heads=num_heads, conv1d_kernel_size=4, dropout=0.0
                ),
            )
        ],
    )
    return cortex.factory.build_cortex(cfg)


def build_mlstm_preup(
    *, d_hidden: int = 128, proj_factor: float = 2.0, num_heads: int = 4
) -> cortex.stacks.CortexStack:
    """mLSTM cell in a PreUp block; cell runs at inner dim = proj_factor*d_hidden."""
    cfg = cortex.config.CortexStackConfig(
        d_hidden=d_hidden,
        post_norm=True,
        blocks=[
            cortex.config.PreUpBlockConfig(
                proj_factor=proj_factor,
                # hidden_size may be None here; the stack builder sets it to int(proj_factor * d_hidden) for PreUp
                cell=cortex.config.mLSTMCellConfig(
                    hidden_size=None, num_heads=num_heads, chunk_size=256, conv1d_kernel_size=4
                ),
            )
        ],
    )
    return cortex.factory.build_cortex(cfg)


def build_slstm_postup_axon(
    *, d_hidden: int = 128, proj_factor: float = 1.5, num_heads: int = 4
) -> cortex.stacks.CortexStack:
    """sLSTM PostUp variant with AxonLayer headwise gates enabled via flag.

    Only the per-head gate projections use Axon; the core sLSTM kernel remains unchanged.
    """
    cfg = cortex.config.CortexStackConfig(
        d_hidden=d_hidden,
        post_norm=True,
        blocks=[
            cortex.config.PreUpBlockConfig(
                # hidden_size is inferred from PreUp: int(proj_factor * d_hidden)
                cell=cortex.config.AxonConfig(
                    hidden_size=None, activation="silu", use_fullrank_rtu=False, use_untraced_linear=True
                )
            ),
            cortex.config.PostUpBlockConfig(
                proj_factor=proj_factor,
                cell=cortex.config.sLSTMCellConfig(
                    hidden_size=None,
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    dropout=0.0,
                    use_axon_layer=True,
                ),
            ),
        ],
    )
    return cortex.factory.build_cortex(cfg)


def build_mlstm_preup_axon(
    *, d_hidden: int = 128, proj_factor: float = 2.0, num_heads: int = 4
) -> cortex.stacks.CortexStack:
    """mLSTM PreUp variant with AxonLayer gates (3H→NH) enabled via flag."""
    cfg = cortex.config.CortexStackConfig(
        d_hidden=d_hidden,
        post_norm=True,
        blocks=[
            cortex.config.PassThroughBlockConfig(
                # hidden_size is inferred from PreUp: int(proj_factor * d_hidden)
                cell=cortex.config.AxonConfig(
                    hidden_size=None, activation="silu", use_fullrank_rtu=False, use_untraced_linear=True
                )
            ),
            cortex.config.PreUpBlockConfig(
                proj_factor=proj_factor,
                cell=cortex.config.mLSTMCellConfig(
                    hidden_size=None,
                    num_heads=num_heads,
                    chunk_size=256,
                    conv1d_kernel_size=4,
                    use_axon_layer=True,
                    use_axon_qkv=True,
                ),
            ),
        ],
    )
    return cortex.factory.build_cortex(cfg)


def build_axons_preup(*, d_hidden: int = 128, proj_factor: float = 2.0) -> cortex.stacks.CortexStack:
    """Axons (streaming RTU, diagonal) wrapped in a PreUp block.

    - The PreUp block projects inputs to an inner dim of ``proj_factor*d_hidden``
      before applying the Axons cell.
    - Axons assumes D == H internally and projects its 2H activation
      back to H, keeping the external hidden size consistent.
    """
    cfg = cortex.config.CortexStackConfig(
        d_hidden=d_hidden,
        post_norm=True,
        blocks=[
            cortex.config.PassThroughBlockConfig(
                # hidden_size is inferred from PreUp: int(proj_factor * d_hidden)
                cell=cortex.config.AxonConfig(
                    hidden_size=None, activation="silu", use_fullrank_rtu=False, use_untraced_linear=True
                )
            ),
            cortex.config.PreUpBlockConfig(
                proj_factor=proj_factor,
                # hidden_size is inferred from PreUp: int(proj_factor * d_hidden)
                cell=cortex.config.AxonConfig(
                    hidden_size=None, activation="silu", use_fullrank_rtu=False, use_untraced_linear=True
                ),
            ),
        ],
    )
    return cortex.factory.build_cortex(cfg)


# Registry of available stacks for the evaluation harness
STACKS: typing.Dict[str, StackSpec] = {
    # Single‑block templates
    "slstm": StackSpec(name="slstm_postup", builder=lambda: build_slstm_postup(), d_hidden=128),
    "mlstm": StackSpec(name="mlstm_preup", builder=lambda: build_mlstm_preup(), d_hidden=128),
    "slstm_axon": StackSpec(name="slstm_postup_axon", builder=lambda: build_slstm_postup_axon(), d_hidden=128),
    "mlstm_axon": StackSpec(name="mlstm_preup_axon", builder=lambda: build_mlstm_preup_axon(), d_hidden=128),
    "axons": StackSpec(name="axons_preup", builder=lambda: build_axons_preup(), d_hidden=128),
    # Composite templates
    # xLSTM: alternates mLSTM (PreUp) and sLSTM (PostUp)
    "xlstm": StackSpec(
        name="xlstm", builder=lambda: cortex.stacks.xlstm.build_xlstm_stack(d_hidden=128, num_blocks=3), d_hidden=128
    ),
    # Small and deeper variants for quick sweeps
    "xlstm_tiny": StackSpec(
        name="xlstm_tiny",
        builder=lambda: cortex.stacks.xlstm.build_xlstm_stack(d_hidden=128, num_blocks=2),
        d_hidden=128,
    ),
    "xlstm_deep": StackSpec(
        name="xlstm_deep",
        builder=lambda: cortex.stacks.xlstm.build_xlstm_stack(d_hidden=128, num_blocks=6),
        d_hidden=128,
    ),
    # Mixed auto stack cycling Axon/mLSTM/sLSTM with PreUp/PreUp/PostUp
    "cortex_auto": StackSpec(
        name="cortex_auto_stack",
        builder=lambda: cortex.stacks.build_cortex_auto_stack(
            d_hidden=128, num_layers=2, compile_blocks=False, pattern="AMS"
        ),
        d_hidden=128,
    ),
    # Variant with per-block torch.compile enabled for A/B comparisons
    "cortex_auto_compiled": StackSpec(
        name="cortex_auto_stack",
        builder=lambda: cortex.stacks.build_cortex_auto_stack(
            d_hidden=128, num_layers=2, compile_blocks=True, pattern="AXMS"
        ),
        d_hidden=128,
    ),
    "cortex_auto_axon": StackSpec(
        name="cortex_auto_stack",
        builder=lambda: cortex.stacks.build_cortex_auto_stack(d_hidden=128, num_layers=2, pattern="M^X^S^"),
        d_hidden=128,
    ),
}


__all__ = [
    "StackSpec",
    "STACKS",
    "build_slstm_postup",
    "build_mlstm_preup",
    "build_axons_preup",
    "build_cortex_auto_stack",
]
