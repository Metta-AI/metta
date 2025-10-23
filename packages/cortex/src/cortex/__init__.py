"""Cortex: Modular neural stack library for stateful memory cells and composable blocks."""

from cortex.blocks import (
    AdapterBlock,
    BaseBlock,
    PassThroughBlock,
    PostUpBlock,
    PreUpBlock,
    build_block,
    register_block,
)
from cortex.cells import (
    AxonCell,
    AxonLayer,
    CausalConv1d,
    LSTMCell,
    MemoryCell,
    build_cell,
    mLSTMCell,
    register_cell,
    sLSTMCell,
)
from cortex.config import (
    AdapterBlockConfig,
    AxonConfig,
    BlockConfig,
    CausalConv1dConfig,
    CellConfig,
    CortexStackConfig,
    LSTMCellConfig,
    PassThroughBlockConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)
from cortex.factory import build_cortex, build_from_dict
from cortex.stacks import CortexStack
from cortex.types import MaybeState, ResetMask, State, Tensor
from cortex.utils import TRITON_AVAILABLE, select_backend

__all__ = [
    # Configuration
    "BlockConfig",
    "AdapterBlockConfig",
    "PassThroughBlockConfig",
    "PreUpBlockConfig",
    "PostUpBlockConfig",
    "CortexStackConfig",
    "CellConfig",
    "AxonConfig",
    "CausalConv1dConfig",
    "LSTMCellConfig",
    "mLSTMCellConfig",
    "sLSTMCellConfig",
    # Main classes
    "CortexStack",
    # Cells
    "MemoryCell",
    "AxonCell",
    "AxonLayer",
    "CausalConv1d",
    "LSTMCell",
    "mLSTMCell",
    "sLSTMCell",
    "register_cell",
    "build_cell",
    # Blocks
    "BaseBlock",
    "AdapterBlock",
    "PassThroughBlock",
    "PreUpBlock",
    "PostUpBlock",
    "register_block",
    "build_block",
    # Types
    "MaybeState",
    "ResetMask",
    "State",
    "Tensor",
    # Factory functions
    "build_cortex",
    "build_from_dict",
    # Utils
    "TRITON_AVAILABLE",
    "select_backend",
]
