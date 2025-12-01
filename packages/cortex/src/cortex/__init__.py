"""Cortex: Modular neural stack library for stateful memory cells and composable blocks."""

from cortex.blocks import (
    AdapterBlock,
    BaseBlock,
    ColumnBlock,
    PassThroughBlock,
    PostUpBlock,
    PreUpBlock,
    build_block,
    build_column_auto_block,
    build_column_auto_config,
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
    ColumnBlockConfig,
    CortexStackConfig,
    LSTMCellConfig,
    PassThroughBlockConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    RouterConfig,
    XLCellConfig,
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
    "ColumnBlockConfig",
    "PassThroughBlockConfig",
    "PreUpBlockConfig",
    "PostUpBlockConfig",
    "RouterConfig",
    "CortexStackConfig",
    "CellConfig",
    "AxonConfig",
    "CausalConv1dConfig",
    "LSTMCellConfig",
    "XLCellConfig",
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
    "ColumnBlock",
    "build_column_auto_config",
    "build_column_auto_block",
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
