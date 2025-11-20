"""Memory cell implementations for stateful neural computation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cortex.cells.agalite import AGaLiTeCell
    from cortex.cells.base import MemoryCell
    from cortex.cells.conv import CausalConv1d
    from cortex.cells.core import AxonCell, AxonLayer
    from cortex.cells.hf_llama import HFLlamaLayerCell
    from cortex.cells.lstm import LSTMCell
    from cortex.cells.mlstm import mLSTMCell
    from cortex.cells.registry import build_cell, get_cell_class, register_cell
    from cortex.cells.slstm import sLSTMCell
    from cortex.cells.xl import XLCell

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "AGaLiTeCell": ("cortex.cells.agalite", "AGaLiTeCell"),
    "MemoryCell": ("cortex.cells.base", "MemoryCell"),
    "CausalConv1d": ("cortex.cells.conv", "CausalConv1d"),
    "AxonCell": ("cortex.cells.core", "AxonCell"),
    "AxonLayer": ("cortex.cells.core", "AxonLayer"),
    "HFLlamaLayerCell": ("cortex.cells.hf_llama", "HFLlamaLayerCell"),
    "LSTMCell": ("cortex.cells.lstm", "LSTMCell"),
    "mLSTMCell": ("cortex.cells.mlstm", "mLSTMCell"),
    "build_cell": ("cortex.cells.registry", "build_cell"),
    "get_cell_class": ("cortex.cells.registry", "get_cell_class"),
    "register_cell": ("cortex.cells.registry", "register_cell"),
    "sLSTMCell": ("cortex.cells.slstm", "sLSTMCell"),
    "XLCell": ("cortex.cells.xl", "XLCell"),
}

__all__ = [
    "MemoryCell",
    "CausalConv1d",
    "LSTMCell",
    "mLSTMCell",
    "AxonCell",
    "AxonLayer",
    "sLSTMCell",
    "XLCell",
    "HFLlamaLayerCell",
    "AGaLiTeCell",
    "register_cell",
    "build_cell",
    "get_cell_class",
]

def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'metta.cortex.cells' has no attribute '{name}'")

    module_path, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_path)
    attr = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
