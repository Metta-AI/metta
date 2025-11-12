"""Build Cortex stacks from Hugging Face CausalLM models (LLaMA focus)."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from cortex.blocks import PassThroughBlock
from cortex.config import CortexStackConfig, PassThroughBlockConfig
from cortex.stacks.base import CortexStack

# Local cell config
from cortex.cells.hf_llama import HFLlamaLayerConfig  # noqa: F401


def build_llama_stack_from_model(
    model: PreTrainedModel,
    *,
    mem_len: int = 0,
    compile_blocks: bool = False,
) -> CortexStack:
    """Build a CortexStack wrapping LLaMA decoder layers; expects float hidden_states and leaves embed/head in HF."""
    # Resolve submodel with layers
    layers = model.model.layers  # type: ignore[attr-defined]
    hf_submodel = model.model  # type: ignore[attr-defined]

    hidden_size = int(model.config.hidden_size)  # type: ignore[attr-defined]

    blocks: list[PassThroughBlockConfig] = []
    for layer in layers:
        cell_cfg = HFLlamaLayerConfig(
            hidden_size=hidden_size,
            mem_len=int(mem_len),
        )
        # attach runtime objects on the config (CellConfig allows extra fields)
        setattr(cell_cfg, "hf_layer", layer)
        setattr(cell_cfg, "hf_submodel", hf_submodel)
        setattr(cell_cfg, "hf_config", model.config)
        blocks.append(PassThroughBlockConfig(cell=cell_cfg))

    stack_cfg = CortexStackConfig(
        blocks=blocks,
        d_hidden=hidden_size,
        post_norm=False,
        compile_blocks=bool(compile_blocks),
    )
    return CortexStack(stack_cfg)


def build_hf_stack(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[str | torch.device] = None,
    num_layers: Optional[int] = None,
    mem_len: int = 0,
    compile_blocks: bool = False,
) -> CortexStack:
    """Load a HF CausalLM (LLaMA) by name and return a CortexStack wrapping its layers."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
    )
    if device is not None:
        model.to(device)
    model.eval()

    model_type = getattr(model.config, "model_type", None)
    if model_type != "llama":
        raise NotImplementedError(f"build_hf_stack currently supports LLaMA; got model_type={model_type}")

    if num_layers is not None and num_layers > 0:
        # Optionally truncate to the first N layers for quicker experiments
        model.model.layers = model.model.layers[:num_layers]  # type: ignore[attr-defined]

    return build_llama_stack_from_model(
        model,
        mem_len=mem_len,
        compile_blocks=compile_blocks,
    )


__all__ = ["build_hf_stack", "build_llama_stack_from_model"]
