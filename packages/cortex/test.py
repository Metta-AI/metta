"""Quick manual forward pass for the HF→Cortex LLaMA wrapper."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from cortex.stacks.hf import build_hf_stack, build_llama_stack_from_model


def _unwrap_cache(obj):
    return getattr(obj, "data", obj)


def print_hf_stack_state(stack, state) -> None:
    print("Top-level keys:", list(state.keys()))
    for i, _ in enumerate(stack.blocks):
        block_key = f"PassThroughBlock_{i}"
        cell_key = "HFLlamaLayerCell"
        cell_state = state.get(block_key).get(cell_key)
        pos = cell_state.get("pos")
        cache = _unwrap_cache(cell_state.get("cache"))
        # Query the correct layer slot to recover past length (kv_len includes qlen)
        qlen = 1
        kv_len, kv_off = cache.get_mask_sizes(torch.arange(qlen, device=pos.device), i)
        past_len = kv_len - qlen
        print(f"Layer {i}: pos={pos.tolist()} cache_past_len={int(past_len)} kv_offset={int(kv_off)}")


def build_models(model_name: Optional[str], device: str, dtype: torch.dtype):
    if model_name:
        hf = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        stack = build_hf_stack(model_name, torch_dtype=dtype, device=device, compile_blocks=False).eval()
        return hf, stack
    # Tiny no-download config
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    hf = LlamaForCausalLM(cfg).to(device).eval()
    stack = build_llama_stack_from_model(hf, compile_blocks=False).to(device).eval()
    return hf, stack


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None, help="HF model name (default: tiny local LLaMA config)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tokens", type=int, default=8)
    args = p.parse_args(argv)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device

    hf, stack = build_models(args.model, device, dtype)

    B, T = args.batch, args.tokens
    vocab = int(hf.config.vocab_size)
    ids = torch.randint(0, vocab, (B, T), device=device)
    embeds = hf.model.embed_tokens(ids)

    print("Embeds:", embeds.shape, embeds.dtype, embeds.device)
    y, st = stack(embeds, None)
    print("Stack out:", y.shape)
    print_hf_stack_state(stack, st)

    # Optionally project to logits via HF head
    y_norm = hf.model.norm(y)
    logits = hf.lm_head(y_norm)
    print("Logits:", logits.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
