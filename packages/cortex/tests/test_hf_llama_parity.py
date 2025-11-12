import os
import torch
import pytest

from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from cortex.stacks.hf import build_llama_stack_from_model


# ---------------------------- Kernel-level tests ----------------------------
# Skip this module entirely by default (slow).
# Set RUN_SLOW_CORTEX_TESTS=1 to enable; otherwise keep skipped.
_RUN_SLOW = os.getenv("RUN_SLOW_CORTEX_TESTS", "0").lower() in {"1", "true", "yes", "y", "on"}
pytestmark = (
    pytest.mark.slow
    if _RUN_SLOW
    else pytest.mark.skip(reason="slow full-rank RTU parity suite (set RUN_SLOW_CORTEX_TESTS=1 to run)")
)


@pytest.mark.parametrize("B,T,H", [(2, 12, 32)])
def test_llama_stack_full_parity(B: int, T: int, H: int) -> None:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=H,
        intermediate_size=H * 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    hf = LlamaForCausalLM(cfg).eval()

    stack = build_llama_stack_from_model(hf, compile_blocks=False).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    embeds = hf.model.embed_tokens(input_ids)

    with torch.no_grad():
        ref = hf.model(inputs_embeds=embeds, use_cache=True).last_hidden_state

    with torch.no_grad():
        out, _ = stack(embeds, None)
        out = hf.model.norm(out)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("chunk", [1, 3, 5])
def test_llama_stack_streaming_chunk_parity(chunk: int) -> None:
    torch.manual_seed(1)
    B, T, H = 2, 17, 32
    cfg = LlamaConfig(
        hidden_size=H,
        intermediate_size=H * 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    hf = LlamaForCausalLM(cfg).eval()
    # Use a mem_len >= T to avoid trimming during chunked streaming
    stack = build_llama_stack_from_model(hf, mem_len=T, compile_blocks=False).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    embeds = hf.model.embed_tokens(input_ids)

    # Baseline full pass
    with torch.no_grad():
        ref_full = hf.model(inputs_embeds=embeds, use_cache=True).last_hidden_state

    # HF streaming with DynamicCache
    cache = DynamicCache(config=hf.config)
    out_chunks = []
    with torch.no_grad():
        t = 0
        while t < T:
            e = embeds[:, t : t + chunk]
            o = hf.model(inputs_embeds=e, use_cache=True, past_key_values=cache)
            out_chunks.append(o.last_hidden_state)
            t += chunk
        ref_stream = torch.cat(out_chunks, dim=1)

    # Cortex streaming
    out_chunks = []
    st = None
    with torch.no_grad():
        t = 0
        while t < T:
            e = embeds[:, t : t + chunk]
            y, st = stack(e, st)
            y = hf.model.norm(y)
            out_chunks.append(y)
            t += chunk
        out_stream = torch.cat(out_chunks, dim=1)

    # Parity checks
    assert torch.allclose(ref_stream, ref_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_stream, ref_full, atol=1e-5, rtol=1e-5)


def test_smollm_llama_parity_and_streaming() -> None:
    # Download + larger model; enable only when explicitly requested
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for SmolLM parity test (memory)")

    device = torch.device("cuda")
    dtype = torch.float16

    # Define sequence length before building the stack so mem_len can match
    B, T = 1, 24
    name = "HuggingFaceTB/SmolLM-360M"
    hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    # Use a mem_len >= T to avoid trimming during chunked streaming
    stack = build_llama_stack_from_model(hf, mem_len=T, compile_blocks=False).to(device).eval()
    vocab = int(hf.config.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (B, T), device=device)
    embeds = hf.model.embed_tokens(input_ids)

    with torch.no_grad():
        ref_full = hf.model(inputs_embeds=embeds, use_cache=True).last_hidden_state

    # streaming (chunk=8)
    chunk = 8
    cache = DynamicCache(config=hf.config)
    out_chunks = []
    with torch.no_grad():
        t = 0
        while t < T:
            e = embeds[:, t : t + chunk]
            o = hf.model(inputs_embeds=e, use_cache=True, past_key_values=cache)
            out_chunks.append(o.last_hidden_state)
            t += chunk
        ref_stream = torch.cat(out_chunks, dim=1)

    st = None
    out_chunks = []
    with torch.no_grad():
        t = 0
        while t < T:
            e = embeds[:, t : t + chunk]
            y, st = stack(e, st)
            y = hf.model.norm(y)
            out_chunks.append(y)
            t += chunk
        out_stream = torch.cat(out_chunks, dim=1)

    # fp16 tolerances (allow drift on GPU)
    # Compare Cortex streaming to HF streaming (HF streaming may diverge slightly from full in fp16)
    max_diff = (out_stream - ref_stream).abs().max().item()
    assert max_diff < 2e-2, f"Cortex vs HF streaming max diff {max_diff}"
