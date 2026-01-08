import os

import pytest
import torch
from cortex.cuda_utils import is_cuda_supported
from cortex.stacks.hf import build_llama_stack_from_model
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

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
    stack = build_llama_stack_from_model(hf, mem_len=T, compile_blocks=False).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    embeds = hf.model.embed_tokens(input_ids)

    with torch.no_grad():
        ref_full = hf.model(inputs_embeds=embeds, use_cache=True).last_hidden_state

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

    assert torch.allclose(ref_stream, ref_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_stream, ref_full, atol=1e-5, rtol=1e-5)


def test_smollm_llama_parity_and_streaming() -> None:
    if not is_cuda_supported():
        pytest.skip("CUDA required for SmolLM parity test (memory)")

    device = torch.device("cuda")
    dtype = torch.float16

    B, T = 1, 24
    name = "HuggingFaceTB/SmolLM-360M"
    hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    stack = build_llama_stack_from_model(hf, mem_len=T, compile_blocks=False).to(device).eval()
    vocab = int(hf.config.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (B, T), device=device)
    embeds = hf.model.embed_tokens(input_ids)

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

    max_diff = (out_stream - ref_stream).abs().max().item()
    assert max_diff < 0.1, f"Cortex vs HF streaming max diff {max_diff}"


def test_llama_static_cache_updates_inplace() -> None:
    torch.manual_seed(0)

    B, T, H = 2, 5, 32
    mem_len = 8
    cfg = LlamaConfig(
        hidden_size=H,
        intermediate_size=H * 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    hf = LlamaForCausalLM(cfg).eval()
    stack = build_llama_stack_from_model(hf, mem_len=mem_len, compile_blocks=False).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    embeds = hf.model.embed_tokens(input_ids)

    st = None
    k_ptr = None
    v_ptr = None
    last_k = None
    last_kv_len = None

    for t in range(T):
        e = embeds[:, t : t + 1]
        _, st = stack(e, st)

        block0 = st.get("PassThroughBlock_0")
        cell = block0.get("HFLlamaLayerCell")
        assert all(k in cell.keys() for k in ("k", "v", "kv_len")), "Missing KV tensors in state"

        k = cell.get("k")
        v = cell.get("v")
        kv_len = cell.get("kv_len")

        n_kv = k.shape[1]
        head_dim = k.shape[-1]
        assert k.shape == (B, n_kv, mem_len, head_dim)
        assert v.shape == (B, n_kv, mem_len, head_dim)
        assert kv_len.shape == (B,)

        if t == 0:
            k_ptr = k.data_ptr()
            v_ptr = v.data_ptr()
        else:
            assert k.data_ptr() == k_ptr
            assert v.data_ptr() == v_ptr

        if last_kv_len is not None:
            expected = torch.clamp(last_kv_len + 1, max=mem_len)
            assert torch.equal(kv_len, expected)
        last_kv_len = kv_len.clone()

        if last_k is not None:
            assert (k - last_k).abs().sum() > 0
        last_k = k.clone()


def _run_hf_stream_with_resets(hf: LlamaForCausalLM, embeds: torch.Tensor, resets_bt: torch.Tensor) -> torch.Tensor:
    """Reference streaming by running each sequence independently and resetting its cache on resets."""
    B, T, _ = embeds.shape
    outs = []
    for b in range(B):
        cache = DynamicCache(config=hf.config)
        o_chunks = []
        for t in range(T):
            if resets_bt[b, t].item() != 0:
                cache = DynamicCache(config=hf.config)
            e = embeds[b : b + 1, t : t + 1]
            o = hf.model(inputs_embeds=e, use_cache=True, past_key_values=cache)
            o_chunks.append(o.last_hidden_state)
        outs.append(torch.cat(o_chunks, dim=1))
    return torch.cat(outs, dim=0)


def test_llama_per_timestep_resets_parity() -> None:
    torch.manual_seed(42)
    B, T, H = 2, 11, 32
    cfg = LlamaConfig(
        hidden_size=H,
        intermediate_size=H * 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    hf = LlamaForCausalLM(cfg).eval()
    mem_len = T
    stack = build_llama_stack_from_model(hf, mem_len=mem_len, compile_blocks=False).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    embeds = hf.model.embed_tokens(input_ids)

    resets_bt = torch.zeros(B, T, dtype=torch.long)
    resets_bt[0, 3] = 1
    resets_bt[1, 7] = 1

    with torch.no_grad():
        ref = _run_hf_stream_with_resets(hf, embeds, resets_bt)

    out_chunks = []
    st = None
    with torch.no_grad():
        for t in range(T):
            e = embeds[:, t : t + 1]
            y, st = stack(e, st, resets=resets_bt[:, t : t + 1])
            y = hf.model.norm(y)
            out_chunks.append(y)
        out = torch.cat(out_chunks, dim=1)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_llama_mem_len_smaller_than_chunk_T_gt_M() -> None:
    torch.manual_seed(0)
    B, T, H = 2, 12, 32
    mem_len = 5
    cfg = LlamaConfig(
        hidden_size=H,
        intermediate_size=H * 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    hf = LlamaForCausalLM(cfg).eval()
    stack = build_llama_stack_from_model(hf, mem_len=mem_len, compile_blocks=False).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    embeds = hf.model.embed_tokens(input_ids)

    with torch.no_grad():
        ref = hf.model(inputs_embeds=embeds, use_cache=True).last_hidden_state

    with torch.no_grad():
        out, st = stack(embeds, None)
        out = hf.model.norm(out)

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()

    block0 = st.get("PassThroughBlock_0")
    cell = block0.get("HFLlamaLayerCell")
    k = cell.get("k")
    v = cell.get("v")
    kv_len = cell.get("kv_len")

    n_kv = k.shape[1]
    head_dim = k.shape[-1]
    assert k.shape == (B, n_kv, mem_len, head_dim)
    assert v.shape == (B, n_kv, mem_len, head_dim)
    assert kv_len.shape == (B,)
    assert torch.equal(kv_len, torch.full_like(kv_len, mem_len))


def test_llama_streaming_chunk_T_gt_M() -> None:
    torch.manual_seed(0)
    B, T_total, H = 2, 17, 32
    mem_len = 5
    chunk = 7

    cfg = LlamaConfig(
        hidden_size=H,
        intermediate_size=H * 2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
    )
    hf = LlamaForCausalLM(cfg).eval()
    stack = build_llama_stack_from_model(hf, mem_len=mem_len, compile_blocks=False).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (B, T_total))
    embeds = hf.model.embed_tokens(input_ids)

    with torch.no_grad():
        ref_full = hf.model(inputs_embeds=embeds, use_cache=True).last_hidden_state

    cache = DynamicCache(config=hf.config)
    ref_chunks = []
    with torch.no_grad():
        t = 0
        while t < T_total:
            e = embeds[:, t : t + chunk]
            o = hf.model(inputs_embeds=e, use_cache=True, past_key_values=cache)
            ref_chunks.append(o.last_hidden_state)
            legacy = cache.to_legacy_cache()
            cropped = []
            for k_layer, v_layer in legacy:
                if k_layer is None or v_layer is None or k_layer.shape[2] <= mem_len:
                    cropped.append((k_layer, v_layer))
                else:
                    cropped.append((k_layer[:, :, -mem_len:, :], v_layer[:, :, -mem_len:, :]))
            cache = DynamicCache.from_legacy_cache(tuple(cropped))
            t += chunk
        ref_window_stream = torch.cat(ref_chunks, dim=1)

    out_chunks = []
    st = None
    with torch.no_grad():
        t = 0
        while t < T_total:
            e = embeds[:, t : t + chunk]
            y, st = stack(e, st)
            y = hf.model.norm(y)
            out_chunks.append(y)
            t += chunk
        out_stream = torch.cat(out_chunks, dim=1)

    assert out_stream.shape == ref_full.shape
    assert torch.isfinite(out_stream).all()
    torch.testing.assert_close(
        out_stream,
        ref_window_stream,
        rtol=1e-3,
        atol=1e-2,
        msg="Cortex streaming with T>mem_len diverges from HF sliding-window reference",
    )

    block0 = st.get("PassThroughBlock_0")
    cell = block0.get("HFLlamaLayerCell")
    k = cell.get("k")
    v = cell.get("v")
    kv_len = cell.get("kv_len")

    n_kv = k.shape[1]
    head_dim = k.shape[-1]
    assert k.shape == (B, n_kv, mem_len, head_dim)
    assert v.shape == (B, n_kv, mem_len, head_dim)
    assert kv_len.shape == (B,)
    assert torch.equal(kv_len, torch.full_like(kv_len, mem_len))
