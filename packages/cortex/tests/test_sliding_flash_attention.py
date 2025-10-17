import torch
from cortex._triton_stub import install_triton_stub

install_triton_stub()

from cortex import CortexStackConfig, PassThroughBlockConfig, build_cortex  # noqa: E402
from cortex.config import SlidingFlashAttentionConfig  # noqa: E402


def _build_stack(d_hidden: int = 32, window: int = 4) -> tuple:
    recipe = CortexStackConfig(
        d_hidden=d_hidden,
        blocks=[
            PassThroughBlockConfig(
                cell=SlidingFlashAttentionConfig(
                    hidden_size=None,
                    num_heads=4,
                    window_size=window,
                    dropout=0.0,
                )
            )
        ],
        post_norm=False,
    )
    stack = build_cortex(recipe)
    state = stack.init_state(batch=3, device=torch.device("cpu"), dtype=torch.float32)
    return stack, state, window


def test_sliding_flash_attention_sequence_matches_step():
    torch.manual_seed(0)
    stack, state, window = _build_stack(d_hidden=32, window=5)
    batch, seq_len, hidden = 3, 6, 32
    x = torch.randn(batch, seq_len, hidden)

    seq_out, seq_state = stack(x, state)

    step_state = stack.init_state(batch=batch, device=x.device, dtype=x.dtype)
    outputs = []
    for t in range(seq_len):
        out_t, step_state = stack.step(x[:, t], step_state)
        outputs.append(out_t.unsqueeze(1))
    step_out = torch.cat(outputs, dim=1)

    assert torch.allclose(seq_out, step_out, atol=1e-5)
    cache = step_state["PassThroughBlock_0"]["SlidingFlashAttentionCell"]["cache_len"]
    assert torch.all(cache <= window - 1)
    assert torch.equal(seq_state["PassThroughBlock_0"]["SlidingFlashAttentionCell"]["cache_len"], cache)


def test_sliding_flash_attention_resets_clear_cache():
    stack, state, window = _build_stack(d_hidden=16, window=3)
    batch, hidden = 3, 16
    x0 = torch.randn(batch, hidden)
    out, state = stack.step(x0, state)
    assert out.shape == (batch, hidden)

    resets = torch.tensor([True, False, True])
    x1 = torch.randn(batch, hidden)
    _, state = stack.step(x1, state, resets=resets)

    cell_state = state["PassThroughBlock_0"]["SlidingFlashAttentionCell"]
    cache_len = cell_state["cache_len"]
    assert torch.equal(cache_len[resets], torch.ones_like(cache_len[resets]))
    assert torch.all(cache_len[~resets] >= 1)
