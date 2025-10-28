import torch
from cortex import (
    ColumnBlockConfig,
    CortexStackConfig,
    LSTMCellConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    RouterConfig,
    build_cortex,
)


def _stack_with_column(d_hidden: int = 64, k: int = 3):
    experts = [
        PreUpBlockConfig(cell=LSTMCellConfig(hidden_size=None), proj_factor=2.0),
        PostUpBlockConfig(cell=LSTMCellConfig(hidden_size=None), proj_factor=1.5),
    ]
    experts = (experts * ((k + len(experts) - 1) // len(experts)))[:k]

    col = ColumnBlockConfig(experts=experts, router=RouterConfig(d_key=None, temperature=1.0, top_k=None))
    cfg = CortexStackConfig(d_hidden=d_hidden, blocks=[col], post_norm=False, compile_blocks=True)
    return build_cortex(cfg)


def test_column_shapes_and_state():
    torch.manual_seed(0)
    d_hidden = 32
    B, T = 2, 5
    stack = _stack_with_column(d_hidden=d_hidden, k=3)

    x_seq = torch.randn(B, T, d_hidden)
    state = stack.init_state(batch=B, device=x_seq.device, dtype=x_seq.dtype)
    y, new_state = stack(x_seq, state)
    assert y.shape == x_seq.shape
    # Verify nested expert states updated from init -> new_state
    block_key = next(k for k in new_state.keys() if str(k).startswith("ColumnBlock_"))
    init_block_state = state.get(block_key)
    new_block_state = new_state.get(block_key)
    assert init_block_state is not None and new_block_state is not None
    expert_keys = [k for k in new_block_state.keys() if str(k).startswith("expert_")]
    assert expert_keys, "No expert states found in Column state"
    for ek in expert_keys:
        init_expert = init_block_state.get(ek)
        new_expert = new_block_state.get(ek)
        assert new_expert is not None
        # Experts wrap an LSTM cell in this test setup
        lstm_init = init_expert.get("LSTMCell") if init_expert is not None else None
        lstm_new = new_expert.get("LSTMCell")
        assert lstm_new is not None
        # Compare hidden state 'h' changed after forward
        if lstm_init is not None and "h" in lstm_init.keys() and "h" in lstm_new.keys():
            assert not torch.allclose(lstm_init["h"], lstm_new["h"]), "Expert LSTM 'h' state did not update"
        if lstm_init is not None and "c" in lstm_init.keys() and "c" in lstm_new.keys():
            assert not torch.allclose(lstm_init["c"], lstm_new["c"]), "Expert LSTM 'c' state did not update"

    # Step mode
    x_step = torch.randn(B, d_hidden)
    y_step, new_state = stack.step(x_step, new_state)
    assert y_step.shape == x_step.shape


def test_router_uniform_init():
    d_hidden = 16
    k = 4
    stack = _stack_with_column(d_hidden=d_hidden, k=k)
    col = next(b for b in stack.blocks if hasattr(b, "keys"))
    gate = col._compute_gate(torch.zeros(1, d_hidden))  # type: ignore[attr-defined]
    assert gate.shape[0] == k
    diff = torch.abs(gate - (1.0 / k)).max().item()
    assert diff < 0.25, f"Gate not near-uniform at init: max diff {diff}"
