import torch
from cortex import (
    AxonLayer,
    ColumnBlockConfig,
    CortexStackConfig,
    LSTMCellConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    RouterConfig,
    XLCellConfig,
    build_column_auto_block,
    build_column_auto_config,
    build_cortex,
    mLSTMCellConfig,
    sLSTMCellConfig,
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
    from cortex import ColumnBlock as _ColumnBlock

    col = next(b for b in stack.blocks if isinstance(b, _ColumnBlock))
    B2, T2 = 2, 3
    x = torch.randn(B2, T2, d_hidden)
    expert_outs = []
    for expert in col.experts:
        y_i, _ = expert(x, None)
        expert_outs.append(y_i)
    gate = col.router(expert_outs)
    assert gate.shape[0] == k
    diff = torch.abs(gate - (1.0 / k)).max().item()
    assert diff < 0.25, f"Gate not near-uniform at init: max diff {diff}"


def test_auto_config_builtin_patterns():
    cfg = build_column_auto_config(d_hidden=64, pattern="AXMS")
    assert isinstance(cfg, ColumnBlockConfig)
    assert len(cfg.experts) == 4

    cfg2 = build_column_auto_config(d_hidden=64, pattern="A X M S")
    assert len(cfg2.experts) == 4

    cfg3 = build_column_auto_config(d_hidden=64, pattern="M^X^S^")
    assert len(cfg3.experts) == 3
    assert isinstance(cfg3.experts[0], PreUpBlockConfig)
    assert isinstance(cfg3.experts[1], PostUpBlockConfig)
    assert isinstance(cfg3.experts[2], PostUpBlockConfig)


def test_auto_config_custom_overrides_and_tokens():
    custom = {
        "M": PreUpBlockConfig(cell=mLSTMCellConfig(num_heads=8, chunk_size=32)),
        "M^": PreUpBlockConfig(cell=mLSTMCellConfig(use_axon_layer=True, use_axon_qkv=False)),
        "P": PostUpBlockConfig(cell=sLSTMCellConfig(num_heads=2)),
    }
    cfg = build_column_auto_config(d_hidden=64, pattern="A P M M^", custom_map=custom)
    assert len(cfg.experts) == 4
    assert isinstance(cfg.experts[2], PreUpBlockConfig)
    assert isinstance(cfg.experts[2].cell, mLSTMCellConfig)
    assert cfg.experts[2].cell.num_heads == 8
    assert cfg.experts[2].cell.chunk_size == 32
    assert isinstance(cfg.experts[3].cell, mLSTMCellConfig)
    assert cfg.experts[3].cell.use_axon_layer is True
    assert cfg.experts[3].cell.use_axon_qkv is False


def test_auto_config_axonify_flags():
    cfg = build_column_auto_config(d_hidden=64, pattern="M^X^S^")
    m_cfg = cfg.experts[0]
    x_cfg = cfg.experts[1]
    s_cfg = cfg.experts[2]
    assert isinstance(m_cfg, PreUpBlockConfig)
    assert isinstance(x_cfg, PostUpBlockConfig)
    assert isinstance(s_cfg, PostUpBlockConfig)
    assert isinstance(m_cfg.cell, mLSTMCellConfig) and m_cfg.cell.use_axon_layer and m_cfg.cell.use_axon_qkv
    assert isinstance(x_cfg.cell, XLCellConfig) and x_cfg.cell.use_axon_qkv
    assert isinstance(s_cfg.cell, sLSTMCellConfig) and s_cfg.cell.use_axon_layer


def test_auto_block_forward_and_state():
    d_hidden = 32
    block = build_column_auto_block(d_hidden=d_hidden, pattern="AXMSM^X^S^")
    B, T = 2, 5
    x = torch.randn(B, T, d_hidden)
    state = block.init_state(batch=B, device=x.device, dtype=x.dtype)
    y, new_state = block(x, state)
    assert y.shape == x.shape
    assert new_state.batch_size[0] == B

    col_cfg = build_column_auto_config(d_hidden=d_hidden, pattern="AXMS")
    stack_cfg = CortexStackConfig(d_hidden=d_hidden, blocks=[col_cfg], post_norm=False)
    stack = build_cortex(stack_cfg)
    x_step = torch.randn(B, d_hidden)
    y_step, _ = stack.step(x_step, stack.init_state(batch=B, device=x_step.device, dtype=x_step.dtype))
    assert y_step.shape == x_step.shape


def test_auto_router_override():
    router = RouterConfig(d_key=128, top_k=2, temperature=0.7)
    cfg = build_column_auto_config(d_hidden=64, pattern="AXMS", router=router)
    assert cfg.router.d_key == 128
    assert cfg.router.top_k == 2
    assert cfg.router.temperature == 0.7


def test_auto_block_axonified_modules():
    block = build_column_auto_block(d_hidden=32, pattern="M^X^S^")
    m_cell = block.experts[0].cell  # type: ignore[attr-defined]
    assert m_cell.use_axon_qkv is True
    assert isinstance(m_cell.igate, AxonLayer) and isinstance(m_cell.fgate, AxonLayer)
    x_cell = block.experts[1].cell  # type: ignore[attr-defined]
    assert isinstance(x_cell.q_proj, AxonLayer)
    assert isinstance(x_cell.k_proj, AxonLayer)
    assert isinstance(x_cell.v_proj, AxonLayer)
    s_cell = block.experts[2].cell  # type: ignore[attr-defined]
    assert hasattr(s_cell, "if_fused") and hasattr(s_cell, "zo_fused")
