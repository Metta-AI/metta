import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("mamba_ssm", reason="Mamba components require the mamba-ssm CUDA package.")

from metta.agent.components.mamba import MambaBackboneConfig
from metta.agent.components.mamba.backbone import MambaBackboneComponent


def _make_backbone(d_model: int = 32) -> MambaBackboneComponent:
    return MambaBackboneComponent(
        MambaBackboneConfig(
            in_key="encoded",
            out_key="core",
            input_dim=16,
            d_model=d_model,
            d_intermediate=d_model * 2,
            n_layer=2,
            max_cache_size=64,
        )
    )


def test_rollout_updates_and_resets_cache():
    component = _make_backbone()

    td = TensorDict(  # 2 envs, single step
        {
            "encoded": torch.randn(2, 16),
            "dones": torch.zeros(2),
            "truncateds": torch.zeros(2),
            "training_env_ids": torch.tensor([[0], [1]]),
        },
        batch_size=[2],
    )

    out = component(td)
    assert out["core"].shape == torch.Size([2, component.config.d_model])
    assert "transformer_position" in out.keys()

    pos = out["transformer_position"]
    assert torch.all(pos > 0)

    # Trigger reset on env 1
    td_reset = td.clone()
    td_reset["dones"][1] = 1.0
    out_reset = component(td_reset)
    pos_reset = out_reset["transformer_position"]
    assert pos_reset[1] == 0
    assert pos_reset[0] > pos[0]


def test_training_keeps_shape_and_updates_cache():
    component = _make_backbone()

    td = TensorDict(
        {
            "encoded": torch.randn(4, 2, 16),
            "bptt": torch.full((4,), 2, dtype=torch.long),
            "transformer_position": torch.zeros(4, dtype=torch.long),
        },
        batch_size=[4],
    )

    out = component(td)
    assert out["core"].shape == torch.Size([4, component.config.d_model])
