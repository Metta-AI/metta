import torch
from tensordict import TensorDict

from metta.agent.components.mamba import MambaBackboneConfig
from metta.agent.components.mamba.backbone import MambaBackboneComponent


def test_mamba_backbone_forward_rollout():
    config = MambaBackboneConfig(
        in_key="encoded",
        out_key="core",
        input_dim=16,
        d_model=32,
        d_intermediate=64,
        n_layer=2,
        max_cache_size=16,
    )
    component = MambaBackboneComponent(config)

    td = TensorDict(
        {
            "encoded": torch.randn(4, 16),
            "dones": torch.zeros(4),
            "truncateds": torch.zeros(4),
            "training_env_ids": torch.tensor([[0], [1], [2], [3]]),
        },
        batch_size=[4],
    )

    out = component(td)
    assert "core" in out.keys()
    assert out["core"].shape == torch.Size([4, 32])


def test_mamba_backbone_forward_training():
    config = MambaBackboneConfig(
        in_key="encoded",
        out_key="core",
        input_dim=16,
        d_model=32,
        d_intermediate=64,
        n_layer=2,
        max_cache_size=16,
    )
    component = MambaBackboneComponent(config)

    td = TensorDict(
        {
            "encoded": torch.randn(6, 2, 16),
            "bptt": torch.full((6,), 2, dtype=torch.long),
            "transformer_position": torch.zeros(6, dtype=torch.long),
        },
        batch_size=[6],
    )

    out = component(td)
    assert out["core"].shape == torch.Size([6, 32])


def test_mamba_config_resolves_stride_multiple():
    cfg = MambaBackboneConfig(
        in_key="encoded",
        out_key="core",
        input_dim=8,
        d_model=64,
        ssm_expand=2,
        ssm_headdim=16,
    )
    resolved = cfg.resolved_ssm_cfg()
    ratio = cfg.d_model * resolved["expand"] // resolved["headdim"]
    assert ratio % 8 == 0
