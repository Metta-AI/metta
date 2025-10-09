import torch
from tensordict import TensorDict

from metta.agent.components.drama import DramaWorldModelConfig
from metta.agent.components.drama.world_model_component import DramaWorldModelComponent


def test_drama_world_model_forward_rollout() -> None:
    config = DramaWorldModelConfig(
        in_key="latent",
        out_key="core",
        action_key="actions",
        stoch_dim=8,
        action_dim=4,
        d_model=96,
        d_intermediate=192,
        n_layer=1,
    )
    component = DramaWorldModelComponent(config)

    td = TensorDict(
        {
            "latent": torch.randn(6, 8),
            "actions": torch.randint(0, config.action_dim, (6,)),
        },
        batch_size=[6],
    )

    out = component(td)
    assert out[config.out_key].shape == torch.Size([6, config.d_model])
