import numpy as np

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def test_policy_env_interface_round_trip_serialization():
    config = MettaGridConfig(
        game=GameConfig(
            num_agents=4,
            obs=ObsConfig(width=5, height=5, num_tokens=100),
            max_steps=100,
            resource_names=["ore", "wood"],
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects={"wall": WallConfig()},
            map_builder=RandomMapBuilder.Config(width=10, height=10, agents=4, seed=42),
        )
    )

    original = PolicyEnvInterface.from_mg_cfg(config)
    dumped = original.model_dump(mode="json")
    restored = PolicyEnvInterface.model_validate(dumped)

    assert restored.num_agents == original.num_agents
    assert restored.obs_width == original.obs_width
    assert restored.obs_height == original.obs_height
    assert restored.tags == original.tags
    assert restored.tag_id_to_name == original.tag_id_to_name

    assert restored.observation_space.shape == original.observation_space.shape
    assert restored.observation_space.dtype == original.observation_space.dtype
    assert np.array_equal(restored.observation_space.low, original.observation_space.low)
    assert np.array_equal(restored.observation_space.high, original.observation_space.high)

    assert restored.action_space.n == original.action_space.n
    assert restored.action_space.start == original.action_space.start

    assert len(restored.obs_features) == len(original.obs_features)
    for r, o in zip(restored.obs_features, original.obs_features, strict=True):
        assert r.id == o.id
        assert r.name == o.name
        assert r.normalization == o.normalization

    assert restored.action_names == original.action_names
