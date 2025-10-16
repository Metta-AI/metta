from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from metta.agent.policies.fast import FastConfig
from metta.cogworks.curriculum import env_curriculum
from metta.rl.system_config import SystemConfig, seed_everything
from metta.rl.training.training_environment import TrainingEnvironmentConfig, VectorizedTrainingEnvironment
from mettagrid.builder.envs import make_arena


def _build_env_and_policy(
    factory: Callable[[], TrainingEnvironmentConfig],
    data_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    system_cfg = SystemConfig(
        device="cpu",
        vectorization="serial",
        seed=42,
        torch_deterministic=True,
        local_only=True,
        data_dir=data_dir,
    )
    seed_everything(system_cfg)

    env_cfg = factory()
    env = VectorizedTrainingEnvironment(env_cfg)
    try:
        policy_cfg = FastConfig()
        policy = policy_cfg.make_policy(env.meta_data)

        state_dict = {name: param.detach().cpu().clone() for name, param in policy.state_dict().items()}
        metadata: dict[str, Any] = {
            "obs_width": env.meta_data.obs_width,
            "obs_height": env.meta_data.obs_height,
            "obs_features": env.meta_data.obs_features,
            "action_names": env.meta_data.action_names,
            "num_agents": env.meta_data.num_agents,
            "feature_normalizations": env.meta_data.feature_normalizations,
        }
    finally:
        env.close()

    return state_dict, metadata


def test_same_seed_gives_same_policy_and_environment(tmp_path: Path) -> None:
    def env_factory() -> TrainingEnvironmentConfig:
        return TrainingEnvironmentConfig(
            curriculum=env_curriculum(make_arena(num_agents=24)),
            num_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=8,
            vectorization="serial",
            seed=987,
        )

    first_state, first_metadata = _build_env_and_policy(env_factory, tmp_path / "run1")
    second_state, second_metadata = _build_env_and_policy(env_factory, tmp_path / "run2")

    assert first_metadata == second_metadata
    for name, tensor in first_state.items():
        assert torch.equal(tensor, second_state[name]), f"Parameter {name} differed despite identical seeds"
