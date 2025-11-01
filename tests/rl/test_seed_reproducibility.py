from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from metta.agent.policies.fast import FastConfig
from metta.cogworks.curriculum import CurriculumConfig, SingleTaskGenerator, env_curriculum
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
        game_rules = env.game_rules
        policy_cfg = FastConfig()
        policy = policy_cfg.make_policy(game_rules)

        state_dict = {name: param.detach().cpu().clone() for name, param in policy.state_dict().items()}
        metadata: dict[str, Any] = {
            "obs_width": game_rules.obs_width,
            "obs_height": game_rules.obs_height,
            "obs_features": game_rules.obs_features,
            "action_names": game_rules.action_names,
            "num_agents": game_rules.num_agents,
            "feature_normalizations": game_rules.feature_normalizations,
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


def test_training_environment_seed_propagates_to_curriculum() -> None:
    base_env_cfg = make_arena(num_agents=12)

    def sample_rng_sequence(seed: int) -> tuple[float, ...]:
        curriculum_cfg = CurriculumConfig(
            task_generator=SingleTaskGenerator.Config(env=base_env_cfg.model_copy(deep=True)),
            num_active_tasks=8,
            max_task_id=128,
        )
        env_cfg = TrainingEnvironmentConfig(
            curriculum=curriculum_cfg,
            num_workers=1,
            async_factor=1,
            forward_pass_minibatch_target_size=8,
            vectorization="serial",
            seed=seed,
        )
        env = VectorizedTrainingEnvironment(env_cfg)
        try:
            # Inspect curriculum RNG directly to verify the seed wiring.
            return tuple(env._curriculum._rng.random() for _ in range(5))
        finally:
            env.close()

    first = sample_rng_sequence(101)
    second = sample_rng_sequence(101)
    third = sample_rng_sequence(202)

    assert first == second
    assert first != third
