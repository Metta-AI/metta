from __future__ import annotations

import argparse

import torch
from tensordict import TensorDict

from metta.agent.policies.vit import ViTDefaultConfig
from metta.rl.utils import ensure_sequence_metadata
from mettagrid.config import MettaGridConfig
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


def _build_policy(pattern: str, device: torch.device) -> torch.nn.Module:
    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
    config = ViTDefaultConfig(core_resnet_pattern=pattern)
    policy = config.make_policy(policy_env_info)
    policy.initialize_to_environment(policy_env_info, device)
    return policy


def _make_dummy_td(policy_env_info: PolicyEnvInterface, device: torch.device) -> TensorDict:
    obs_shape = policy_env_info.observation_space.shape
    env_obs = torch.zeros((1, *obs_shape), dtype=torch.uint8, device=device)
    if env_obs.shape[1] > 0:
        # coord=255 is padding; keep one token non-padding
        env_obs[0, 0, 0] = 0
    td = TensorDict({"env_obs": env_obs}, batch_size=[1])
    ensure_sequence_metadata(td, batch_size=1, time_steps=1)
    return td


def run_test(pattern: str, device: torch.device) -> None:
    policy_env_info = PolicyEnvInterface.from_mg_cfg(MettaGridConfig())
    policy = _build_policy(pattern, device)
    td = _make_dummy_td(policy_env_info, device)
    policy(td)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test ViTDefaultConfig with a Cortex pattern.")
    parser.add_argument("--pattern", default="Ag,S,A", help="Cortex pattern string to test.")
    parser.add_argument("--device", default="cpu", help="Device to run on (e.g. cpu or cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_test(args.pattern, device)
    print(f"ViT pattern '{args.pattern}' forward pass OK on {device}.")


if __name__ == "__main__":
    main()
