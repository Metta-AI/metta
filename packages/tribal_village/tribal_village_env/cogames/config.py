"""Configuration helpers for Tribal Village training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from mettagrid.policy.policy import PolicySpec

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TribalTrainRequest:
    """CLI-friendly inputs with sensible defaults."""

    checkpoints_path: Path = Path("./train_dir")
    steps: int = 10_000_000
    seed: int = 42
    batch_size: int = 4096
    minibatch_size: int = 4096
    num_workers: Optional[int] = None
    parallel_envs: Optional[int] = 64
    vector_batch_size: Optional[int] = None
    max_steps: int = 1000
    render_scale: int = 1
    render_mode: Literal["ansi", "rgb_array"] = "ansi"
    log_outputs: bool = False

    def make_env_config(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "render_scale": self.render_scale,
            "render_mode": self.render_mode,
        }


@dataclass
class TribalTrainSettings:
    """Resolved trainer settings used by the runtime."""

    policy_class_path: str
    device: "torch.device"
    checkpoints_path: Path
    steps: int
    seed: int
    batch_size: int
    minibatch_size: int
    vector_num_envs: Optional[int]
    vector_batch_size: Optional[int]
    vector_num_workers: Optional[int]
    log_outputs: bool
    env_config: dict[str, Any]
    initial_weights_path: Optional[str]

    @classmethod
    def from_request(
        cls,
        *,
        request: TribalTrainRequest,
        policy_spec: PolicySpec,
        torch_device: "torch.device",
    ) -> "TribalTrainSettings":
        return cls(
            policy_class_path=policy_spec.class_path,
            device=torch_device,
            checkpoints_path=request.checkpoints_path,
            steps=request.steps,
            seed=request.seed,
            batch_size=request.batch_size,
            minibatch_size=request.minibatch_size,
            vector_num_envs=request.parallel_envs,
            vector_batch_size=request.vector_batch_size,
            vector_num_workers=request.num_workers,
            log_outputs=request.log_outputs,
            env_config=request.make_env_config(),
            initial_weights_path=policy_spec.data_path,
        )


__all__ = ["TribalTrainRequest", "TribalTrainSettings"]
