"""Shared trainer context passed to training components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar

import torch

from metta.agent.policy import Policy
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState
from metta.rl.training.distributed_helper import DistributedHelper
from metta.rl.training.experience import Experience
from metta.rl.training.training_environment import TrainingEnvironment

if TYPE_CHECKING:
    from metta.rl.trainer import Trainer
    from metta.rl.training.component import TrainerComponent

T_Component = TypeVar("T_Component")


@dataclass
class TrainerContext:
    """Aggregated view of trainer state exposed to components."""

    trainer: Trainer
    policy: Policy
    env: TrainingEnvironment
    experience: Experience
    optimizer: torch.optim.Optimizer
    config: TrainerConfig
    device: torch.device
    stopwatch: Stopwatch
    distributed: DistributedHelper
    trainer_state: TrainerState
    run_dir: Optional[Path]
    run_name: Optional[str]
    get_epoch: Callable[[], int]
    get_agent_step: Callable[[], int]
    latest_policy_uri_fn: Callable[[], Optional[str]] | None = None
    save_policy_fn: Callable[[dict[str, Any], bool], Optional[str]] | None = None
    save_trainer_state_fn: Callable[[], None] | None = None
    checkpoint_manager: Any | None = None
    stats_client: Any | None = None
    components: Dict[type, TrainerComponent] = field(default_factory=dict)
    gradient_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def epoch(self) -> int:
        return self.get_epoch()

    @property
    def agent_step(self) -> int:
        return self.get_agent_step()

    def set_run_info(self, *, run_dir: Optional[Path], run_name: Optional[str]) -> None:
        self.run_dir = run_dir
        self.run_name = run_name

    def register_component(self, component: TrainerComponent) -> None:
        self.components[type(component)] = component

    def get_component(self, component_type: Type[T_Component]) -> Optional[T_Component]:
        for registered in self.components.values():
            if isinstance(registered, component_type):
                return registered  # type: ignore[return-value]
        return None

    def update_gradient_stats(self, stats: Dict[str, float]) -> None:
        self.gradient_stats = stats

    def latest_policy_uri(self) -> Optional[str]:
        if self.latest_policy_uri_fn is None:
            return None
        return self.latest_policy_uri_fn()

    def save_policy(self, metadata: dict[str, Any], *, final: bool = False) -> Optional[str]:
        if self.save_policy_fn is None:
            return None
        return self.save_policy_fn(metadata, final)

    def save_trainer_state(self) -> None:
        if self.save_trainer_state_fn is not None:
            self.save_trainer_state_fn()
