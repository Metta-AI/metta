"""Shared trainer context passed to training components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar

import torch

from metta.agent.policy import Policy
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid.profiling.memory_monitor import MemoryMonitor
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.mettagrid.profiling.system_monitor import SystemMonitor
from metta.rl.training.distributed_helper import DistributedHelper
from metta.rl.training.experience import Experience
from metta.rl.training.training_environment import TrainingEnvironment

if TYPE_CHECKING:
    from metta.rl.trainer import Trainer
    from metta.rl.trainer_config import TrainerConfig
    from metta.rl.training.component import TrainerComponent
    from metta.rl.training.stats_reporter import StatsReporter

T_Component = TypeVar("T_Component")


@dataclass
class TrainerContext:
    """Aggregated view of trainer state exposed to components."""

    trainer: Trainer
    policy: Policy
    env: TrainingEnvironment
    experience: Experience
    optimizer: torch.optim.Optimizer
    config: "TrainerConfig"
    device: torch.device
    stopwatch: Stopwatch
    distributed: DistributedHelper
    run_dir: Optional[Path]
    run_name: Optional[str]
    latest_policy_uri_fn: Callable[[], Optional[str]] | None = None
    latest_policy_uri_value: Optional[str] = None
    latest_eval_scores: Optional[EvalRewardSummary] = None
    latest_losses_stats: Dict[str, float] = field(default_factory=dict)
    save_policy_fn: Callable[[dict[str, Any], bool], Optional[str]] | None = None
    save_trainer_state_fn: Callable[[], None] | None = None
    checkpoint_manager: Any | None = None
    stats_client: Any | None = None
    stats_reporter: "StatsReporter" | None = None
    get_train_epoch_fn: Callable[[], Callable[[], None]] | None = None
    set_train_epoch_fn: Callable[[Callable[[], None]], None] | None = None
    components: Dict[type, TrainerComponent] = field(default_factory=dict)
    gradient_stats: Dict[str, float] = field(default_factory=dict)
    memory_monitor: MemoryMonitor | None = None
    system_monitor: SystemMonitor | None = None
    _epoch: int = 0
    _agent_step: int = 0
    update_epoch: int = 0
    mb_idx: int = 0
    training_env_id: slice | None = None
    stop_rollout: bool = False
    stop_update_epoch: bool = False

    @property
    def epoch(self) -> int:
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._epoch = value

    @property
    def agent_step(self) -> int:
        return self._agent_step

    @agent_step.setter
    def agent_step(self, value: int) -> None:
        self._agent_step = value

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
        if self.latest_policy_uri_value:
            return self.latest_policy_uri_value
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

    def get_train_epoch_callable(self) -> Callable[[], None]:
        if self.get_train_epoch_fn is None:
            raise RuntimeError("TrainerContext has no getter for train epoch callable")
        return self.get_train_epoch_fn()

    def set_train_epoch_callable(self, fn: Callable[[], None]) -> None:
        if self.set_train_epoch_fn is None:
            raise RuntimeError("TrainerContext has no setter for train epoch callable")
        self.set_train_epoch_fn(fn)
