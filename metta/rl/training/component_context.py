"""Shared context object passed to trainer components."""

from __future__ import annotations

import dataclasses
import typing

import torch.optim

import metta.agent.policy
import metta.cogworks.curriculum.curriculum
import metta.eval.eval_request_config
import metta.rl.training.distributed_helper as training_distributed_helper
import metta.rl.training.experience as training_experience
import metta.rl.training.stats_reporter as training_stats_reporter
import metta.rl.training.training_environment as training_environment
import mettagrid.profiling.memory_monitor
import mettagrid.profiling.stopwatch
import mettagrid.profiling.system_monitor

if typing.TYPE_CHECKING:
    import metta.cogworks.curriculum


@dataclasses.dataclass(slots=True)
class TrainingEnvWindow:
    """Serializable view of the environment slice used for training."""

    start: int
    stop: int
    step: int = 1

    def to_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    @classmethod
    def from_slice(cls, window: slice) -> "TrainingEnvWindow":
        start = 0 if window.start is None else int(window.start)
        stop = 0 if window.stop is None else int(window.stop)
        step = 1 if window.step is None else int(window.step)
        return cls(start=start, stop=stop, step=step)


@dataclasses.dataclass(slots=True)
class TrainerState:
    """Serializable trainer state that can be checkpointed."""

    epoch: int = 0
    agent_step: int = 0
    latest_policy_uri: typing.Optional[str] = None
    latest_eval_scores: typing.Optional[metta.eval.eval_request_config.EvalRewardSummary] = None
    latest_losses_stats: typing.Dict[str, float] = dataclasses.field(default_factory=dict)
    gradient_stats: typing.Dict[str, float] = dataclasses.field(default_factory=dict)
    training_env_window: typing.Optional[TrainingEnvWindow] = None
    optimizer_state: typing.Optional[typing.Dict[str, typing.Any]] = None
    stopwatch_state: typing.Optional[typing.Dict[str, typing.Any]] = None
    curriculum_state: typing.Optional[typing.Dict[str, typing.Any]] = None
    latest_saved_policy_epoch: int = 0
    loss_states: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)


class ComponentContext:
    """Aggregated view of trainer state and runtime dependencies."""

    def __init__(
        self,
        *,
        state: typing.Optional[TrainerState],
        policy: metta.agent.policy.Policy,
        env: training_environment.TrainingEnvironment,
        experience: training_experience.Experience,
        optimizer: torch.optim.Optimizer,
        config: typing.Any,
        stopwatch: mettagrid.profiling.stopwatch.Stopwatch,
        distributed: training_distributed_helper.DistributedHelper,
        run_name: typing.Optional[str] = None,
        curriculum: typing.Optional[metta.cogworks.curriculum.curriculum.Curriculum] = None,
    ) -> None:
        self.state = state or TrainerState()
        self.policy = policy
        self.env = env
        self.experience = experience
        self.optimizer = optimizer
        self.config = config
        self.stopwatch = stopwatch
        self.distributed = distributed
        self.run_name = run_name
        self.curriculum = curriculum

        self.timing_baseline = {"agent_step": 0, "wall_time": 0.0}

        self.stats_reporter: training_stats_reporter.StatsReporter | None = None
        self.memory_monitor: mettagrid.profiling.memory_monitor.MemoryMonitor | None = None
        self.system_monitor: mettagrid.profiling.system_monitor.SystemMonitor | None = None
        self.latest_policy_uri_fn: typing.Callable[[], typing.Optional[str]] | None = None
        self.losses: typing.Dict[str, typing.Any] = {}

        self.get_train_epoch_fn: typing.Callable[[], typing.Callable[[], None]] | None = None
        self.set_train_epoch_fn: typing.Callable[[typing.Callable[[], None]], None] | None = None

        self._training_env_id: slice | None = (
            self.state.training_env_window.to_slice() if self.state.training_env_window else None
        )

    # ------------------------------------------------------------------
    # Epoch / step tracking
    # ------------------------------------------------------------------
    @property
    def epoch(self) -> int:
        return self.state.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.state.epoch = value

    @property
    def agent_step(self) -> int:
        return self.state.agent_step

    @agent_step.setter
    def agent_step(self, value: int) -> None:
        self.state.agent_step = value

    # ------------------------------------------------------------------
    # Training environment window
    # ------------------------------------------------------------------
    @property
    def training_env_id(self) -> slice | None:
        return self._training_env_id

    @training_env_id.setter
    def training_env_id(self, value: slice | None) -> None:
        self._training_env_id = value
        if value is None:
            self.state.training_env_window = None
        else:
            self.state.training_env_window = TrainingEnvWindow.from_slice(value)

    # ------------------------------------------------------------------
    # Latest policy helpers
    # ------------------------------------------------------------------
    @property
    def latest_policy_uri_value(self) -> typing.Optional[str]:
        return self.state.latest_policy_uri

    @latest_policy_uri_value.setter
    def latest_policy_uri_value(self, value: typing.Optional[str]) -> None:
        self.state.latest_policy_uri = value

    def latest_policy_uri(self) -> typing.Optional[str]:
        if self.state.latest_policy_uri:
            return self.state.latest_policy_uri
        if self.latest_policy_uri_fn is None:
            return None
        uri = self.latest_policy_uri_fn()
        self.state.latest_policy_uri = uri
        return uri

    @property
    def latest_saved_policy_epoch(self) -> int:
        return self.state.latest_saved_policy_epoch

    @latest_saved_policy_epoch.setter
    def latest_saved_policy_epoch(self, value: int) -> None:
        self.state.latest_saved_policy_epoch = value

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------
    @property
    def latest_eval_scores(self) -> typing.Optional[metta.eval.eval_request_config.EvalRewardSummary]:
        return self.state.latest_eval_scores

    @latest_eval_scores.setter
    def latest_eval_scores(self, value: typing.Optional[metta.eval.eval_request_config.EvalRewardSummary]) -> None:
        self.state.latest_eval_scores = value

    @property
    def latest_losses_stats(self) -> typing.Dict[str, float]:
        return self.state.latest_losses_stats

    @latest_losses_stats.setter
    def latest_losses_stats(self, value: typing.Dict[str, float]) -> None:
        self.state.latest_losses_stats = dict(value)

    @property
    def gradient_stats(self) -> typing.Dict[str, float]:
        return self.state.gradient_stats

    @gradient_stats.setter
    def gradient_stats(self, value: typing.Dict[str, float]) -> None:
        self.state.gradient_stats = dict(value)

    def update_gradient_stats(self, stats: typing.Dict[str, float]) -> None:
        self.gradient_stats = stats

    # ------------------------------------------------------------------
    # Epoch lifecycle helpers
    # ------------------------------------------------------------------
    def reset_for_epoch(self) -> None:
        self.training_env_id = None

    def record_rollout(self, agent_steps: int, world_size: int) -> None:
        self.agent_step += agent_steps * world_size

    def advance_epoch(self, epochs: int) -> None:
        self.epoch += epochs

    # ------------------------------------------------------------------
    # Training epoch callable indirection
    # ------------------------------------------------------------------
    def get_train_epoch_callable(self) -> typing.Callable[[], None]:
        if self.get_train_epoch_fn is None:
            raise RuntimeError("ComponentContext has no getter for train epoch callable")
        return self.get_train_epoch_fn()

    def set_train_epoch_callable(self, fn: typing.Callable[[], None]) -> None:
        if self.set_train_epoch_fn is None:
            raise RuntimeError("ComponentContext has no setter for train epoch callable")
        self.set_train_epoch_fn(fn)
