"""Experiment orchestration with state machines and simple inheritance.

This module provides a clean base controller (SweepOrchestrator) that manages
universal experiment state and trial lifecycles using explicit state machines.
Concrete strategies inherit and implement domain-specific logic.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional

from metta.common.util.constants import SOFTMAX_S3_POLICY_PREFIX
from metta.sweep.config import SweepOrchestratorConfig
from metta.sweep.models import JobDefinition, JobStatus
from metta.sweep.protocols import Dispatcher, Store

logger = logging.getLogger(__name__)


# ============= Trial State Machine =============


class TrialState(Enum):
    """All possible states for a trial in its lifecycle."""

    SUGGESTED = auto()  # Created by optimizer, not yet dispatched
    PENDING = auto()  # Dispatched but waiting for resources
    TRAINING = auto()  # Currently training
    AWAITING_EVALUATION = auto()  # Training done, eval needs to be dispatched
    EVALUATING = auto()  # Running evaluation
    COMPLETED = auto()  # Successfully finished
    FAILED = auto()  # Failed during execution


@dataclass
class StateTransition:
    """Record of a state transition with metadata."""

    from_state: TrialState
    to_state: TrialState
    timestamp: datetime
    reason: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Observation:
    """Observation from a completed trial."""

    score: float
    cost: float = 0.0
    metrics: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TransitionContext:
    """Context provided to transition hooks for side effects."""

    trial: "Trial"
    orchestrator: "SweepOrchestrator"
    transition: StateTransition
    store: Any  # Store protocol
    dispatcher: Any  # Dispatcher protocol
    extra: dict = field(default_factory=dict)  # Additional data

    @property
    def from_state(self) -> TrialState:
        """Source state of the transition."""
        return self.transition.from_state

    @property
    def to_state(self) -> TrialState:
        """Target state of the transition."""
        return self.transition.to_state

    @property
    def trial_id(self) -> str:
        """Convenience accessor for trial ID."""
        return self.trial.id


@dataclass
class TransitionHook:
    """Defines a side effect to execute on state transitions.

    Used for infrastructure concerns like logging, metrics, network calls.
    NOT for core optimization logic.
    """

    from_state: Optional[TrialState]  # None = any state
    to_state: Optional[TrialState]  # None = any state
    action: Callable[[TransitionContext], None]
    name: str = ""
    priority: int = 0  # Higher runs first

    def matches(self, from_state: TrialState, to_state: TrialState) -> bool:
        """Check if this hook applies to a transition."""
        from_match = self.from_state is None or self.from_state == from_state
        to_match = self.to_state is None or self.to_state == to_state
        return from_match and to_match


@dataclass
class Trial:
    """A trial with explicit state machine and metadata."""

    id: str
    params: dict
    state: TrialState = TrialState.SUGGESTED

    # Optional fields
    budget: Optional[float] = None
    parent_id: Optional[str] = None  # For warm starts/promotions
    checkpoint_uri: Optional[str] = None

    # State tracking
    history: list[StateTransition] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize valid state transitions."""
        self.transitions = {
            TrialState.SUGGESTED: [TrialState.PENDING],
            TrialState.PENDING: [TrialState.TRAINING, TrialState.FAILED],
            TrialState.TRAINING: [TrialState.AWAITING_EVALUATION, TrialState.COMPLETED, TrialState.FAILED],
            TrialState.AWAITING_EVALUATION: [TrialState.EVALUATING, TrialState.FAILED],
            TrialState.EVALUATING: [TrialState.COMPLETED, TrialState.FAILED],
            # Terminal states
            TrialState.COMPLETED: [],
            TrialState.FAILED: [],
        }

    def can_transition_to(self, new_state: TrialState) -> bool:
        """Check if transition to new_state is valid."""
        return new_state in self.transitions.get(self.state, [])

    def transition_to(self, new_state: TrialState, reason: str = "", **metadata) -> StateTransition:
        """Execute state transition with validation and recording.

        Args:
            new_state: Target state to transition to
            reason: Human-readable reason for transition
            **metadata: Additional metadata to record with transition

        Returns:
            StateTransition record

        Raises:
            ValueError: If transition is invalid
        """
        if not self.can_transition_to(new_state):
            valid_states = self.transitions.get(self.state, [])
            raise ValueError(
                f"Invalid transition: {self.state.name} -> {new_state.name}. "
                f"Valid transitions: {[s.name for s in valid_states]}"
            )

        # Record transition
        transition = StateTransition(
            from_state=self.state, to_state=new_state, timestamp=datetime.now(), reason=reason, metadata=metadata
        )
        self.history.append(transition)

        # Update state
        self.state = new_state

        # Update timing information
        if new_state == TrialState.TRAINING:
            self.started_at = datetime.now()
        elif new_state in (TrialState.COMPLETED, TrialState.FAILED):
            self.completed_at = datetime.now()

        # Store metadata from transition
        if metadata:
            self.metadata.update(metadata)

        return transition

    @property
    def is_terminal(self) -> bool:
        """Check if trial is in a terminal state."""
        return self.state in (TrialState.COMPLETED, TrialState.FAILED)

    @property
    def is_active(self) -> bool:
        """Check if trial is actively using resources."""
        return self.state in (TrialState.TRAINING, TrialState.EVALUATING)

    @property
    def duration(self) -> Optional[float]:
        """Calculate trial duration in seconds."""
        if self.started_at is None:
            return None

        end_time = self.completed_at if self.completed_at else datetime.now()
        return (end_time - self.started_at).total_seconds()

    def get_state_history(self) -> str:
        """Get a formatted string of the trial's state history.

        Useful for debugging and understanding trial lifecycle.

        Returns:
            Formatted string showing state transitions
        """
        lines = [f"Trial {self.id} State History:"]
        lines.append(f"  Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Initial: {TrialState.SUGGESTED.name}")

        for transition in self.history:
            time_str = transition.timestamp.strftime("%H:%M:%S")
            lines.append(f"  {time_str}: {transition.from_state.name} -> {transition.to_state.name}")
            if transition.reason:
                lines.append(f"    Reason: {transition.reason}")
            if transition.metadata:
                lines.append(f"    Metadata: {transition.metadata}")

        lines.append(f"  Current: {self.state.name}")
        if self.duration:
            lines.append(f"  Duration: {self.duration:.1f}s")

        return "\n".join(lines)


class SweepOrchestrator(ABC):
    """Base orchestrator that manages universal state and experiment flow.

    Subclasses implement strategy-specific logic via hooks while inheriting
    common state management, trial lifecycle, and resource coordination.
    """

    def __init__(
        self,
        experiment_id: str,
        dispatcher: Dispatcher,
        store: Store,
        config: "SweepOrchestratorConfig",
    ):
        """Initialize orchestrator with core dependencies.

        Args:
            experiment_id: Unique identifier for this experiment
            dispatcher: Dispatcher for job execution
            store: Store for run data (e.g., WandB)
            config: Configuration for sweep orchestration
        """
        self.experiment_id = experiment_id
        self.dispatcher = dispatcher
        self.store = store

        # Configuration
        self.max_parallel = config.max_parallel
        self.poll_interval = config.poll_interval
        self.initial_wait = config.initial_wait
        self.metric_key = config.metric_key
        self.cost_key = config.cost_key
        self.skip_evaluation = config.skip_evaluation
        self.stop_on_error = config.stop_on_error
        self.resume = config.resume

        # Universal state - all strategies need these
        self.trials: dict[str, Trial] = {}
        self.observations: dict[str, Observation] = {}
        self.running: set[str] = set()
        self.completed: set[str] = set()
        self.failed: set[str] = set()

        # Metrics
        self.best_score = float("-inf")
        self.total_cost = 0.0
        self.start_time = datetime.now()

        # Hook system for infrastructure side effects
        self.transition_hooks: list[TransitionHook] = []
        self._register_infrastructure_hooks()

        # Let subclass initialize strategy-specific state
        self.setup()

    # ============= Main Orchestration Loop =============

    def run(self) -> None:
        """Main experiment loop - coordinates all operations.

        This is the core loop that:
        1. Syncs state from external sources
        2. Processes completed runs
        3. Generates new trials
        4. Dispatches trials
        5. Checks stopping conditions
        """
        logger.info(f"Starting experiment {self.experiment_id}")

        # If resuming, sync state immediately to recover existing trials
        if self.resume:
            logger.info("Resuming from existing state...")
            self._recover_trials_from_store()
            logger.info(f"Recovered {len(self.trials)} existing trials")
        else:
            # Initial wait to let any existing runs populate
            time.sleep(self.initial_wait)

        while not self.should_stop():
            try:
                # Step 1: Sync state from external store (WandB, etc)
                self._sync_state()

                # Step 2: Check for and dispatch new trials
                self._dispatch_new_trials()

                # Step 3: Sleep before next iteration
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("Experiment interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                # Continue running unless it's a critical error
                if self.stop_on_error:
                    break
                time.sleep(self.poll_interval)

        cost_info = f" (metric: '{self.cost_key}')" if self.cost_key else ""
        logger.info(
            f"Experiment {self.experiment_id} complete. "
            f"Completed: {self.n_completed}, Failed: {self.n_failed}, "
            f"Best {self.metric_key}: {self.best_score:.4f}, "
            f"Total cost: {self.total_cost:.2f}{cost_info}"
        )

    def _recover_trials_from_store(self) -> None:
        """Recover trial state from store when resuming a sweep.

        This reconstructs Trial objects from existing runs in WandB.
        """
        runs = self.store.fetch_runs(filters={"group": self.experiment_id})

        for run in runs:
            trial_id = run.run_id

            # Skip if we already have this trial
            if trial_id in self.trials:
                continue

            # Reconstruct trial from run data
            params = {}
            if run.summary:
                # Extract parameters from sweep/suggestion if available
                params = run.summary.get("sweep/suggestion", {})

            # Create trial object
            trial = Trial(id=trial_id, params=params)

            # Determine the most accurate state, preferring active states before terminal ones.
            if run.has_failed:
                trial.state = TrialState.FAILED
                self.failed.add(trial_id)
            elif run.has_started_eval and not run.has_been_evaluated:
                # Evaluation already dispatched and still running
                trial.state = TrialState.EVALUATING
                self.running.add(trial_id)
            elif run.has_started_training and not run.has_completed_training:
                trial.state = TrialState.TRAINING
                self.running.add(trial_id)
            elif run.has_completed_training and not self.skip_evaluation and not run.has_been_evaluated:
                trial.state = TrialState.AWAITING_EVALUATION
            elif run.has_been_evaluated or (self.skip_evaluation and run.has_completed_training):
                trial.state = TrialState.COMPLETED
                self.completed.add(trial_id)

                # Extract observation if available
                if run.summary and self.metric_key in run.summary:
                    score = float(run.summary[self.metric_key])

                    # Extract cost using cost_key if provided
                    if self.cost_key and self.cost_key in run.summary:
                        cost = float(run.summary[self.cost_key])
                    else:
                        # Legacy fallback
                        cost = float(run.cost if hasattr(run, "cost") else run.summary.get("cost", 0.0))

                    observation = Observation(score=score, cost=cost, metrics=run.summary.copy())
                    self.observations[trial_id] = observation

                    # Update best score and total cost
                    if score > self.best_score:
                        self.best_score = score
                    self.total_cost += cost
            else:
                trial.state = TrialState.PENDING

            # Register the recovered trial
            self.trials[trial_id] = trial

            logger.info(f"Recovered trial {trial_id} in state {trial.state.name}")

    def _sync_state(self) -> None:
        """Synchronize state with external store (WandB, etc).

        Fetches runs from the store and updates local state accordingly.
        """
        # Fetch runs from store
        runs = self.store.fetch_runs(filters={"group": self.experiment_id})

        # Display monitoring table if we have runs
        if runs:
            from metta.sweep.utils import make_monitor_table

            table_lines = make_monitor_table(
                runs=runs,
                title="Sweep Status",
                logger_prefix="[SweepOrchestrator]",
                include_score=True,
                truncate_run_id=True,
            )
            for line in table_lines:
                logger.info(line)

        # Process each run
        for run in runs:
            trial_id = run.run_id  # RunInfo uses run_id, not id

            # Skip if we don't know about this trial
            if trial_id not in self.trials:
                # This could be a run from a previous session or manually dispatched
                # Only log as debug to avoid spam
                logger.debug(f"Unknown trial {trial_id} in store, skipping")
                continue

            trial = self.trials[trial_id]

            # Update trial state based on run status
            # Check if run has failed
            if run.has_failed:
                if trial_id not in self.failed:
                    self._update_trial_state(trial_id, TrialState.FAILED, reason="Run failed")
                    self.failed.add(trial_id)
                    self.running.discard(trial_id)
                    self.on_trial_failed(trial, getattr(run, "error", "Unknown error"))
                continue

            # Check if run is stale (no updates for 20+ minutes)
            if hasattr(run, "status") and run.status == JobStatus.STALE:
                if trial_id not in self.failed:
                    logger.warning(f"Trial {trial_id} is STALE (no updates for 20+ minutes), marking as failed")
                    self._update_trial_state(
                        trial_id, TrialState.FAILED, reason="Run stale - no updates for 20+ minutes"
                    )
                    self.failed.add(trial_id)
                    self.running.discard(trial_id)
                    self.on_trial_failed(trial, "Job stale - no updates for 20+ minutes")
                continue

            # Check state transitions (not mutually exclusive - a run can have multiple states true)

            # First, check if training has started (PENDING → TRAINING)
            if run.has_started_training and trial.state == TrialState.PENDING:
                self._update_trial_state(trial_id, TrialState.TRAINING, reason="Started training")

            # Check if training is complete (TRAINING → AWAITING_EVALUATION)
            if run.has_completed_training and trial.state == TrialState.TRAINING:
                if trial_id not in self.completed and trial_id not in self.failed:
                    # Check if evaluation is needed or if we can mark complete
                    if self.skip_evaluation:
                        # No separate evaluation, training is complete
                        self._process_completed_run(run, trial)
                    else:
                        # Training finished, transition to awaiting evaluation
                        # The checkpoint URI will be set by the hook
                        self._update_trial_state(
                            trial_id, TrialState.AWAITING_EVALUATION, reason="Training finished, awaiting evaluation"
                        )

            # Check if evaluation is complete (EVALUATING or AWAITING_EVALUATION → COMPLETED)
            # We check multiple states because evaluation might complete before we mark it as EVALUATING
            if run.has_been_evaluated:
                if trial_id not in self.completed:
                    # If still awaiting evaluation, first transition to EVALUATING
                    if trial.state == TrialState.AWAITING_EVALUATION:
                        self._update_trial_state(
                            trial_id, TrialState.EVALUATING, reason="Evaluation completed (fast completion)"
                        )

                    # Now process completion if in EVALUATING state
                    if trial.state == TrialState.EVALUATING:
                        # Evaluation complete
                        self._process_completed_run(run, trial)

    def _process_completed_run(self, run: Any, trial: Trial) -> None:
        """Process a completed run and extract observation.

        Args:
            run: Run object from the store
            trial: Corresponding trial object
        """
        trial_id = trial.id
        summary = run.summary or {}

        # Check if the metric exists - fail hard if not found
        if self.metric_key not in summary:
            error_msg = (
                f"CRITICAL: Metric '{self.metric_key}' not found in run {run.run_id} summary. "
                f"The sweep cannot optimize without this metric. Please verify your evaluation "
                f"is producing the expected metric."
            )
            logger.error(error_msg)
            raise KeyError(error_msg)

        # Extract observation from run
        score = float(summary[self.metric_key])

        # Extract cost using cost_key if provided, otherwise default to 0
        if self.cost_key:
            if self.cost_key not in summary:
                logger.warning(
                    f"Cost metric '{self.cost_key}' not found in run {run.run_id} summary. "
                    f"Using 0 as cost. Available keys: {list(summary.keys())}"
                )
                cost = 0.0
            else:
                cost = float(summary[self.cost_key])
        else:
            # Legacy fallback: check for 'cost' field or run.cost attribute
            cost = float(run.cost if hasattr(run, "cost") else summary.get("cost", 0.0))

        metrics = summary.copy()

        observation = Observation(score=score, cost=cost, metrics=metrics)

        # Update trial state to completed and pass observation through context
        # The WandB update will be handled by the hook
        self._update_trial_state(
            trial_id, TrialState.COMPLETED, reason="Evaluation complete", score=score, observation=observation
        )  # Pass observation for hook

        # Record observation
        self._record_observation(trial_id, observation)

        # Update sets
        self.completed.add(trial_id)
        self.running.discard(trial_id)

        # Call optimization logic hook
        self.on_trial_completed(trial, observation)

        # Check if we should checkpoint (optimization decision, not infrastructure)
        if self.should_checkpoint(trial, observation):
            trial.checkpoint_uri = f"{SOFTMAX_S3_POLICY_PREFIX}/{trial_id}:latest"
            logger.info(f"Checkpointed trial {trial_id} with score {score:.4f} at {trial.checkpoint_uri}")

    def _dispatch_new_trials(self) -> None:
        """Generate and dispatch new trials based on available resources."""
        # First, check for any trials awaiting evaluation
        awaiting_eval = [t for t in self.trials.values() if t.state == TrialState.AWAITING_EVALUATION]

        if awaiting_eval:
            logger.info(f"Dispatching evaluation for {len(awaiting_eval)} trials")
            for trial in awaiting_eval:
                # Convert trial to evaluation job
                job = self._trial_to_job(trial)

                # Dispatch evaluation
                dispatch_id = self.dispatcher.dispatch(job)

                # Mark evaluation as started in WandB
                try:
                    self.store.update_run_summary(trial.id, {"has_started_eval": True})
                except Exception as e:
                    logger.warning(f"Failed to mark eval started for {trial.id}: {e}")

                # Update trial state to EVALUATING
                self._update_trial_state(
                    trial.id, TrialState.EVALUATING, reason="Evaluation dispatched", dispatch_id=dispatch_id
                )

                logger.info(f"Dispatched evaluation for trial {trial.id}")

        # Then handle new training trials if we have slots
        # Calculate available slots
        available_slots = self._calculate_available_slots()

        if available_slots <= 0:
            return

        # Generate new trials
        new_trials = self.suggest_trials(available_slots)

        if not new_trials:
            return

        logger.info(f"Dispatching {len(new_trials)} new trials")

        for trial in new_trials:
            # Register trial
            self._register_trial(trial)

            # Convert trial to job format for dispatcher
            job = self._trial_to_job(trial)

            # Dispatch via dispatcher
            dispatch_id = self.dispatcher.dispatch(job)

            # Update trial state
            self._update_trial_state(trial.id, TrialState.PENDING, reason="Dispatched", dispatch_id=dispatch_id)

            # Track as running
            self.running.add(trial.id)

            logger.info(f"Dispatched trial {trial.id} with params {trial.params}")

    def _trial_to_job(self, trial: Trial) -> JobDefinition:
        """Convert a trial to a job format for the dispatcher.

        This is a bridge between the Trial abstraction and the dispatcher's
        job format. Override if you need custom conversion.

        Args:
            trial: Trial to convert

        Returns:
            Job object suitable for dispatcher
        """
        # Create a basic JobDefinition - subclasses should override for custom logic
        return JobDefinition(
            run_id=trial.id,
            cmd="",  # Subclass should set appropriate command
            args={"group": self.experiment_id},
            overrides=trial.params,  # Trial params become config overrides
            metadata={
                "budget": trial.budget,
                "checkpoint": trial.checkpoint_uri,
            },
        )

    def _calculate_available_slots(self) -> int:
        """Calculate available resource slots for new trials."""
        current_active = self.n_active
        return max(0, self.max_parallel - current_active)

    # ============= State Management =============

    def _register_trial(self, trial: Trial) -> None:
        """Register a new trial in the system."""
        if trial.id in self.trials:
            raise ValueError(f"Trial {trial.id} already registered")

        self.trials[trial.id] = trial
        logger.debug(f"Registered trial {trial.id}")

    def _update_trial_state(self, trial_id: str, new_state: TrialState, **metadata) -> None:
        """Update trial state with validation and execute hooks.

        Args:
            trial_id: ID of trial to update
            new_state: Target state
            **metadata: Additional metadata for the transition
        """
        trial = self.trials.get(trial_id)
        if not trial:
            logger.warning(f"Attempted to update unknown trial {trial_id}")
            return

        old_state = trial.state
        try:
            reason = metadata.pop("reason", "")
            transition = trial.transition_to(new_state, reason=reason, **metadata)
            logger.debug(f"Trial {trial_id}: {old_state.name} → {new_state.name}")

            # Create context for hooks (infrastructure side effects)
            context = TransitionContext(
                trial=trial,
                orchestrator=self,
                transition=transition,
                store=self.store,
                dispatcher=self.dispatcher,
                extra=metadata,  # Pass remaining metadata
            )

            # Execute matching hooks for side effects
            self._execute_hooks(old_state, new_state, context)

        except ValueError as e:
            logger.error(f"Invalid state transition for {trial_id}: {e}")

    def _record_observation(self, trial_id: str, observation: Observation) -> None:
        """Record observation from completed trial."""
        self.observations[trial_id] = observation

        # Update best score if this is better
        if observation.score > self.best_score:
            self.best_score = observation.score

        # Update total cost
        self.total_cost += observation.cost

        logger.debug(f"Recorded observation for {trial_id}: score={observation.score:.4f}")

    # ============= Hook System for Infrastructure =============

    def add_hook(
        self,
        from_state: Optional[TrialState],
        to_state: Optional[TrialState],
        action: Callable[[TransitionContext], None],
        name: str = "",
        priority: int = 0,
    ) -> None:
        """Add a transition hook for infrastructure side effects.

        Args:
            from_state: Source state (None = any)
            to_state: Target state (None = any)
            action: Function to execute
            name: Hook name for debugging
            priority: Execution priority (higher = first)
        """
        hook = TransitionHook(from_state, to_state, action, name, priority)
        self.transition_hooks.append(hook)
        # Sort by priority (higher first)
        self.transition_hooks.sort(key=lambda h: -h.priority)

    def _execute_hooks(self, from_state: TrialState, to_state: TrialState, context: TransitionContext) -> None:
        """Execute all matching hooks for a transition.

        Infrastructure side effects only - not core logic.
        """
        for hook in self.transition_hooks:
            if hook.matches(from_state, to_state):
                try:
                    logger.debug(f"Executing hook '{hook.name}' for {from_state.name} → {to_state.name}")
                    hook.action(context)
                except Exception as e:
                    logger.error(f"Hook '{hook.name}' failed: {e}", exc_info=True)
                    # Continue with other hooks - side effects shouldn't break core flow

    def _register_infrastructure_hooks(self) -> None:
        """Register default hooks for common infrastructure concerns.

        These handle side effects like:
        - Initializing runs in WandB
        - Setting S3 checkpoint URIs
        - Updating WandB metrics
        - Logging state changes
        """
        # Initialize run in WandB when trial is dispatched
        self.add_hook(
            TrialState.SUGGESTED, TrialState.PENDING, self._init_wandb_run_hook, name="init_wandb_run", priority=20
        )

        # Set checkpoint URI when training completes
        self.add_hook(
            TrialState.TRAINING,
            TrialState.AWAITING_EVALUATION,
            self._set_checkpoint_uri_hook,
            name="set_checkpoint_uri",
            priority=10,
        )

        # Update WandB when evaluation completes
        self.add_hook(
            TrialState.EVALUATING, TrialState.COMPLETED, self._update_wandb_hook, name="update_wandb", priority=10
        )

        # Log all transitions (lower priority)
        self.add_hook(
            None,  # Any state
            None,  # Any state
            self._log_transition_hook,
            name="log_transition",
            priority=-10,
        )

    def _init_wandb_run_hook(self, ctx: TransitionContext) -> None:
        """Infrastructure hook: Initialize WandB run when trial is dispatched."""
        # Initialize run in store with trial metadata
        initial_summary = {
            "sweep/suggestion": ctx.trial.params,
            "sweep/trial_id": ctx.trial_id,
        }
        if ctx.trial.budget:
            initial_summary["sweep/budget"] = ctx.trial.budget

        try:
            ctx.store.init_run(ctx.trial_id, group=ctx.orchestrator.experiment_id, initial_summary=initial_summary)
            logger.info(f"Initialized WandB run for {ctx.trial_id}")
        except Exception as e:
            logger.error(f"Failed to initialize WandB run for {ctx.trial_id}: {e}")

    def _set_checkpoint_uri_hook(self, ctx: TransitionContext) -> None:
        """Infrastructure hook: Set S3 checkpoint URI."""
        ctx.trial.checkpoint_uri = f"{SOFTMAX_S3_POLICY_PREFIX}/{ctx.trial_id}:latest"
        logger.info(f"Set checkpoint URI for {ctx.trial_id}: {ctx.trial.checkpoint_uri}")

    def _update_wandb_hook(self, ctx: TransitionContext) -> None:
        """Infrastructure hook: Update WandB with sweep metrics."""
        # Only update if we have an observation in context
        if "observation" in ctx.extra:
            observation = ctx.extra["observation"]
            sweep_data = {
                "sweep/score": observation.score,
                "sweep/cost": observation.cost,
                "sweep/suggestion": ctx.trial.params,
            }

            # Update remote store (WandB)
            try:
                # Note: We need the run_id, which might be different from trial_id
                # For now, assume they're the same (this might need adjustment)
                self.store.update_run_summary(ctx.trial_id, sweep_data)
                cost_info = f" (from '{ctx.orchestrator.cost_key}')" if ctx.orchestrator.cost_key else ""
                logger.info(
                    f"Updated WandB for {ctx.trial_id}: "
                    f"score={observation.score:.6f} (from '{ctx.orchestrator.metric_key}'), "
                    f"cost={observation.cost:.2f}{cost_info}"
                )
            except Exception as e:
                logger.error(f"Failed to update WandB for {ctx.trial_id}: {e}")

    def _log_transition_hook(self, ctx: TransitionContext) -> None:
        """Infrastructure hook: Log all state transitions."""
        logger.info(
            f"[TRANSITION] {ctx.trial_id}: "
            f"{ctx.from_state.name} → {ctx.to_state.name} "
            f"(reason: {ctx.transition.reason})"
        )

    # ============= Abstract Methods (Must Override) =============

    @abstractmethod
    def setup(self) -> None:
        """Initialize strategy-specific state and components.

        Called once during __init__ after universal state is set up.
        Use this to initialize optimizers, models, etc.
        """
        pass

    @abstractmethod
    def suggest_trials(self, n_slots: int) -> list[Trial]:
        """Suggest next trials to run given available slots.

        This is the core strategy method where different algorithms
        (Bayesian, ASHA, PBT, etc) implement their logic.

        Args:
            n_slots: Number of available resource slots

        Returns:
            List of trials to dispatch (may be fewer than n_slots)
        """
        pass

    @abstractmethod
    def should_stop(self) -> bool:
        """Determine if experiment should terminate.

        Check stopping conditions like max trials, convergence,
        budget exhausted, etc.

        Returns:
            True if experiment should stop
        """
        pass

    # ============= Optional Hooks (Can Override) =============

    def on_trial_completed(self, trial: Trial, observation: Observation) -> None:  # noqa: B027
        """Hook called when a trial completes successfully.

        Override to update strategy-specific state, models, etc.
        Default implementation does nothing as metrics are updated by _record_observation.
        """
        return None

    def on_trial_failed(self, trial: Trial, error: str) -> None:
        """Hook called when a trial fails.

        Override to handle failures in strategy-specific ways.
        Default implementation just logs.
        """
        logger.warning(f"Trial {trial.id} failed: {error}")

    def should_checkpoint(self, trial: Trial, observation: Observation) -> bool:
        """Determine if trial should be checkpointed.

        Override to implement checkpointing strategies.
        Default returns False (no checkpointing).
        """
        return False

    def should_promote(self, trial: Trial) -> bool:
        """Determine if trial should be promoted (ASHA/Hyperband).

        Override for successive halving strategies.
        Default returns False (no promotion).
        """
        return False

    # ============= Utility Methods =============

    @property
    def active_trials(self) -> list[Trial]:
        """Get all trials currently using resources."""
        return [t for t in self.trials.values() if t.is_active]

    @property
    def completed_trials(self) -> list[Trial]:
        """Get all successfully completed trials."""
        return [t for t in self.trials.values() if t.state == TrialState.COMPLETED]

    @property
    def n_completed(self) -> int:
        """Number of completed trials."""
        return len(self.completed)

    @property
    def n_failed(self) -> int:
        """Number of failed trials."""
        return len(self.failed)

    @property
    def n_active(self) -> int:
        """Number of active trials."""
        return len(self.running)

    @property
    def elapsed_time(self) -> float:
        """Elapsed time since experiment started (seconds)."""
        return (datetime.now() - self.start_time).total_seconds()

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get trial by ID."""
        return self.trials.get(trial_id)

    def get_observation(self, trial_id: str) -> Optional[Observation]:
        """Get observation for trial."""
        return self.observations.get(trial_id)
