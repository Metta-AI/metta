"""End-to-end sweep orchestrator behavior using in-memory fakes."""

from __future__ import annotations

from metta.sweep.config import SweepOrchestratorConfig
from metta.sweep.models import JobTypes
from metta.sweep.orchestrator import SweepOrchestrator, Trial, TrialState
from tests.sweep.fakes import FakeDispatcher, FakeStore


class MinimalSweep(SweepOrchestrator):
    """Simple orchestrator that dispatches one training then stops."""

    def __init__(
        self,
        experiment_id: str,
        dispatcher: FakeDispatcher,
        store: FakeStore,
        config: SweepOrchestratorConfig,
    ) -> None:
        self._suggested = False
        super().__init__(experiment_id, dispatcher, store, config)

    def setup(self) -> None:
        pass

    def suggest_trials(self, n_slots: int) -> list[Trial]:
        if self._suggested or n_slots <= 0:
            return []
        self._suggested = True
        return [Trial(id=f"{self.experiment_id}_trial_0001", params={"lr": 1e-3})]

    def should_stop(self) -> bool:
        return self.n_completed + self.n_failed >= 1


def test_end_to_end_with_fakes_handles_resume_and_stale() -> None:
    # Prepare fakes and config
    store = FakeStore()
    dispatcher = FakeDispatcher(store=store)
    config = SweepOrchestratorConfig(max_parallel=1, poll_interval=0.01, initial_wait=0.0, metric_key="metric")

    orch = MinimalSweep(experiment_id="exp", dispatcher=dispatcher, store=store, config=config)

    # Manually mark an already-known run stale to exercise stale path
    running_id = "exp_trial_0002"
    store.ensure_run(run_id=running_id, group="exp", summary={"metric": 0.5})
    store.start_training(running_id)
    store.mark_stale(running_id, stale_seconds=7200)
    orch.trials[running_id] = Trial(id=running_id, params={}, state=TrialState.TRAINING)

    # Run a single sync/dispatch iteration
    orch._sync_state()
    orch._dispatch_new_trials()

    # Assertions:
    # - Stale run should be marked failed
    assert running_id in orch.failed
    assert orch.trials[running_id].state == TrialState.FAILED
    # - Dispatcher dispatched one training job for the new suggestion
    assert len(dispatcher.dispatched_by_type[JobTypes.LAUNCH_TRAINING]) == 1
    dispatched_run = dispatcher.dispatched_by_type[JobTypes.LAUNCH_TRAINING][0].run_id
    assert dispatched_run.startswith("exp_trial_000")
    # - Slot accounting respected: only one training dispatched due to max_parallel=1
    assert len(dispatcher.jobs) == 1
