from datetime import datetime, timezone

from metta.adaptive.models import RunInfo
from metta.sweep.schedulers.state import SchedulerState


def _now():
    return datetime.now(timezone.utc)


def test_scheduler_state_refresh_and_transitions():
    state = SchedulerState()
    now = _now()

    pending = RunInfo(
        run_id="pending",
        created_at=now,
        last_updated_at=now,
        has_started_training=False,
        has_completed_training=False,
        has_started_eval=False,
        has_been_evaluated=False,
        has_failed=False,
    )
    in_eval = RunInfo(
        run_id="in_eval",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=True,
        has_started_eval=True,
        has_been_evaluated=False,
        has_failed=False,
        summary={"sweep/suggestion": {"lr": 0.01}},
    )
    completed = RunInfo(
        run_id="done",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=True,
        has_started_eval=True,
        has_been_evaluated=True,
        has_failed=False,
        summary={"sweep/suggestion": {"lr": 0.02}},
    )

    state.refresh([pending, in_eval, completed], force_eval=True)

    assert state.runs_in_training == {"pending"}
    assert state.runs_pending_force_eval == {"in_eval"}
    assert "in_eval" not in state.runs_in_eval
    # Suggestions are cached for in-flight runs only
    assert state.in_progress_suggestions["in_eval"] == {"lr": 0.01}
    assert "done" not in state.in_progress_suggestions

    # Scheduling eval moves it out of pending-force set and into eval tracking
    state.mark_eval_scheduled("in_eval")
    assert state.runs_pending_force_eval == set()
    assert state.runs_in_eval == {"in_eval"}

    # Refresh after completion prunes suggestions and marks completion
    in_eval_done = RunInfo(
        run_id="in_eval",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=True,
        has_started_eval=True,
        has_been_evaluated=True,
        has_failed=False,
        summary={"sweep/suggestion": {"lr": 0.01}},
    )
    state.refresh([pending, in_eval_done], force_eval=False)

    assert "in_eval" in state.runs_completed
    assert "in_eval" not in state.in_progress_suggestions
    assert state.eval_capacity(max_concurrent_evals=1) == 1
