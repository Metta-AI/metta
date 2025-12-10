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
    assert "in_eval" in state.runs_in_eval
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


def test_force_eval_counts_capacity_and_is_one_shot():
    state = SchedulerState()
    now = _now()

    run = RunInfo(
        run_id="r1",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=True,
        has_started_eval=True,
        has_been_evaluated=False,
        has_failed=False,
    )

    # First refresh should mark pending force eval and also track as in-eval for capacity
    state.refresh([run], force_eval=True)
    assert state.runs_pending_force_eval == {"r1"}
    assert state.runs_in_eval == {"r1"}
    assert state.eval_capacity(max_concurrent_evals=1) == 0

    # After scheduling, pending is cleared
    state.mark_eval_scheduled("r1")
    assert state.runs_pending_force_eval == set()

    # Subsequent refresh should NOT re-add to pending while still in eval
    state.refresh([run], force_eval=True)
    assert state.runs_pending_force_eval == set()


def test_training_done_no_eval_keeps_eval_tracking():
    state = SchedulerState()
    now = _now()

    in_eval = RunInfo(
        run_id="r_eval",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=True,
        has_started_eval=True,
        has_been_evaluated=False,
        has_failed=False,
    )
    state.refresh([in_eval], force_eval=False)
    assert state.runs_in_eval == {"r_eval"}

    training_done = RunInfo(
        run_id="r_eval",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=True,
        has_started_eval=False,
        has_been_evaluated=False,
        has_failed=False,
    )
    state.refresh([training_done], force_eval=False)

    # Should keep eval tracking to avoid rescheduling duplicates
    assert state.runs_in_eval == {"r_eval"}
    assert state.runs_in_training == set()


def test_suggestion_caching_is_sticky():
    state = SchedulerState()
    now = _now()

    first = RunInfo(
        run_id="r_sugg",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=False,
        has_started_eval=False,
        has_been_evaluated=False,
        has_failed=False,
        summary={"sweep/suggestion": {"lr": 0.1}},
    )
    state.refresh([first], force_eval=False)
    assert state.in_progress_suggestions["r_sugg"]["lr"] == 0.1

    updated = RunInfo(
        run_id="r_sugg",
        created_at=now,
        last_updated_at=now,
        has_started_training=True,
        has_completed_training=False,
        has_started_eval=False,
        has_been_evaluated=False,
        has_failed=False,
        summary={"sweep/suggestion": {"lr": 0.2}},
    )
    state.refresh([updated], force_eval=False)

    # Existing suggestion should not be overwritten
    assert state.in_progress_suggestions["r_sugg"]["lr"] == 0.1


def test_model_dump_roundtrip():
    state = SchedulerState(
        runs_in_training={"t1"},
        runs_in_eval={"e1"},
        runs_completed={"c1"},
        runs_pending_force_eval={"p1"},
        in_progress_suggestions={"t1": {"lr": 0.3}},
    )

    restored = SchedulerState.model_validate(state.model_dump())

    assert restored.runs_in_training == {"t1"}
    assert restored.runs_in_eval == {"e1"}
    assert restored.runs_completed == {"c1"}
    assert restored.runs_pending_force_eval == {"p1"}
    assert restored.in_progress_suggestions == {"t1": {"lr": 0.3}}
