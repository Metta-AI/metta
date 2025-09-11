# Adaptive Experiments: Controller Hooks, Retries, and Generalization (Epic)

## Summary
- Introduce a controller-owned on_eval_completed hook with an idempotent store flag to enable domain-specific post-processing (sweeps, learning progress, curriculum) without adding state to schedulers.
- Harden Store interactions (WandB) with standardized retries using the existing retry utility.
- Keep schedulers pure and stateless; keep the controller simple and robust.
- Optional: add coarse-grained job types (RESUME_TRAINING, WARM_START_TRAINING) for inter-run adaptation.
- Document stable summary key conventions so schedulers can read aggregates directly from `RunInfo.summary`.

## Goals
- Reliability: Post-eval processing runs exactly once per evaluated run, even across restarts.
- Simplicity: Hooks live in the controller; schedulers remain stateless decision-makers.
- Modularity: Sweep/curriculum/LP logic plugs in via callbacks and summary schemas.
- Observability: Standardized, minimal summary keys for cross-experiment scheduling.

## Non‑Goals
- Mid-training reconfiguration by the controller (handled in-trainer for curriculum).
- Full-blown workflow engine; keep the control loop minimal.
- Replacing trainer-level checkpointing or WandB logging.

## Background & Current State
- Controller loop is clean (fetch → check complete → compute capacity → schedule → dispatch), but lacks lifecycle hooks.
- `RunInfo` is intentionally thin; raw `summary` is available and should remain the primary substrate for scheduler logic.
- Sweeps currently expect an `Observation` on `RunInfo`. This leaks sweep-specific concerns into a general model.
- WandB interactions (init, fetch, update) are not consistently wrapped with retries.

Existing utilities:
- Retry helpers: `common/src/metta/common/util/retry.py`
  - `retry_function`, `retry_on_exception`, `calculate_backoff_delay`

## Proposed Changes (High Level)
1) Controller‑owned on_eval_completed hook with a summary guard.
2) Add retries to Store (WandB) methods: `init_run`, `fetch_runs`, `update_run_summary`.
3) Make observations sweep‑specific via the hook; remove `Observation` from `RunInfo`.
4) Optional: Add JobTypes.RESUME_TRAINING and JobTypes.WARM_START_TRAINING.
5) Document stable summary key conventions for schedulers.

---

## Detailed Design

### 1) Controller Hook: on_eval_completed
- Ownership: Controller, not Scheduler.
- Timing: Run immediately after `fetch_runs`, before any completion checks or scheduling.
- Detection: For each run, if `run.has_been_evaluated is True` and `summary.get('post_eval_processed') != True`, then invoke the hook.
- Idempotency: After hook success, set `summary.post_eval_processed = True` and `summary.post_eval_processed_at = <timestamp>` via Store.
- API:
  - `on_eval_completed(run: RunInfo, store: Store, all_runs: list[RunInfo]) -> None`
  - Passed at controller construction time (or via config); default is no‑op.
- Immediate availability: If the hook produces data schedulers need in the same loop, also update the in-memory `run.summary` before calling Store (so `schedule()` sees it in the current cycle).
- Error handling: Wrap hook + summary update with retries; if retries exhaust, leave the flag unset and retry next loop.

Suggested flag key: `post_eval_processed` (or `adaptive/post_eval_processed` if you prefer namespacing).

### 2) Store (WandB) Retries
- Decorate with `retry_on_exception`:
  - `WandbStore.init_run(...)`
  - `WandbStore.fetch_runs(...)`
  - `WandbStore.update_run_summary(...)`
- Rationale: Transient network/rate-limit errors are common; these operations are idempotent or harmless to retry.
- Controller can rely on Store retries and remove special-casing for initial fetch timeouts.

### 3) Sweep Observation Extraction via Hook
- Remove `Observation` from `RunInfo` to keep the model general.
- Provide a sweep-specific `on_eval_completed` hook that:
  - Reads raw metrics from `run.summary`.
  - Computes a normalized observation dict: `{score, cost, suggestion/config, ...}`.
  - Writes results back to `run.summary` under a stable prefix, e.g., `observation/score`, `observation/cost`, `observation/suggestion`.
- Schedulers then consume observations from `summary` rather than `RunInfo` attributes.

### 4) Optional Job Types for Inter‑Run Adaptation
- `RESUME_TRAINING`: Continue an existing run with new overrides (same `run_id`).
- `WARM_START_TRAINING`: Start a new run that loads weights from a previous policy URI (`warm_start_uri`).
- Dispatch implications:
  - RESUME: pass `run=<run_id>` and overrides; trainer resumes from its own checkpoints.
  - WARM_START: pass `warm_start_uri=...`; trainer loads weights before training.
- Safety: Dispatchers should fail hard on launch errors (no auto‑retries) to avoid double launches.

### 5) Summary Key Conventions (Scheduler‑Visible)
- Curriculum: `curriculum/*`, task selection telemetry, optional per‑task aggregates.
- Learning Progress: `lp/task/<id>/*` (e.g., EMA/slope/weight), along with `metric/agent_step`.
- Sweeps: `observation/*` for normalized results; rely on cost/runtime fields or precomputed cost.
- Trainer can log these during train/eval; hooks can augment/normalize them post‑eval.

---

## Implementation Plan & Milestones

Phase 1 — Foundation (Controller + Store)
- Add `on_eval_completed` to `AdaptiveController` and implement guarded execution with retries.
- Decorate `WandbStore` methods with `retry_on_exception`.
- Acceptance: Hook executes exactly once per evaluated run despite transient Store/API failures.

Phase 2 — Sweeps Integration
- Implement a default sweep `on_eval_completed` that writes `observation/*` to the summary.
- Remove `Observation` from `RunInfo`; adjust sweep schedulers to read `summary['observation/*']`.
- Acceptance: Batched/synced sweep works end‑to‑end with observations produced via hook.

Phase 3 — Curriculum Friendly Surface
- Document minimal `curriculum/*` metrics and how trainers should log them.
- Provide a curriculum example scheduler that reads raw summaries and makes decisions (no mid‑training config changes required).
- Acceptance: Example curriculum adaptive experiment schedules tasks and logs curriculum stats.

Phase 4 — Learning Progress (LP)
- Document `lp/task/<id>/*` conventions and/or leverage Stats service if preferred.
- Provide an LP scheduler using summary aggregates and/or post‑eval computations in the hook.
- Acceptance: LP scheduler selects tasks based on computed LP and converges under test conditions.

Phase 5 — Optional Inter‑Run Adaptation
- Add `RESUME_TRAINING` / `WARM_START_TRAINING` job types and dispatcher argument plumbing.
- Add `warm_start_uri` support in trainer if needed.
- Acceptance: A demo shows warm‑starting a new run from a previous policy via the adaptive tool.

---

## Acceptance Criteria (Global)
- Controller detects eval completions and runs the hook once per run.
- Store operations are resilient to transient errors (retries with backoff).
- Schedulers remain stateless; no controller or scheduler keeps persistent in‑memory state across loops.
- Sweeps no longer depend on `RunInfo.observation`; observation lives in the summary.
- Documentation clearly states the summary keys that schedulers rely on.

## Risks & Mitigations
- Double processing: Mitigated by the `post_eval_processed` flag.
- Duplicate launches: Dispatchers fail hard (no auto‑retries). Controller reschedules in later cycles if needed.
- Schema drift in summary keys: Mitigated by documenting conventions and providing examples/tests.
- Hook failures blocking progress: Isolated per‑run with retries; failures don’t block other runs.

## Out of Scope
- Controller-driven mid-training reconfiguration (handled inside trainer curriculum logic).
- Historical time-series queries from the controller (prefer logging aggregates as summary fields).

## Open Questions
- Do we want namespaced flags (e.g., `adaptive/post_eval_processed`) or the simple `post_eval_processed`?
- For sweeps, should we standardize a minimal observation schema across projects (e.g., always `observation/score`, `observation/cost`)?
- Should we also add an `on_training_finished` hook now or defer until needed?

## References
- Retry utility: `common/src/metta/common/util/retry.py`
- Controller: `metta/adaptive/adaptive_controller.py`
- Store (WandB): `metta/adaptive/stores/wandb.py`
- Dispatchers: `metta/adaptive/dispatcher/*.py`
- Scheduler example (PoC): `metta/adaptive/schedulers/train_and_eval.py`
- Sweep batched/synced (reference): `metta/sweep/schedulers/batched_synced.py`
