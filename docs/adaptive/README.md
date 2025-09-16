# Adaptive Experiments — Recipe Book

This guide shows how to run adaptive experiments (single run train→eval and hyperparameter sweeps) and gives a friendly deep dive into how the pieces fit together.

## Quick Start

- Train & Eval (adaptive)
  - `uv run ./tools/run.py experiments.recipes.adaptive.train_and_eval dispatcher_type=skypilot max_trials=3 gpus=4`
  - Local instead of Skypilot: `dispatcher_type=local`

Notes:
- W&B must be configured; AdaptiveTool uses your WandB config (auto if not set explicitly).
- Arguments are automatically classified as function parameters vs configuration overrides.
- No more `--args` or `--overrides` flags needed.

## Recipes

- Train & Eval
  - Entry point: `experiments.recipes.adaptive.train_and_eval`
  - What it does: trains a series of runs and evaluates each when training completes.
  - Scheduler config: `TrainAndEvalConfig` (typed)
    - Fields (common): `recipe_module`, `train_entrypoint`, `eval_entrypoint`, `max_trials`, `gpus`, `experiment_id`, `train_overrides`
  - Typical run: `uv run ./tools/run.py experiments.recipes.adaptive.train_and_eval max_trials=3 gpus=4`

## Reference — Scheduler Configs (typed)

- TrainAndEvalConfig
  - `recipe_module`: Python module path for recipe (e.g., `experiments.recipes.arena`)
  - `train_entrypoint`: function name (e.g., `train`)
  - `eval_entrypoint`: function name (e.g., `evaluate`)
  - `max_trials`: total number of training runs
  - `gpus`: GPUs per training job
  - `experiment_id`: W&B group name and run-id prefix
  - `train_overrides`: dict of config overrides applied to all training jobs

- BatchedSyncedSchedulerConfig
  - `recipe_module`, `train_entrypoint`, `eval_entrypoint`: same as above
  - `max_trials`: total trials across all batches
  - `batch_size`: number of concurrent training jobs per batch (also respects max_parallel in controller)
  - `gpus`, `nodes`: resources per training job
  - `experiment_id`: W&B group and run-id prefix
  - `train_overrides`: shared overrides for all training jobs
  - Optional: `eval_overrides`, `stats_server_uri`

Both configs are Pydantic models (subclass of Config) and serialize cleanly via `model_dump_json`.

## Deep Dive — How it all fits together

- Entry point builds a typed scheduler config and returns an `AdaptiveTool`:
  - Example: `experiments.recipes.adaptive.train_and_eval` builds `TrainAndEvalConfig` and returns an `AdaptiveTool` with `scheduler_type=TRAIN_AND_EVAL` and the config.

- AdaptiveTool wires components and runs the controller:
  - Creates the scheduler from your typed config (type-checked), builds the dispatcher (local or Skypilot), and creates a W&B store.
  - Writes `adaptive_config.json` to `run_dir` for reproducibility.

- Controller loop (simple and resilient):
  - Fetches all runs for the experiment from W&B (with retries).
  - Runs `on_eval_completed` for any run with completed eval that hasn't been post-processed. This hook is idempotent using namespaced flags:
    - `adaptive/post_eval_processed = True`
    - `adaptive/post_eval_processed_at = <timestamp>`
  - The default sweep hook writes normalized results into summary:
    - `observation/score`: taken from your Protein metric (or `evaluator/eval_arena/score` fallback)
    - `observation/cost`: derived from runtime if no cost exists
    - `observation/suggestion`: persisted at training dispatch time
  - Computes available training slots based on `config.max_parallel` and current pending/in-training runs.
  - Calls `scheduler.schedule(runs, available_training_slots)` to get new jobs.
  - Dispatches jobs:
    - Local: `uv run ./tools/run.py <recipe> key=val key2=val2`
    - Skypilot: `<SKYPILOT_LAUNCH_PATH> [--gpus N] [--nodes M] <recipe> key=val key2=val2`
  - Updates store for new training runs (initializes run, writes suggestion), or marks eval started.
  - Loops until `scheduler.is_experiment_complete(runs)` returns True.

- Job definition (single-source of truth):
  - `args`: dict of function args passed directly (training uses `run`, `group`; eval uses `policy_uri`)
  - `overrides`: dict of config overrides passed directly (includes suggestions for sweeps)
  - Dispatchers do not invent args from metadata; everything is explicit and formed in utils/schedulers.

- Schedulers (two included):
  - TrainAndEvalScheduler: schedules evals for runs with training done, and new training up to `max_trials`.
  - BatchedSyncedOptimizingScheduler: waits for all active runs to complete (barrier), collects observations from `summary['observation/*']`, queries Protein for suggestions, merges suggestions into training overrides, and launches the next batch.

- Store (W&B) resiliency:
  - `init_run`, `fetch_runs`, `update_run_summary` are wrapped with exponential backoff retries.
  - The controller's hook invocation also uses retries before setting the processed flag.

- State management:
  - Optional `StateStore` (FileStateStore) persists scheduler state for resumption.
  - Schedulers implementing `SchedulerWithState` can load/save state automatically.

## Tips & Patterns

- Automatic argument classification
  - The runner automatically determines which arguments go to the function vs configuration overrides based on introspection.
  - Function parameters become function arguments, Tool fields become configuration overrides.

- Resource sweeping
  - Use the `gpus` field in your typed config to sweep resource allocation across runs.

- Local development
  - Use `dispatcher_type=local` for fast iteration when building or debugging recipes.

- Custom schedulers
  - Create a Pydantic config (subclass of Config) for your scheduler inputs.
  - Implement a scheduler with `schedule(runs, available_training_slots)` and `is_experiment_complete(runs)`.
  - Add an enum value to `SchedulerType` and a branch in AdaptiveTool to construct your scheduler from your typed config.

## Standard Keys

- Namespaced flags (controller-managed):
  - `adaptive/post_eval_processed`, `adaptive/post_eval_processed_at`

- Observation fields (hook-written):
  - `observation/score` (from your configured metric)
  - `observation/cost` (runtime-derived if missing)
  - `observation/suggestion` (persisted at training dispatch)

These conventions keep schedulers stateless and make experiments easy to reason about from summaries alone.