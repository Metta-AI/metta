# Checkpoint Serialization & Loading Refactor

This document outlines the repo changes made when introducing
`PolicyArtifact`-based checkpoint loading and safetensors
serialization. Follow these steps from a fresh checkout to reproduce the
work.

## 1. Shared Policy Serialization Helpers

Create `metta/rl/policy_serialization.py` containing:

- `PolicyArtifact` dataclass that stores either a ready-to-use
  `Policy` or a `(PolicyArchitecture, state_dict)` pair. Implement
  validation in `__post_init__`.
- `instantiate(env_metadata, strict=True)` method that rebuilds the
  policy from `PolicyArchitecture` + weights when `policy` is `None`.
- Helper functions to serialize/deserialize architectures:
  `_architecture_class_path`, `_dump_policy_architecture`, and
  `_load_policy_architecture` (using `load_symbol`).
- Functions `save_policy_artifact(base_path, policy, policy_architecture,
  detach_buffers=True)` and `load_policy_artifact(base_path)` that write/
  read `{base}.safetensors` weight files and `{base}.policyarchitecture`
  JSON payloads.

Populate implementation exactly as in the diff (ordered dicts, CPU
detaching, safetensors usage).

## 2. Update Checkpointer to Use Shared Helpers

In `metta/rl/training/checkpointer.py`:

- Replace previous serialization logic with imports from
  `metta.rl.policy_serialization`.
- Add `policy_architecture` parameter to `Checkpointer.__init__` and
  store it on `self._policy_architecture` with a read-only property.
- Update `load_or_create_policy` to use `self._policy_architecture` and
  handle the new bundle API
  (`bundle.policy` fallback → `bundle.instantiate(env_metadata)`).
- Expose `save_policy_artifact`/`load_policy_artifact` helpers that forward
  to shared helper functions.

## 3. CheckpointManager Returns Bundles

In `metta/rl/checkpoint_manager.py`:

- Import `PolicyArtifact` + `load_policy_artifact`.
- Extend `_find_latest_checkpoint_in_dir` to consider both `.pt` and
  `.safetensors` files.
- Modify `_load_checkpoint_file` to return a bundle containing the
  torch-loaded policy.
- Add `_load_bundle_from_path` that selects between `.pt` legacy files
  (using `_load_checkpoint_file`) and safetensor bundles
  (`load_policy_artifact` on the path without suffix).
- Change `load_from_uri` to return `PolicyArtifact` everywhere
  (file, directory, s3, mock). For mock URIs return a bundle wrapping a
  `MockAgent`.
- Remove `load_agent` and its caching logic. Delete
  `tests/rl/test_checkpoint_manager.py` entirely and migrate any
  remaining references.

## 4. TrainTool Adjustments

In `metta/tools/train.py`:

- Pass `policy_architecture=self.policy_architecture` when constructing
  a `Checkpointer` inside `_load_or_create_policy`.
- Drop the explicit architecture argument when calling
  `load_or_create_policy` (new signature uses property).

## 5. Losses Use Bundles

Update `metta/rl/loss/tl_kickstarter.py` and
`metta/rl/loss/sl_kickstarter.py`:

- Replace direct policy loads with bundles. Fetch bundle via
  `CheckpointManager.load_from_uri`.
- If bundle already carries a policy, reuse it; otherwise instantiate
  using `self.env.meta_data`. Throw if metadata is not available.
- Maintain backward compatibility with policies whose
  `initialize_to_environment` requires legacy positional arguments
  (wrap in a try/except TypeError and fall back to old behaviour).
- Detach gradients as before.

## 6. Simulation Accepts Bundles

In `metta/sim/simulation.py`:

- Accept `policy: Policy | PolicyArtifact` in the constructor
  and `Simulation.create`.
- Store bundles (`self._policy_bundle`, `self._npc_policy_bundle`) and
  instantiate policies during initialization once
  `EnvironmentMetaData` is built.
- For `Simulation.create`, default to a bundle wrapping `MockAgent` when
  `policy_uri` is `None`.
- Add `_instantiate_policy` helper that caches the instantiated policy
  back onto the bundle to avoid repeated rebuilds.

## 7. Evaluation and Tooling Updates

- `metta/eval/eval_service.py`: load bundle, pass bundle to
  `Simulation` instead of raw policy.
- `metta/tools/sim.py` & `tools/request_eval.py`: use bundles for URI
  validation (drop direct policy references).
- `metta/setup/shell.py`: adjust example code to reference bundles and
  note the need for environment metadata when instantiating.

## 8. Notebook Integration

In `experiments/marimo/01-hello-world-marimo.py` (both evaluation code
paths):

- Import `EnvironmentMetaData`.
- Replace direct policy loads with bundles, build metadata from the
  constructed `MettaGridEnv`, and instantiate policies before use.

## 9. Trainer Tests

- In `tests/integration/test_trainer_checkpoint.py`, assert on bundle
  contents (`bundle.policy`) instead of `checkpoint_manager.load_agent`.

## 10. URI Integration Tests

In `tests/rl/test_checkpoint_uri_integration.py`:

- Update assertions to check `loaded.policy` is a `torch.nn.Module`.
- Adjust S3 mock test accordingly.

## 11. Tool Tests

In `tests/tools/test_new_policy_system.py`, update mock URI assertions to
work with bundles (`bundle.policy`).

## 12. Dependency Changes

- Add `safetensors>=0.4.3` to `[project.dependencies]` in
  `pyproject.toml`.
- Regenerate `uv.lock` after installing the new dependency.

## 13. Formatting & Housekeeping

- Run `uv run ruff format` on all modified Python files.
- Remove the now-unused `tests/rl/test_checkpoint_manager.py` file.

## 14. Instantiate Policies via Bundle Helper

- Simplify all call sites that previously branched on
  `bundle.policy is not None` to simply call
  `bundle.instantiate(env_metadata)` (which now caches the policy on the
  bundle). Updated locations:
  - `metta/rl/training/checkpointer.py` when loading checkpoints.
  - `metta/rl/loss/tl_kickstarter.py` and `metta/rl/loss/sl_kickstarter.py`
    (retain legacy fallback only when metadata is unavailable).
  - `metta/sim/simulation.py` `_instantiate_policy` helper.
- Enhance `PolicyArtifact.instantiate` to store the instantiated
  policy back onto the bundle for reuse.

## 15. Cleanup Dead CheckpointManager Code

- Removed the unused `CheckpointManager.load_agent` method now that all
  loading is performed via `load_from_uri` returning bundles.
- Added clarifying docstrings outlining the separation of concerns
  between `CheckpointManager` (persistence) and `Checkpointer`
  (orchestration), and dropped unused helper wrappers from
  `Checkpointer` to reinforce the distinction.

## 16. Architecture Manifest Emission

- On `Checkpointer.register`, the master rank now writes a
  `model_architecture.json` manifest (containing class path and config
  dump) alongside policy checkpoints; the file is created once per run
  if absent.

## 17. Training Metrics Terminology

- Renamed checkpoint metadata helpers to `_collect_training_metrics` and
  updated `CheckpointManager.save_agent` to accept a `training_metrics`
  dict, clarifying the payload being stored with each checkpoint.

## 18. Policy Bundle Metrics & Architecture Manifest

- `PolicyArtifact` now tracks optional `training_metrics` (only
  for the state-dict branch) and caches instantiated policies.
- Safetensor bundles persist training metrics to
  `<checkpoint>.metrics.json`; architecture information is sourced from
  the run-level `model_architecture.json` manifest instead of per-bundle
  serialization.

## 19. Checkpoint Format Options

- `CheckpointConfig` gains a `checkpoint_format` field (`pt` or
  `safetensors`). `TrainTool` passes this through to `CheckpointManager`,
  which now handles both formats: safetensor saves delegate to
  `save_policy_artifact`, persist metrics alongside weights, and loads
  inspect the file extension to select the appropriate path.

By executing the steps above in order, a coding agent can reproduce the
entire checkpoint bundle refactor on a clean copy of the repository.
