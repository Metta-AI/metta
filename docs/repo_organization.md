# Metta Repository Organization – Forward Plan (Sep 2025)

## Baseline Snapshot
- Workspace Python packages we ship today: `agent/` (`metta-agent`), `app_backend/` (`metta-app-backend`), `codebot/`, `common/` (`metta-common`), `experiments/`, `gitta/`, `packages/cogames/`, `packages/mettagrid/`, `packages/pufferlib-core/`, and `softmax/`.
- Front-end and client surfaces live in `gridworks/`, `mettascope/`, `observatory/`, `library/`, `home/`, and `cogweb/`; shared scenes and assets stay in `resources/` and `scenes/`.
- Tooling, automation, and infra live under `devops/`, `docs/`, `scripts/`, `tools/`, `mcp_servers/`, and `train_dir/`.
- The legacy `metta/` namespace still aggregates `adaptive/`, `cogworks/`, `common/`, `eval/`, `gridworks/`, `map/`, `rl/`, `setup/`, `sim/`, `sweep/`, `tests_support/`, `tools/`, and `utils/`, plus the compatibility shim `metta/common` alongside the `mettagrid/` and `cogames/` stubs at repo root.

| Subdir (metta/*) | .py files | Outgoing cross-subpkg refs | Incoming refs |
| --- | ---: | ---: | ---: |
| adaptive | 11 | 4 | 10 |
| cogworks | 9 | 0 | 6 |
| common | 6 | 0 | 73 |
| eval | 7 | 13 | 9 |
| gridworks | 6 | 9 | 0 |
| map | 2 | 2 | 0 |
| rl | 43 | 38 | 15 |
| setup | 37 | 19 | 3 |
| sim | 11 | 13 | 16 |
| sweep | 6 | 3 | 2 |
| tests_support | 2 | 2 | 0 |
| tools | 9 | 51 | 7 |
| utils | 5 | 5 | 18 |

Numbers capture import edges between metta subdirectories (self-imports omitted).

## Dependency Hot Spots & Caveats
- **`metta.common` as shared spine**: 73 inbound references from other metta subpackages plus heavy usage across `app_backend/`, `devops/`, and `softmax/`. Any migration must preserve the `metta.common` API contract indefinitely; dropping the shim would break test fixtures (`conftest.py:8`, `app_backend/tests/conftest.py:14`) and CLI tooling (`metta/tools/play.py:9-11`).
- **`tools/` is the hub**: 51 outgoing references into `rl/`, `sim/`, `eval/`, `adaptive/`, `setup/`, and `sweep/`, while seven other subpackages import it. Example: `metta/tools/train.py:13-16` binds together `metta.agent`, `metta.common`, `metta.rl`, and `metta.sim`. Pulling `tools/` into a "core" package without refactors would drag the entire training stack with it.
- **Strongly connected component (`setup`, `tools`, `eval`, `rl`, `sim`)**: These directories import one another, creating cycles we must break before standalone packages exist. Key edges:
  - `setup -> tools` via CLI helpers (`metta/setup/metta_cli.py:14` → `metta.tools.utils.auto_config`).
  - `tools -> setup` via environment bootstrapping (`metta/tools/utils/auto_config.py:9-11`).
  - `rl <-> sim` through config and execution (`metta/rl/training/evaluator.py:13` ↔ `metta/sim/simulation.py:1-40`).
  - `rl <-> eval` (`metta/rl/training/stats_reporter.py:14` and `metta/eval/eval_service.py:1-20`).
- **`utils/` not purely foundational**: `metta/utils/live_run_monitor.py:39` pulls in `metta.adaptive.*`, so `utils/` cannot be dropped into a base package without pruning.
- **External workspace touch points**: `metta/rl` and `metta/sim` call into `metta.agent.*` (19 refs) and `metta.app_backend.*` (10 refs), tying training workflows to agent definitions and service clients (`metta/rl/training/evaluator.py:33-37`, `metta/sim/simulation.py:1-40`).
- **`mettagrid` dependence**: Training and tooling reference `mettagrid.*` configs and utilities extensively (`metta/rl/trainer_config.py`, `metta/tools/sim.py`, `metta/gridworks/routes/configs.py`), so any package split must carry that dependency graph in lockstep.

## Mergeable Phases
1. **Inventory & Guardrails (Phase 0)**
   - Document the supported `metta.common` shim surface; add regression tests for `metta.common.test_support` and the helper APIs that other packages import.
   - Normalize `pyproject.toml` dependency groups, ensure `py.typed` markers, and clean coverage/CI config so every package has a consistent baseline before refactors.
2. **Decouple the Hot Spots (Phase 1)**
   - Relocate `metta.tools.utils.auto_config` (and similar helpers) so CLI bootstrap code in `setup/` stops importing deep training modules.
   - Extract shared eval/sim configuration objects into a neutral module, letting `rl/` depend on lightweight interfaces instead of `metta.eval` and `metta.sim` internals.
   - Move adaptive-aware utilities out of `metta/utils` to break the unexpected `utils -> adaptive` coupling.
3. **Extract `metta-core` (Phase 2)**
   - After Phase 1 removes downstream dependencies, consolidate `utils/`, `tests_support/`, and stable helpers (including the `metta/common` shim) into a new top-level package with its own `pyproject.toml`.
   - Update imports across the workspace to consume the new `metta-core` namespace while keeping the old shim temporarily re-exported.
4. **Extract `metta-training` Monolith (Phase 3)**
   - Migrate the strongly connected set—`rl/`, `sim/`, `tools/`, `cogworks/`, and training-facing slices of `adaptive/` + `sweep/`—into a dedicated package. Preserve shared entrypoints (e.g., `metta/tools/train.py`) and verify the package exposes the APIs required by `app_backend/`, `experiments/`, and `devops/`.
   - Provide thin adapters for CLI and backend clients so the rest of the repo stops importing deep internals.
5. **Platform Packages & Cleanup (Phase 4)**
   - With training centralized, carve out `metta-cli` (`setup/`), `metta-maptools` (`gridworks/`, `map/`), and potential `metta-eval` and `metta-orchestrator` packages as independent follow-ups.
   - Remove superseded stubs (`mettagrid/`, `cogames/`), tighten documentation, and drop compatibility shims once import callers are migrated.

Each phase is scoped to a mergeable chunk that keeps the tree buildable; we can pause between phases without breaking consumers.

## Refined Roadmap
1. **Harden workspace packages**
   - Ensure each existing workspace member declares its `metta.*` namespace, ships `py.typed`, and has current dependency groups in its `pyproject.toml`.
   - Align CI, publishing scripts, and documentation with the `metta-*` names we actively ship (`metta-agent`, `metta-app-backend`, `metta-common`, `gitta`, `softmax`).
2. **Reduce high-risk coupling before we split**
   - Relocate `metta.tools.utils.auto_config` (and friends) so CLI bootstrapping no longer forms a cycle between `setup/` and `tools/`, nor forces `rl/` to import `tools/` for configuration (`metta/rl/checkpoint_manager.py:1-40`).
   - Move shared evaluation config artifacts (`metta/eval/eval_request_config.py`, `metta/eval/analysis_config.py`) into the training package or a new neutral module to break the `rl -> eval` dependency while keeping `eval/` free to depend on `rl/`.
   - Extract simulation config/data objects (`metta/sim/simulation_config.py`, `metta/sim/utils.py`) that `rl/` consumes into a shared neutral layer so `sim/` can remain downstream of `rl/`.
   - Split `metta/utils/live_run_monitor.py` (and any other adaptive-aware helpers) into a training/ops-oriented module, keeping the base `utils/` package dependency-light.
   - Catalogue and cap the `metta/common` import surface the shim must preserve; add tests guarding `metta.common.test_support` and other exported symbols before we move code.
3. **Extract packages in dependency order**
   - **`metta-core`**: start with pruned `utils/`, `tests_support/`, and other low-level helpers once they are free of training/adaptive references; keep the `metta/common` shim here until its consumers migrate.
   - **`metta-training`**: carve out the strongly connected set—`rl/`, `sim/`, `tools/`, and `cogworks/`—plus training-facing pieces of `adaptive/` and `sweep/`. Because these directories share runtime assumptions, extract them together behind a single `metta.training` namespace before attempting finer-grained splits.
   - **`metta-orchestrator`**: after `metta-training` exists, isolate orchestration code (`adaptive/`, `sweep/`, dispatcher components) that survives the refactors, exposing stable hooks for CLI and devops callers.
   - **`metta-cli`**: move `setup/` once its dependencies on `metta-training` flow through thin APIs (no direct import of deep training modules). Align CLI entry points with the new packages and keep bootstrapping logic local.
   - **`metta-maptools`**: group `gridworks/` and `map/` once they depend only on `metta-common`, `metta-core`, and the public `metta-training` APIs.
   - Optional follow-up: evaluate whether `eval/` can stand alone as `metta-eval` after configs and checkpoint hooks move to `metta-training`.
4. **Adopt the split**
   - Update imports to target the new namespaces, adjust workspace membership in `pyproject.toml`, and delete emptied folders under `metta/`.
   - Refresh CI, release automation, and docs so they reference the new package boundaries and dependencies (`uv` workspace membership, coverage config, ops runbooks).

## Open Questions
- Where should the `metta.common.test_support` API ultimately live once `metta-core` exists? (Candidates: stay in `metta-core`, or move next to the downstream test suites that rely on it.)
- How do we expose training configuration (`auto_config`, eval request config, simulation config) so that CLI tooling, `app_backend/`, and experiments can consume it without re-entangling the split?
- Do we want a dedicated shared layer for `mettagrid` helpers to avoid duplicating that dependency in every new package?
- Should any part of `softmax/` merge into the new `metta-core`, or does it remain a separate surface for external branding?
