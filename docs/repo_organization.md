# Softmax Repository Organization – Blueprint (Sep 27, 2025)

We completed Phase 0 (Inventory & Guardrails) and Phase 1 (Decouple the Hot Spots) under the former `metta` name. The
repository rename to **`softmax`** lands at the start of Phase 2, and this document captures the updated structure,
dependencies, and migration steps requested in the latest review cycle.

## Snapshot After Phase 1
- Packages shipping today (with `pyproject.toml` in workspace): `agent/`, `app_backend/`, `codebot/`, `common/`,
  `config/`, `experiments/`, `gitta/`, `packages/cogames/`, `packages/mettagrid/`, `packages/pufferlib-core/`,
  `shared/`, and `softmax/`.
- Monolithic `metta/` namespace still hosts `adaptive/`, `cogworks/`, `common/`, `config/`, `eval/`, `gridworks/`, `map/`,
  `rl/`, `setup/`, `shared/`, `sim/`, `sweep/`, `tests_support/`, `tools/`, and `utils/`. Shims under `metta.common`
  continue to protect downstream consumers until we ship the new `softmax.lib` surface.
- Dependency hot spots we just mitigated: `setup → tools` cycle removed, `metta.shared` introduced as shared data hub,
  adaptive live-run monitor now isolated from `metta.utils`.

## Decisions From Phase 1 Retro & Engineer Feedback
- **Repository rename**: land `softmax` as the canonical root; `metta` survives only as compatibility namespaces until
  depreciation milestones clear.
- **`src/` everywhere**: every Python distribution (internal or external) must adopt `src/` layout with a single
  `pyproject.toml` per package. No code lives beside configuration/tests at the package root.
- **Central library**: `metta.common` graduates into `softmax.lib`, exposing the stable runtime surface while continuing
  to re-export `metta.common` for a limited sunset period.
- **Package homing**: all Python packages (library or service) live under `packages/` to simplify workspace discovery and
  tooling. Exception: top-level apps (`agent/`, `app_backend/`, `experiments/`) that ship deployable artifacts can stay
  top-level but still use `src/`.
- **Dependency gates**: every package owns its dependency list in `pyproject.toml`; CI enforces `uv run --exact -m pytest`
  (or the package’s bespoke test entrypoint) before merge. No implicit cross-package imports without declared extras.
- **Implicit namespaces**: evaluate PEP 420 namespace adoption and remove redundant `__init__.py` once the new package
  boundaries are in place and tooling (mypy, pytest discovery, packaging) is confirmed compatible.
- **Cogworks scope**: redefine `cogworks` as the gameplay content SDK (scenario authoring, asset bundling, validation
  tools). Anything that is engine runtime or orchestration moves out to `softmax.training` or `softmax.maptools`.
- **Dependency graph clarity**: document and maintain the allowed import edges so Phase 2 refactors don’t recreate cycles.

## Packaging Principles (effective Phase 2)
- One directory per installable: `packages/<name>/pyproject.toml` and `packages/<name>/src/<python namespace>/...`.
- Public API published through `__all__` / curated `__init__.py` files; internal modules remain private.
- Each package lists runtime dependencies, optional extras, and development dependencies explicitly. Cross-package usage
  must go through the published API and declare the dependency.
- Every package supplies `py.typed`, type-checks with `mypy`, formats with `ruff format`, linted by `ruff check`, and
  tests via `uv run --exact` in CI.
- Compatibility shims (`metta.*`) live only inside `packages/softmax-lib` during Phase 2; Phase 3 removes them outright as a deliberate breaking change.

## Target Package Inventory & Responsibilities

| Package | Location | Namespace | Responsibilities | Depends On | Notes |
| --- | --- | --- | --- | --- | --- |
| softmax-lib | `packages/softmax-lib` | `softmax.lib` (`metta.common` shim) | Core runtime helpers, data models, test support; hosts compatibility exports | `softmax.config`, `softmax.shared` | Replaces `metta/common`; Phase 3 deletes the shim + legacy namespace |
| softmax-config | `packages/softmax-config` | `softmax.config` | Auto-configuration, environment detection, credential plumbing | none | Break lazy imports; expose typed interfaces |
| softmax-shared | `packages/softmax-shared` | `softmax.shared` | Evaluation/simulation schemas, registries | none | Already stands alone; fold policy registry here |
| softmax-training | `packages/softmax-training` | `softmax.training` | RL loops, simulation drivers, orchestration CLI, training tools | `softmax.lib`, `softmax.config`, `softmax.shared`, `packages/mettagrid` | Absorbs `rl/`, `sim/`, `tools/`, training slices of `adaptive/` & `sweep/` |
| softmax-cogworks | `packages/softmax-cogworks` | `softmax.cogworks` | Content authoring SDK, asset validators, scenario packaging | `softmax.lib`, `softmax.shared` | Only gameplay content code lives here |
| softmax-maptools | `packages/softmax-maptools` | `softmax.maptools` | Grid/map editors, browser clients, static asset pipeline | `softmax.lib` | Hosts former `gridworks/`, `map/`, relevant front-ends |
| softmax-cli | `packages/softmax-cli` | `softmax.cli` | Developer setup, local orchestration commands, bootstrap scripts | `softmax.lib`, `softmax.config`, `softmax.training` | Supersedes `setup/` & CLI glue |
| softmax-orchestrator | `packages/softmax-orchestrator` | `softmax.orchestrator` | Task scheduling, adaptive dispatch, integrations | `softmax.lib`, `softmax.training` | Optional deployable service |
| Packages under `packages/` (external) | `packages/*` | varies | Third-party engines (`cogames`, `mettagrid`, `pufferlib_core`) | - | Continue to vendor here with `src/` layout |
| Apps | top level (`agent/`, `app_backend/`, `experiments/`) | `softmax.agent`, etc. | Deployment surfaces; consume published APIs only | Declared extras | Must adopt `src/` and declare deps |

## Dependency Guardrails
- `softmax.lib` is the lowest layer. Higher-level packages may depend on it; it depends only on the standard library plus
  `softmax.config`/`softmax.shared`.
- `softmax.config` and `softmax.shared` are peer foundations—no package may import from higher layers.
- `softmax.training`, `softmax.cogworks`, and `softmax.maptools` sit at the middle layer, depending only on the
  foundations and external vendor packages.
- `softmax.cli` and `softmax.orchestrator` sit at the top and may depend on any lower layer.
- CI enforces dependency correctness by running `uv run --exact` with `--frozen` lockfiles per package plus import
  scanners (TBD) to block forbidden edges.

## PEP 420 & `__init__.py` Strategy
1. Consolidate packages into the `packages/` hierarchy with `src/softmax/...` namespaces.
2. Convert compatibility-only directories (`metta/`) into namespace packages by deleting redundant `__init__.py` files
   once all direct imports route through `softmax.*`.
3. Retain `__init__.py` only where the file defines public API (`softmax/lib/__init__.py`, etc.) or carries backward
   warnings. Document the removal plan and validate with `mypy --namespace-packages`, `pytest`, and packaging builds.

## Cogworks Scope Clarification
- Lives entirely inside `softmax.cogworks`.
- Owns authoring tools, simulation fixtures specific to content, asset validation, and exporter pipelines.
- Does **not** contain runtime trainers, orchestration loops, or experiment harnesses—those belong to
  `softmax.training`.
- Collaborates with `softmax.training` via typed request/response models defined in `softmax.shared`.

## Phase Roadmap (Recast Post-Rename)
- **Phase 2 – Extract `softmax-lib` & Foundations (in flight)**
  - Move `metta/utils`, `metta/tests_support`, and shim logic into `packages/softmax-lib/src/softmax/lib`.
  - Break residual `softmax.config` ↔ `setup` couplings; surface interfaces in `softmax.config`.
  - Validate `softmax-lib`, `softmax-config`, and `softmax-shared` independently with `uv run --exact -m pytest`.
- **Phase 3 – Carve Out Training & Cogworks (breaking change)**
  - Relocate `rl/`, `sim/`, `tools/`, and training portions of `adaptive/` & `sweep/` into `packages/softmax-training` with no residual modules left in `metta/`.
  - Migrate gameplay authoring modules into `packages/softmax-cogworks`; keep only content authoring APIs there.
  - Remove every `metta.*` shim related to these packages and accept the breaking import change across the monorepo.
  - Update dependency declarations and enforce public-entry-point imports.
- **Phase 4 – Surface Apps & Tooling**
  - Graduate `setup/` to `packages/softmax-cli`; move orchestration services into `softmax-orchestrator`.
  - Migrate map/grid tooling and web assets into `softmax-maptools`; ensure all front-end build pipelines reference the
    new paths.
  - Remove remaining compatibility shims and prune unused `metta` directories.

## Action Checklist
- [ ] File rename & tooling: update repo metadata, CI, and documentation to reference `softmax`.
- [ ] Create `packages/softmax-lib`, `packages/softmax-config`, and `packages/softmax-shared` with `src/` layout; migrate
      code and wire `metta.common` shim (to be deleted in Phase 3).
- [ ] Define dependency allow-list rules and codify them in CI (import linter + `uv run --exact`).
- [ ] Pilot PEP 420 by removing `__init__.py` from `metta/` shim directories once the new packages export equivalents.
- [ ] Draft `softmax.cogworks` charter and audit modules to move; schedule migrations in Phase 3 tasks.
- [ ] Update onboarding docs so teams install/test packages using `uv run --exact` commands per package.
- [ ] Track downstream adoption; delete the remaining shims as part of the Phase 3 breaking change and broadcast the new import paths.

## Open Questions
- Do we want a dedicated vendor strategy for `mettagrid` artifacts (wheel vs. submodule) before we cut
  `softmax.training`?
- Which packages require optional GPU / RL extras, and how do we model them without violating dependency guardrails?
- Should `softmax.maptools` ship as a Python package plus a bundled front-end build, or do we split the web clients into
  a separate repository once the rename completes?
