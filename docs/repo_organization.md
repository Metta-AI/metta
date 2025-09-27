# Metta Repository Organization – Forward Plan (Sep 2025)

## Baseline Snapshot
- Workspace Python packages we ship today: `agent/` (`metta-agent`), `app_backend/` (`metta-app-backend`), `codebot/`, `common/` (`metta-common`), `config/` (`metta-config` – new neutral auto-config layer), `experiments/`, `gitta/`, `packages/cogames/`, `packages/mettagrid/`, `packages/pufferlib-core/`, `shared/` (`metta-shared` – new data-model hub), and `softmax/`.
- Front-end and client surfaces live in `gridworks/`, `mettascope/`, `observatory/`, `library/`, `home/`, and `cogweb/`; shared scenes and assets stay in `resources/` and `scenes/`.
- Tooling, automation, and infra live under `devops/`, `docs/`, `scripts/`, `tools/`, `mcp_servers/`, and `train_dir/`.
- The legacy `metta/` namespace still aggregates `adaptive/`, `cogworks/`, `common/`, `config/`, `eval/`, `gridworks/`, `map/`, `rl/`, `setup/`, `shared/`, `sim/`, `sweep/`, `tests_support/`, `tools/`, and `utils/`. The `metta/common` compatibility shim stays in place while the new `metta-config` and `metta-shared` packages absorb shared helpers.

| Subdir (metta/*) | .py files | Outgoing cross-subpkg refs | Incoming refs |
| --- | ---: | ---: | ---: |
| adaptive | 12 | 5 | 7 |
| cogworks | 9 | 0 | 6 |
| common | 6 | 0 | 73 |
| config | 2 | 6 | 10 |
| eval | 7 | 16 | 3 |
| gridworks | 6 | 9 | 0 |
| map | 2 | 2 | 0 |
| rl | 43 | 38 | 14 |
| setup | 37 | 19 | 3 |
| shared | 4 | 0 | 19 |
| sim | 11 | 14 | 8 |
| sweep | 6 | 3 | 2 |
| tests_support | 2 | 2 | 0 |
| tools | 9 | 52 | 4 |
| utils | 4 | 0 | 17 |

Numbers capture import edges between metta subdirectories (self-imports omitted). New packages (`config/`, `shared/`) are already absorbing references from `tools/`, `rl/`, and `setup/`.

## Phase Progress (Sep 27, 2025)
- ✅ **Phase 0 – Inventory & Guardrails**: `docs/metta_common_shim.md`, shim regression tests (`tests/metta_common/test_shim.py`), and `py.typed` coverage landed on `richard-phase0`.
- ✅ **Phase 1 – Decouple the Hot Spots** (merged into `richard-repo`):
  - Created `metta.config` for auto-config helpers; all callers now import via `metta.config.auto_config`, eliminating the direct `setup → tools` dependency loop.
  - Centralized shared evaluation / simulation models and policy registration in `metta.shared`, with compatibility re-exports in `metta.eval` and `metta.sim`.
  - Relocated the adaptive live-run monitor to `metta/adaptive/live_run_monitor.py`, leaving `metta.utils` dependency-light.
- ☐ **Phase 2 – Extract `metta-core`**: consolidate remaining baseline utilities and the `metta.common` shim into a standalone package once we finish pruning adaptive/training references.
- ☐ **Phase 3 – Extract `metta-training`**: carve out `rl/`, `sim/`, `tools/`, `cogworks/`, and training slices of `adaptive/`/`sweep/` behind a single namespace, exposing adapters for CLI/backend consumers.
- ☐ **Phase 4 – Platform Packages & Cleanup**: extract `metta-cli`, `metta-maptools`, optional `metta-eval`/`metta-orchestrator`, then retire compatibility shims and stubs.

## Dependency Hot Spots & Caveats
- **`metta.common` remains the shared spine**: 73 inbound references plus external usage (`app_backend`, `devops`, `softmax`). The shim contract in `docs/metta_common_shim.md` and the regression tests stay in force until every consumer migrates to `metta-config`/`metta-shared` APIs.
- **`tools/` is still the runtime hub**: 52 outgoing references, now primarily into `metta.config`, `metta.shared`, `metta.rl`, and orchestration code. Even after Phase 1, we must move the remaining high-traffic entrypoints into the future `metta-training` package.
- **`metta.config ↔ metta.setup` coupling**: `metta.config.auto_config` lazily imports setup components (`AWSSetup`, `WandbSetup`, `ObservatoryKeySetup`) while `metta.setup.mettta_cli` consumes the new API. Before extracting `metta-core`, we must either inject these dependencies or expose thin interfaces so `metta.config` no longer instantiates setup modules directly.
- **`metta.shared` is the new data-model choke point**: 19 inbound references from `rl`, `tools`, `eval`, and `sim` now rely on the shared evaluation models and policy registry helpers. We need a published API (likely via `metta-core`) before retiring compatibility shims.
- **`metta.shared` adoption**: `rl`, `tools`, `eval`, and `sim` now depend on shared data models and policy registry helpers. Any follow-on splits must publish these modules (likely in `metta-core`) to avoid reintroducing cross-package imports.
- **Remaining strongly connected set**: `tools → config → setup` still forms a small cycle even though the direct `setup → tools` import is gone. Breaking the lazy imports (e.g., via dependency injection) will be a prerequisite for Phase 2.
- **`mettagrid` dependency**: Training and evaluation workflows continue to rely on `mettagrid.config`, `mettagrid.profiling`, and `mettagrid.util`. When carving out packages we need a clear plan for packaging/distributing these bindings.

## Mergeable Phases (refreshed)
1. **Inventory & Guardrails (Phase 0)** – ✅ complete.
2. **Decouple the Hot Spots (Phase 1)** – ✅ complete (see progress above).
3. **Extract `metta-core` (Phase 2)**
   - Move `utils/`, `tests_support/`, and the compatibility layer for `metta.common` into a standalone package once adaptive/training references are gone.
   - Publish the `metta.shared` and `metta.config` APIs from this package, keeping backward-compat shims until downstream imports migrate.
4. **Extract `metta-training` (Phase 3)**
   - Relocate the strongly connected training cluster (`rl`, `sim`, `tools`, `cogworks`, plus slices of `adaptive`/`sweep`) behind a dedicated namespace.
   - Provide stable adapters so `app_backend`, CLI tools, and experiments stop importing private modules.
5. **Platform Packages & Cleanup (Phase 4)**
   - Split `setup/` (`metta-cli`), `gridworks/` + `map/` (`metta-maptools`), and optional `metta-eval` / `metta-orchestrator` packages.
   - Remove temporary shims (`metta.common`, `metta.eval.*`, `metta.sim.*`) and delete root-level stubs once imports are updated.

Each remaining phase is scoped to a mergeable chunk, preserving a buildable tree and allowing pause points between phases.

## Refined Roadmap (near-term focus)
1. **Stabilize shared layers**
   - Finalize the API surface of `metta.config` (eliminate lazy imports where possible) and document usage for downstream teams.
   - Expand tests around `metta.shared` models (e.g., replay helpers, WandB metrics) to protect the new contract before relocating code again.
2. **Prepare for `metta-core` extraction**
   - Evict any adaptive/training references left in `metta.utils` or `metta.tests_support`.
   - Introduce a thin compatibility package (or re-export) so the eventual `metta-core` wheel can ship `metta.common` without the legacy shim.
3. **Design `metta-training` boundaries**
   - Define public entrypoints for CLI tools and backend services (e.g., `TrainTool`, `SimTool`, remote eval helpers) to expose once training moves into its own package.
   - Decide how `mettagrid` dependencies will be vendored or declared across the new packages.
4. **Adopt the split**
   - Update `pyproject.toml` workspace membership, CI scripts, and release automation as each package lands.
   - Remove compatibility shims after downstream imports are refactored and validated.

## Target Post-Reorg Layout

Once Phases 2–4 are complete, the repository will be organized as a set of installable packages that expose stable
`metta.*` namespaces. Each package owns its source, tests, and configuration while the legacy monolith shrinks to a set
of compatibility shims scheduled for removal. The intended layout is:

```
metta/
├── core/                  # metta-core: foundational runtime utilities and compatibility shims
│   ├── pyproject.toml
│   └── src/metta/core/
│       ├── common/        # long-lived shim for legacy imports, removed once consumers migrate
│       ├── tests_support/
│       ├── utils/
│       └── shared/        # re-export curated APIs from metta.shared
├── training/              # metta-training: RL, simulation, CLI entrypoints for training
│   ├── pyproject.toml
│   └── src/metta/training/
│       ├── rl/
│       ├── sim/
│       ├── tools/
│       ├── cogworks/
│       ├── sweep/
│       └── adaptive/      # training-oriented components only
├── orchestrator/          # metta-orchestrator: adaptive dispatchers & scheduling backends
│   ├── pyproject.toml
│   └── src/metta/orchestrator/
│       ├── adaptive/
│       ├── scheduler/
│       └── integrations/
├── cli/                   # metta-cli: developer setup and local tooling
│   ├── pyproject.toml
│   └── src/metta/cli/
│       ├── setup/
│       └── commands/
├── maptools/              # metta-maptools: editors, map generation, web front-ends
│   ├── pyproject.toml
│   └── src/metta/maptools/
│       ├── gridworks/
│       ├── map/
│       └── web/
├── eval/                  # metta-eval (optional): evaluation services & reporting
│   ├── pyproject.toml
│   └── src/metta/eval/
│       ├── service/
│       ├── analysis/
│       └── reporting/
├── config/                # metta-config (existing): shared auto-configuration helpers
│   ├── pyproject.toml
│   └── src/metta/config/auto_config.py
└── shared/                # metta-shared (existing): shared data structures & registries
    ├── pyproject.toml
    └── src/metta/shared/
        ├── eval_config.py
        ├── simulation_config.py
        └── policy_registry.py

agent/                     # metta-agent workspace package (model architectures)
app_backend/               # metta-app-backend workspace package
codebot/
common/                    # metta-common (thin distribution that depends on metta-core)
experiments/
packages/                  # external engines (mettagrid, cogames, pufferlib-core)
softmax/
```

Guiding rules for the final layout:

- **Single responsibility**: each package owns a coherent surface area, minimizing cross-package imports.
- **Published APIs**: every package exposes curated entrypoints via `__init__.py`; compatibility modules live only in
  `metta-core` and carry explicit deprecation warnings.
- **Workspace alignment**: all packages are listed in `[tool.uv.workspace]`, ship `py.typed`, and execute their own
  targeted test suites in CI.
- **Temporary shims with exit criteria**: legacy modules (`metta.common`, `metta.eval`, `metta.sim`) re-export from the
  new packages until downstream consumers migrate. Each shim has an agreed deprecation schedule tracked in rollout docs.
- **Shared dependency strategy**: `metta-config` and `metta-shared` remain the canonical homes for reusable configuration
  and data models, preventing future cycles as the repo splits.

## Open Questions
- Where should the `metta.common.test_support` API ultimately live once `metta-core` exists? (Likely inside the new package, but we may prefer a dedicated testing helper module.)
- How do we expose training configuration (`auto_config`, eval request config, simulation config) so that CLI tooling, `app_backend/`, and experiments consume the published APIs without re-entangling splits?
- Do we want a dedicated distribution for `mettagrid` helpers to avoid repeating that dependency across packages?
- Should any parts of `softmax/` fold into `metta-core`, or does it remain separately branded?
