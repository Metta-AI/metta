# Metta Repository Organization – Forward Plan (Sep 2025)

## Baseline Snapshot
- Workspace Python packages shipped from the repo: `agent/` (`metta-agent`), `app_backend/` (`metta-app-backend`), `codebot/`, `common/` (`metta-common`), `experiments/`, `gitta/`, `packages/cogames/`, `packages/mettagrid/`, `packages/pufferlib-core/`, and `softmax/`.
- Front-end and client code now lives in `gridworks/`, `mettascope/`, `observatory/`, `library/`, `home/`, and `cogweb/`; shared scenes and assets remain under `resources/` and `scenes/`.
- Tooling, automation, and infra have settled into `devops/`, `docs/`, `scripts/`, `tools/`, `mcp_servers/`, and `train_dir/`.
- The legacy `metta/` namespace still aggregates `adaptive/`, `cogworks/`, `common/`, `eval/`, `gridworks/`, `map/`, `rl/`, `setup/`, `sim/`, `sweep/`, `tests_support/`, `tools/`, and `utils/`, plus the stub duplicates `mettagrid/` and `cogames/` at repo root.

## Phased Plan
1. **Harden workspace packages**
   - Ensure each package above declares `metta.*` namespaces, ships `py.typed`, and has up-to-date dependency groups in its `pyproject.toml`.
   - Align CI, publishing scripts, and docs with the `metta-*` distribution names we actively ship, including `gitta` and `softmax`.
2. **Retire stubs and dead paths**
   - Delete the root `mettagrid/` and `cogames/` shims once all consumers import from `packages/**`.
   - Collapse the residual `metta/common` shim into `common/src/metta/common` and scrub workspace references to the old path.
   - Audit documentation and scripts so the new front-end and tooling directories (`mcp_servers/`, `scripts/`, `train_dir/`) are the canonical locations.
3. **Split the `metta/` monolith into installable subpackages**
   - `metta-core`: `utils/`, `tools/`, `tests_support/`, lightweight helpers consumed across the stack.
   - `metta-training`: `cogworks/`, `rl/`, `sim/`, and coordinated pieces of `adaptive/` used for training loops.
   - `metta-eval`: everything in `eval/` plus reporting pipelines.
   - `metta-orchestrator`: `sweep/`, schedulers, deployment/adaptive controllers from `adaptive/`.
   - `metta-maptools`: `map/` and the Python portions of `metta/gridworks/`.
   - `metta-cli`: `setup/` and installer/CLI tooling.
   - For each subpackage: create a sibling directory with its own `pyproject.toml`, move code, add `metta/<subpackage>/__init__.py` exports, expose `py.typed`, and backfill tests.
4. **Adopt the split**
   - Update imports across the repo to target the new packages, adjust `pyproject.toml` workspace membership, and remove the emptied directories under `metta/`.
   - Refresh CI, release, and documentation paths once the new packages are live.

## Open Questions
- Do any external consumers still require the `metta.common` import path after the migration to `metta-common`?
- Should `softmax/` remain a separately branded distribution or fold into `metta-core` once the split is complete?
