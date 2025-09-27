# `metta.common` Compatibility Shim

The legacy `metta.common` import path is still widely used across backend services, developer tooling, and tests.  The
shim that lives in `metta/common/` keeps those imports working while the shared code is authored inside the
`metta-common` workspace package (`common/src/metta/common/`).  This file lists the elements that we commit to keeping
stable until every consumer has migrated to the new package layout.

## Supported Submodules

| Submodule | Purpose | Key Symbols Guaranteed |
| --- | --- | --- |
| `metta.common.util.constants` | Shared configuration constants used by CLIs, services, and devops tooling. | `METTA_WANDB_ENTITY`, `METTA_WANDB_PROJECT`, `SOFTMAX_S3_BUCKET`, `SOFTMAX_S3_BASE`, `DEV_STATS_SERVER_URI`, `PROD_STATS_SERVER_URI`, `METTA_AWS_ACCOUNT_ID`, `METTA_AWS_REGION`, `METTA_SKYPILOT_URL`. |
| `metta.common.util.fs` | Repo-aware filesystem helpers used by CLI setup and notebooks. | `get_repo_root`, `get_file_hash`, `cd_repo_root`. |
| `metta.common.util.collections` | Small collection helpers referenced by orchestration and backend code. | `group_by`, `remove_none_values`, `remove_none_keys`, `remove_falsey`, `duplicates`. |
| `metta.common.util.log_config` | Shared logging configuration used by CLIs and services. | `init_logging`, `getRankAwareLogger`. |
| `metta.common.util.heartbeat` | Heartbeat emission helpers for long-running tasks. | `record_heartbeat`. |
| `metta.common.util.retry` | Generic retry wrappers for devops scripts. | `retry_on_exception`, `retry_function`. |
| `metta.common.util.numpy_helpers` | Helper utilities for numpy-heavy workloads. | `batched_array`, `stack_dicts`, `merge_dicts`. |
| `metta.common.tool` | Base class used by CLI entrypoints. | `Tool`. |
| `metta.common.wandb.context` | W&B context helpers consumed by RL tools. | `WandbRun`, `WandbConfig`, `WandbContext`. |
| `metta.common.wandb.utils` | Abort helpers and common WandB helpers. | `abort_requested`. |
| `metta.common.datadog.tracing` | Datadog tracing bootstrap for services. | `init_tracing`, `trace`. |
| `metta.common.test_support` | Pytest fixtures and helpers relied on by repo-wide tests. | `docker_client_fixture`, `pytest_collection_modifyitems`, `isolated_test_schema_uri`. |
| `metta.common.silence_warnings` | Optional warning filters for legacy scripts. | `silence_warnings`. |

> **Note:** The shim also forwards any other `metta.common.*` import that resolves inside `metta-common`.  The list above
covers the modules that currently have active consumers (see the dependency audit in `docs/repo_organization.md`).

## Expectations Until Migration

- Imports must keep working for runtime code (`softmax.orchestrator`, `metta/tools/*`, `devops/*`, etc.) and for test fixtures.  If
  a symbol is moved, the shim must re-export it from its new location until every consumer has been updated.
- New helpers should be added directly to `metta-common`; the shim should stay minimal to make future removal easy.
- Changes to these APIs require coordination across workstreams and should land with tests proving the old import path
  still resolves.

## Regression Tests

The automated tests in `tests/metta_common/test_shim.py` verify that:

- `metta.common.test_support` exposes the expected fixtures.
- Representative util, tool, and wandb imports resolve and expose the symbols listed above.

Any breakage of those tests is a signal that a refactor invalidated the shim contract and needs additional migration work
before merging.
