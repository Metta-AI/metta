# Artifact Path Helpers

Metta stores checkpoints, stats, and replays across multiple backends (local disk, S3, mock URIs, Google Drive). All
code that constructs those paths should go through the helpers in
`packages/mettagrid/python/src/mettagrid/util/artifact_paths.py` so the layout stays consistent and scheme-specific edge
cases stay in one place.

## Core API

| Helper | Purpose |
| ------ | ------- |
| `ensure_artifact_reference(value)` | Wraps a `str`/`Path`/`ArtifactReference` and returns a normalized `ArtifactReference`. Rejects empty strings. |
| `ArtifactReference.join(*segments)` | Append path components in an environment-aware way (handles `Path`, `s3://`, `gdrive://`, etc.). |
| `ArtifactReference.with_policy(run_name, epoch)` | Convenience wrapper to nest runs under optional `run_name` and `epoch` (adds `v{epoch}` when provided). |
| `ArtifactReference.with_simulation(suite, name, simulation_id=None)` | Append simulation metadata using the standard `<suite>/<name>/<id>` layout. |
| `artifact_policy_run_root(base, run_name, epoch)` | Helper for policy replay roots. Returns an `ArtifactReference` or `None`. |
| `artifact_simulation_root(base, suite, name, simulation_id=None)` | Helper for simulation replay directories. |
| `PolicyArtifactLayout.build(...)` | Precomputes checkpoint and replay roots and can hand out paths for simulations/checkpoints. |
| `ArtifactRef` | A Pydantic-compatible wrapper that normalizes artifact strings and exposes `.as_reference()` / `.join()`. |

### Example

```python
from mettagrid.util.artifact_paths import ensure_artifact_reference

run_root = ensure_artifact_reference("s3://softmax-public/replays").with_policy("my_run", epoch=5)
sim_root = run_root.with_simulation("navigation", "maze", simulation_id="abc123")
print(sim_root.as_str())
# -> s3://softmax-public/replays/my_run/v5/navigation/maze/abc123
```

## Usage Guidelines

* Always call `ensure_artifact_reference` (or the higher-level helpers) as soon as you ingest a user/configured path.
  Empty strings now raise immediately, preventing silent fallbacks later in the pipeline.
* Use `ArtifactReference.join` / `with_policy` / `with_simulation` instead of manual string concatenation (e.g.
  `f"{prefix}/{run}/{suite}"`). These helpers normalize trailing slashes and handle scheme-specific quirks such as S3
  bucket roots or Google Drive prefixes.
* When you need a raw string for upload functions (`write_file`, `http_url`, etc.), call `ref.as_str()` on an
  `ArtifactReference` (or keep using `ArtifactRef`, which is already a string) so conversion logic stays centralized.
* Configuration utilities (`auto_replay_dir`, CLI validators, etc.) should rely on `ensure_artifact_reference` to catch
  invalid overrides early.
* Use `artifact_settings()` when you need the canonical replay directory or remote prefix derived from environment/AWS
  configuration.

## Updated Call Sites

The helpers are already wired into the main replay stack:

- `CheckpointManager` now builds remote URIs via `PolicyArtifactLayout` and `ArtifactReference.join`.
- `Simulation` / `EvalService` / `SimTool` construct replay directories through `PolicyArtifactLayout` or
  `ArtifactReference` helpers.
- `ReplayWriter` accepts `ArtifactReference` objects directly and emits canonical strings via `ref.as_str()`.

If you add new tooling that writes artifacts, reuse these helpers instead of reinventing the path handling.
