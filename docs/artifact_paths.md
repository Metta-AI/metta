# Artifact Path Helpers

Metta stores checkpoints, stats, and replays across local paths and a couple of URI schemes (today that mostly means
`s3://` and `gdrive://`). To keep the string fiddling in one place we expose a tiny set of helpers in
`packages/mettagrid/python/src/mettagrid/util/uri.py`.

## Core helpers

| Helper | Purpose |
| ------ | ------- |
| `artifact_join(base, *segments)` | Concatenate path segments onto a base path/URI while handling S3/GDrive quirks. |
| `artifact_policy_run_root(base, run_name, epoch)` | Convenience helper that appends `run_name` and optional `v{epoch}`. |
| `artifact_simulation_root(base, suite, name, simulation_id=None)` | Builds the standard `<suite>/<name>/<id>` sub-directory used for replays. |

### Example

```python
from mettagrid.util.uri import artifact_join, artifact_policy_run_root

run_root = artifact_policy_run_root("s3://softmax-public/replays", run_name="my_run", epoch=5)
sim_root = artifact_join(run_root, "navigation", "maze", "episode123.json.z")
print(sim_root)
# -> s3://softmax-public/replays/my_run/v5/navigation/maze/episode123.json.z
```

## Guidelines

* Call `artifact_join` instead of manual `f"{prefix}/{suffix}"` string operations—this keeps bucket/key handling and
  trailing slash stripping consistent.
* `artifact_policy_run_root` and `artifact_simulation_root` are just thin wrappers around `artifact_join`; use them where
  they make intent clearer (e.g., when building replay destinations).
* If you need to support a new URI scheme, extend `artifact_join` so everything continues to flow through the same code
  path.

## Current usage

- `CheckpointManager` uses `artifact_join` when uploading checkpoints to remote storage.
- `Simulation`, `SimTool`, and `EvalService` rely on `artifact_policy_run_root` / `artifact_simulation_root` for replay
  directories.
- `ReplayWriter` calls `artifact_join` when writing individual episodes.

If you add new tooling that needs to manipulate artifact paths, prefer these helpers over hand-written string splicing.
