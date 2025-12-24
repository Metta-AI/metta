# URI Resolvers

This package provides a pluggable URI resolution system for handling different resource schemes.

Checkpoint URIs point at a checkpoint directory (or `.zip` archive) containing `policy_spec.json`.

## Usage

```python
from mettagrid.util.uri_resolvers.schemes import parse_uri, resolve_uri, get_checkpoint_metadata

# Parse a URI to get its components
parsed = parse_uri("s3://bucket/path/to/run:v5")
print(parsed.scheme)  # "s3"
print(parsed.bucket)  # "bucket"
print(parsed.key)     # "path/to/run:v5"

# Get checkpoint info (run_name, epoch) from parsed URI
info = parsed.checkpoint_info  # ("run_name", 5) or None
if info:
    run_name, epoch = info

# Resolve a URI (normalizes and finds latest checkpoint if applicable)
parsed = resolve_uri("file:///path/to/checkpoints:latest")
print(parsed.canonical)  # "file:///path/to/checkpoints/run:v5" or ".zip"

# Get full checkpoint metadata (resolves URI first)
metadata = get_checkpoint_metadata("s3://bucket/checkpoints/my-run:v5")
print(metadata.run_name)  # "my-run"
print(metadata.epoch)     # 5
print(metadata.uri)       # resolved URI
```
