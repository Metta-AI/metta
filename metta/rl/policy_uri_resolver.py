"""Policy URI resolution for metta:// and :latest URIs."""

import uuid
from pathlib import Path
from typing import TypedDict

import boto3

from metta.tools.utils.auto_config import auto_stats_server_uri
from mettagrid.util.file import ParsedURI


class PolicyMetadata(TypedDict):
    run_name: str
    epoch: int
    uri: str


def _extract_run_and_epoch(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    if ":v" in stem:
        run_name, suffix = stem.rsplit(":v", 1)
        if run_name and suffix.isdigit():
            return (run_name, int(suffix))
    return None


def key_and_version(uri: str) -> tuple[str, int] | None:
    """Extract run name and epoch from a policy URI."""
    parsed = ParsedURI.parse(uri)
    if parsed.scheme == "mock" and parsed.path:
        return (parsed.path, 0)
    if parsed.scheme == "file" and parsed.local_path:
        file_path = Path(parsed.local_path)
    elif parsed.scheme == "s3" and parsed.key:
        file_path = Path(parsed.key)
    else:
        raise ValueError(f"Could not extract key and version from {uri}")

    return _extract_run_and_epoch(file_path)


def _get_all_checkpoints(uri: str) -> list[PolicyMetadata]:
    parsed = ParsedURI.parse(uri)
    if parsed.scheme == "file" and parsed.local_path:
        checkpoint_files = [ckpt for ckpt in parsed.local_path.glob("*.mpt") if ckpt.stem]
    elif parsed.scheme == "s3" and parsed.bucket:
        s3_client = boto3.client("s3")
        prefix = parsed.key or ""
        response = s3_client.list_objects_v2(Bucket=parsed.bucket, Prefix=prefix)

        if response["KeyCount"] == 0:
            return []

        checkpoint_files = [Path(obj["Key"]) for obj in response["Contents"] if obj["Key"].endswith(".mpt")]
    else:
        raise ValueError(f"Cannot get checkpoints from uri: {uri}")

    checkpoint_metadata: list[PolicyMetadata] = []
    for path in checkpoint_files:
        run_and_epoch = _extract_run_and_epoch(path)
        if run_and_epoch:
            path_uri = uri.rstrip("/") + "/" + path.name
            metadata: PolicyMetadata = {
                "run_name": run_and_epoch[0],
                "epoch": run_and_epoch[1],
                "uri": path_uri,
            }
            checkpoint_metadata.append(metadata)

    return checkpoint_metadata


def get_latest_checkpoint(uri: str) -> PolicyMetadata | None:
    checkpoints = _get_all_checkpoints(uri)
    if checkpoints:
        return max(checkpoints, key=lambda p: p["epoch"])
    return None


def _resolve_metta_uri(uri: str) -> str:
    """Resolve a metta:// URI to its underlying storage URI.

    Supported formats:
    - metta://policy/<policy_version_id> - resolves via observatory API to s3:// path
    """
    parsed = ParsedURI.parse(uri)
    if parsed.scheme != "metta" or not parsed.path:
        raise ValueError(f"Invalid metta:// URI: {uri}")

    path_parts = parsed.path.split("/")
    if len(path_parts) < 2 or path_parts[0] != "policy":
        raise ValueError(f"Unsupported metta:// URI format: {uri}. Expected metta://policy/<policy_version_id>")

    policy_version_id_str = path_parts[1]
    try:
        policy_version_id = uuid.UUID(policy_version_id_str)
    except ValueError as e:
        raise ValueError(f"Invalid policy version ID in URI: {policy_version_id_str}") from e

    stats_server_uri = auto_stats_server_uri()
    if not stats_server_uri:
        raise ValueError("Cannot resolve metta:// URI: stats server not configured")

    from metta.app_backend.clients.stats_client import StatsClient

    stats_client = StatsClient.create(stats_server_uri)
    policy_version = stats_client.get_policy_version(policy_version_id)

    if not policy_version.s3_path:
        raise ValueError(f"Policy version {policy_version_id} has no s3_path")

    return policy_version.s3_path


def resolve_uri(uri: str) -> str:
    """Resolve a policy URI to a loadable path.

    Handles:
    - metta://policy/<uuid> - resolves via Observatory API
    - path/to/checkpoints:latest - resolves to newest checkpoint
    - file:// and s3:// URIs - normalized to canonical form
    - Plain paths - converted to file:// URIs
    """
    parsed = ParsedURI.parse(uri)

    # Resolve metta:// URIs first (they may resolve to URIs with :latest)
    if parsed.scheme == "metta":
        resolved_uri = _resolve_metta_uri(uri)
        return resolve_uri(resolved_uri)

    # Resolve :latest suffix
    if uri.endswith(":latest"):
        base_uri = uri[:-7]
        latest = get_latest_checkpoint(base_uri)
        if not latest:
            raise ValueError(f"No latest checkpoint found for {base_uri}")
        return latest["uri"]

    return parsed.canonical


def get_policy_metadata(uri: str) -> PolicyMetadata:
    """Get metadata (run_name, epoch, uri) from a policy URI."""
    normalized_uri = resolve_uri(uri)
    metadata = key_and_version(normalized_uri)
    if not metadata:
        raise ValueError(f"Could not extract metadata from uri {uri}")
    run_name, epoch = metadata
    return {"run_name": run_name, "epoch": epoch, "uri": normalized_uri}
