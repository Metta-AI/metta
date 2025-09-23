"""Helpers for constructing artifact paths (local or remote)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from mettagrid.util.uri import ParsedURI

ArtifactBase = Union[str, Path]


def _clean_segments(segments: Iterable[str]) -> list[str]:
    return [seg.strip("/") for seg in segments if seg and seg.strip("/")]


def artifact_path_join(base: ArtifactBase, *segments: str) -> ArtifactBase:
    """Join *segments* onto *base* handling both local paths and URIs."""

    cleaned = _clean_segments(segments)
    if not cleaned:
        return base

    if isinstance(base, Path):
        return base.joinpath(*cleaned)

    base_str = str(base)

    if base_str.startswith("gdrive://"):
        prefix = base_str[len("gdrive://") :].rstrip("/")
        joined = "/".join(filter(None, [prefix, *cleaned]))
        return f"gdrive://{joined}"

    parsed = ParsedURI.parse(base_str)

    if parsed.scheme == "file" and parsed.local_path is not None:
        result_path = parsed.local_path.joinpath(*cleaned)
        return str(result_path)

    if parsed.scheme == "s3":
        key_parts = []
        if parsed.key:
            key_parts.append(parsed.key.rstrip("/"))
        key_parts.extend(cleaned)
        key = "/".join(part for part in key_parts if part)
        return f"s3://{parsed.bucket}/{key}" if parsed.bucket else f"s3:///{key}"

    if parsed.scheme == "mock":
        path = "/".join(filter(None, [(parsed.path or "").rstrip("/"), *cleaned]))
        return f"mock://{path}"

    if parsed.scheme in {"http", "https"}:
        base_http = base_str.rstrip("/")
        suffix = "/".join(cleaned)
        return f"{base_http}/{suffix}"

    result_path = Path(base_str).joinpath(*cleaned)
    return str(result_path)


def artifact_policy_run_root(
    base: ArtifactBase | None,
    *,
    run_name: str | None,
    epoch: int | None,
) -> ArtifactBase | None:
    """Return the replay root for a policy run under *base*.

    ``base`` can be a filesystem path or URI prefix. ``run_name`` is required for
    stable directory layouts; if it is falsy the base is returned verbatim. When
    ``epoch`` is truthy, a ``v{epoch}`` component is appended to produce versioned
    replay buckets.
    """

    if base is None or not run_name:
        return base

    run_root = artifact_path_join(base, run_name)
    if epoch:
        return artifact_path_join(run_root, f"v{epoch}")
    return run_root


__all__ = ["artifact_path_join", "artifact_policy_run_root"]
