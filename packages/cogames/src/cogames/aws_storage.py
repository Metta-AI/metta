"""Helpers for syncing CoGames checkpoints to Softmax S3."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from metta.tools.utils.auto_config import auto_policy_storage_decision
from metta.utils.uri import ParsedURI

try:
    from rich.console import Console
except ImportError:  # pragma: no cover - fallback for type checkers
    Console = Any  # type: ignore

logger = logging.getLogger(__name__)

_REMOTE_FOLDER = "cogames"


@dataclass(frozen=True)
class DownloadOutcome:
    downloaded: bool
    reason: Literal[
        "downloaded",
        "aws_not_enabled",
        "no_base_prefix",
        "invalid_prefix",
        "boto3_missing",
        "not_found",
        "error",
    ]
    details: Optional[str] = None


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "-", text)
    collapsed = re.sub(r"-+", "-", cleaned).strip("-._")
    return collapsed.lower() or "run"


def _policy_segment(policy_class_path: str) -> str:
    return policy_class_path.rsplit(".", 1)[-1]


def _base_segments(
    game_name: Optional[str],
    policy_class_path: str,
    checkpoint_stem: str,
) -> list[str]:
    segments: list[str] = []
    if game_name:
        segments.append(_slugify(game_name))
    segments.append(_slugify(_policy_segment(policy_class_path)))
    segments.append(_slugify(checkpoint_stem))
    return segments


def _prefix_exists(client: Any, bucket: str, prefix: str) -> bool:
    normalized = prefix.rstrip("/") + "/"
    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=normalized, MaxKeys=1)
    except Exception:  # pragma: no cover - avoid failing upload on AWS issues
        logger.exception("Unable to list objects for prefix %s", normalized)
        return False
    return response.get("KeyCount", 0) > 0


def maybe_upload_checkpoint(
    *,
    final_checkpoint: Path,
    game_name: Optional[str],
    policy_class_path: str,
    console: Console,
) -> Optional[str]:
    """Upload the checkpoint to Softmax S3 if AWS is configured.

    Returns the remote URI when an upload occurs, otherwise None.
    """

    decision = auto_policy_storage_decision()
    if decision.base_prefix is None:
        logger.debug("Remote policy storage disabled; reason=%s", getattr(decision, "reason", "unknown"))
        return None
    if decision.reason in {"aws_not_enabled", "no_base_prefix"}:
        logger.debug("AWS policy storage not configured (reason=%s)", decision.reason)
        return None
    if decision.reason == "not_connected":
        console.print("[yellow]Softmax AWS login not detected; attempting to upload checkpoint anyway.[/yellow]")

    base_uri = decision.base_prefix.rstrip("/") + f"/{_REMOTE_FOLDER}"

    try:
        parsed = ParsedURI.parse(base_uri)
        bucket, key_prefix = parsed.require_s3()
    except ValueError:
        logger.debug("Skipping S3 upload; %s is not a valid S3 URI", base_uri)
        return None

    try:
        import boto3

        s3_client = boto3.client("s3")
    except ImportError:
        logger.debug("boto3 not available; skipping S3 upload")
        return None

    base_segments = _base_segments(game_name, policy_class_path, final_checkpoint.stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    key_parts = [key_prefix.rstrip("/")] + base_segments + [timestamp]
    candidate_prefix = "/".join(part for part in key_parts if part)

    suffix = 1
    unique_prefix = candidate_prefix
    while _prefix_exists(s3_client, bucket, unique_prefix):
        suffix += 1
        unique_prefix = "/".join(part for part in key_parts[:-1] if part)
        unique_prefix = f"{unique_prefix}/{timestamp}-{suffix}"
        if suffix > 20:
            logger.warning("Unable to determine a unique S3 prefix for %s", candidate_prefix)
            return None

    key = f"{unique_prefix.rstrip('/')}/{final_checkpoint.name}"

    try:
        s3_client.upload_file(str(final_checkpoint), bucket, key)
    except Exception:
        logger.exception("Failed to upload checkpoint %s to s3://%s/%s", final_checkpoint, bucket, key)
        return None

    remote_uri = f"s3://{bucket}/{key}"
    console.print()
    console.print("Uploaded checkpoint to S3:", style="bold")
    console.print(f"  [yellow]{remote_uri}[/yellow]")
    return remote_uri


def maybe_download_checkpoint(
    *,
    policy_path: Path,
    game_name: Optional[str],
    policy_class_path: str,
    console: Console,
) -> DownloadOutcome:
    """Download the checkpoint from Softmax S3 if accessible and available.

    Returns True if the file is downloaded locally, otherwise False.
    """

    decision = auto_policy_storage_decision()
    if decision.base_prefix is None:
        logger.debug("Remote policy storage disabled; reason=%s", getattr(decision, "reason", "unknown"))
        return DownloadOutcome(downloaded=False, reason="no_base_prefix")
    if decision.reason == "aws_not_enabled":
        console.print("[yellow]AWS policy storage not configured; skipping remote download.[/yellow]")
        return DownloadOutcome(downloaded=False, reason="aws_not_enabled")
    if decision.reason == "no_base_prefix":
        logger.debug("AWS policy storage missing remote prefix configuration")
        return DownloadOutcome(downloaded=False, reason="no_base_prefix")
    if decision.reason == "not_connected":
        console.print("[yellow]Softmax AWS login not detected; attempting to fetch checkpoint anyway.[/yellow]")

    base_uri = decision.base_prefix.rstrip("/") + f"/{_REMOTE_FOLDER}"

    try:
        parsed = ParsedURI.parse(base_uri)
        bucket, key_prefix = parsed.require_s3()
    except ValueError:
        logger.debug("Skipping S3 download; %s is not a valid S3 URI", base_uri)
        return DownloadOutcome(downloaded=False, reason="invalid_prefix")

    try:
        import boto3

        s3_client = boto3.client("s3")
    except ImportError:
        console.print("[yellow]boto3 not installed; cannot download policy from S3.[/yellow]")
        logger.debug("boto3 not available; skipping S3 download")
        return DownloadOutcome(downloaded=False, reason="boto3_missing")

    base_segments = _base_segments(game_name, policy_class_path, policy_path.stem)
    search_prefix_parts = [key_prefix.rstrip("/")] + base_segments
    search_prefix = "/".join(part for part in search_prefix_parts if part).rstrip("/") + "/"

    paginator = s3_client.get_paginator("list_objects_v2")

    def _select_best(prefix: str) -> Optional[dict[str, Any]]:
        best: Optional[dict[str, Any]] = None
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if not key or not key.endswith(policy_path.name):
                        continue
                    if best is None:
                        best = obj
                    else:
                        if obj.get("LastModified") and best.get("LastModified"):
                            if obj["LastModified"] > best["LastModified"]:
                                best = obj
                        elif key > best.get("Key", ""):
                            best = obj
        except Exception:  # pragma: no cover - avoid hard failures on listing issues
            logger.exception("Unable to list checkpoints under prefix %s", prefix)
            return None
        return best

    best_object = _select_best(search_prefix)
    if best_object is None:
        fallback_prefix = key_prefix.rstrip("/") + "/"
        best_object = _select_best(fallback_prefix)

    if not best_object or "Key" not in best_object:
        console.print(
            f"[yellow]No remote checkpoint named {policy_path.name} found under s3://{bucket}/{search_prefix}[/yellow]"
        )
        return DownloadOutcome(downloaded=False, reason="not_found", details=search_prefix)

    remote_key = best_object["Key"]
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        console.print(
            f"Downloading policy checkpoint from s3://{bucket}/{remote_key} to {policy_path}...",
            style="yellow",
        )
        s3_client.download_file(bucket, remote_key, str(policy_path))
    except Exception as err:
        logger.exception("Failed to download checkpoint from s3://%s/%s", bucket, remote_key)
        console.print(f"[red]Failed to download checkpoint from s3://{bucket}/{remote_key}: {err}[/red]")
        return DownloadOutcome(downloaded=False, reason="error", details=str(err))

    console.print("Download complete.", style="green")
    return DownloadOutcome(
        downloaded=True,
        reason="downloaded",
        details=f"s3://{bucket}/{remote_key}",
    )


__all__ = ["DownloadOutcome", "maybe_upload_checkpoint", "maybe_download_checkpoint"]
