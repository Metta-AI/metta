"""Helpers for syncing CoGames checkpoints to Softmax S3."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from metta.tools.utils.auto_config import auto_policy_storage_decision
from metta.utils.uri import ParsedURI

try:
    from rich.console import Console
except ImportError:  # pragma: no cover - fallback for type checkers
    Console = Any  # type: ignore

logger = logging.getLogger(__name__)

_REMOTE_FOLDER = "cogames"


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
    if decision.base_prefix is None or decision.reason not in {"env_override", "softmax_connected"}:
        return None

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
) -> bool:
    """Download the checkpoint from Softmax S3 if accessible and available.

    Returns True if the file is downloaded locally, otherwise False.
    """

    decision = auto_policy_storage_decision()
    if decision.base_prefix is None or decision.reason not in {"env_override", "softmax_connected"}:
        return False

    base_uri = decision.base_prefix.rstrip("/") + f"/{_REMOTE_FOLDER}"

    try:
        parsed = ParsedURI.parse(base_uri)
        bucket, key_prefix = parsed.require_s3()
    except ValueError:
        logger.debug("Skipping S3 download; %s is not a valid S3 URI", base_uri)
        return False

    try:
        import boto3

        s3_client = boto3.client("s3")
    except ImportError:
        logger.debug("boto3 not available; skipping S3 download")
        return False

    base_segments = _base_segments(game_name, policy_class_path, policy_path.stem)
    search_prefix_parts = [key_prefix.rstrip("/")] + base_segments
    search_prefix = "/".join(part for part in search_prefix_parts if part).rstrip("/") + "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    best_object: Optional[dict[str, Any]] = None

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=search_prefix):
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if not key or not key.endswith(policy_path.name):
                    continue
                if best_object is None:
                    best_object = obj
                else:
                    if obj.get("LastModified") and best_object.get("LastModified"):
                        if obj["LastModified"] > best_object["LastModified"]:
                            best_object = obj
                    elif key > best_object.get("Key", ""):
                        best_object = obj
    except Exception:  # pragma: no cover - avoid hard failures on listing issues
        logger.exception("Unable to list checkpoints under prefix %s", search_prefix)
        return False

    if not best_object or "Key" not in best_object:
        logger.debug(
            "No remote checkpoint matching %s found under s3://%s/%s",
            policy_path.name,
            bucket,
            search_prefix,
        )
        return False

    remote_key = best_object["Key"]
    policy_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        console.print(f"Downloading policy checkpoint from S3 to {policy_path}...", style="yellow")
        s3_client.download_file(bucket, remote_key, str(policy_path))
    except Exception:
        logger.exception("Failed to download checkpoint from s3://%s/%s", bucket, remote_key)
        return False

    console.print("Download complete.", style="green")
    return True


__all__ = ["maybe_upload_checkpoint", "maybe_download_checkpoint"]
