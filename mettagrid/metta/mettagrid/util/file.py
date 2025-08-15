"""
file.py
================
Read and write files to local, S3, or W&B.
Use EFS on AWS for shared filesystems.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import boto3
import wandb
from botocore.exceptions import ClientError, NoCredentialsError
from wandb.errors import CommError

# --------------------------------------------------------------------------- #
#  Globals                                                                     #
# --------------------------------------------------------------------------- #

WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "metta-research")

# --------------------------------------------------------------------------- #
#  Public IO helpers                                                           #
# --------------------------------------------------------------------------- #


def exists(path: str) -> bool:
    """
    Return *True* if *path* points to an existing local file, S3 object,
    or W&B artifact **version** (latest if omitted).  Network errors are
    propagated so callers can decide how to handle them.
    """
    # ---------- S3 ---------- #
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in {"404", "403", "NoSuchKey"}:
                return False
            raise

    # ---------- W&B ---------- #
    if path.startswith("wandb://"):
        uri = WandbURI.parse(path)
        api = wandb.Api()
        try:
            api.artifact(uri.qname())
            return True
        except CommError:
            return False

    # ---------- local -------- #
    return Path(path).expanduser().exists()


def write_data(path: str, data: Union[str, bytes], *, content_type: str = "application/octet-stream") -> None:
    """
    Write in-memory bytes/str to *local*, *s3://* or *wandb://* destinations.
    """
    logger = logging.getLogger(__name__)

    if isinstance(data, str):
        data = data.encode()

    # ---------- S3 ---------- #
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        try:
            boto3.client("s3").put_object(Body=data, Bucket=bucket, Key=key, ContentType=content_type)
            logger.info("Wrote %d B → %s", len(data), http_url(path))
            return
        except NoCredentialsError as e:
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'")
            raise e

    # ---------- W&B ---------- #
    if path.startswith("wandb://"):
        uri = WandbURI.parse(path)
        upload_bytes_to_wandb(uri, data, name=uri.artifact_path.split("/")[-1])
        logger.info("Wrote %d B → %s", len(data), uri.http_url())
        return

    # ---------- local -------- #
    local_path = Path(path).expanduser().resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    logger.info("Wrote %d B → %s", len(data), local_path)


def write_file(path: str, local_file: str, *, content_type: str = "application/octet-stream") -> None:
    """
    Upload a file from disk to *s3://* or *wandb://* (or copy locally).
    """
    logger = logging.getLogger(__name__)

    # ---------- S3 ---------- #
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        boto3.client("s3").upload_file(local_file, bucket, key, ExtraArgs={"ContentType": content_type})
        logger.info("Uploaded %s → %s (size %d B)", local_file, path, os.path.getsize(local_file))
        return

    # ---------- W&B ---------- #
    if path.startswith("wandb://"):
        uri = WandbURI.parse(path)
        upload_file_to_wandb(uri, local_file, name=uri.artifact_path)
        logger.info("Uploaded %s → %s (size %d B)", local_file, uri, os.path.getsize(local_file))
        return

    # ---------- local -------- #
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_file, dst)
    logger.info("Copied %s → %s (size %d B)", local_file, dst, os.path.getsize(local_file))


def read(path: str) -> bytes:
    """
    Read bytes from local path, S3 object, or W&B artifact.
    """
    logger = logging.getLogger(__name__)

    # ---------- S3 ---------- #
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        try:
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            logger.info("Read %d B from %s", len(body), path)
            return body
        except NoCredentialsError:
            logger.error("AWS credentials not found -- have you run devops/aws/setup_sso.py?")
            raise

    # ---------- W&B ---------- #
    if path.startswith("wandb://"):
        uri = WandbURI.parse(path)
        api = wandb.Api()
        artifact = api.artifact(uri.qname())
        with tempfile.TemporaryDirectory(prefix="wandb_dl_") as tmp:
            local_dir = artifact.download(root=tmp)
            files = list(Path(local_dir).iterdir())
            if not files:
                raise FileNotFoundError(f"No files inside W&B artifact {uri}")
            if len(files) > 1:
                raise ValueError(f"Expected exactly one file inside W&B artifact {uri}, got {len(files)}: {files}")
            data = files[0].read_bytes()
            logger.info("Read %d B from %s", len(data), uri)
            return data

    # ---------- local -------- #
    data = Path(path).expanduser().resolve().read_bytes()
    logger.info("Read %d B from %s", len(data), path)
    return data


@contextmanager
def local_copy(path: str):
    """
    Yield a local *Path* for *path* (supports local / s3:// / wandb://).

    • Local paths are yielded as-is.
    • Remote URIs are streamed into a NamedTemporaryFile that is removed
      when the context exits, so callers never worry about cleanup.

    Usage:
        with local_copy(uri) as p:
            do_something_with(Path(p))
    """
    if path.startswith(("s3://", "wandb://")):
        data = read(path)  # existing helper
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".duckdb")
        tmp.write(data)
        tmp.flush()
        tmp.close()
        try:
            yield Path(tmp.name)
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
    else:
        yield Path(path).expanduser().resolve()


def http_url(path: str) -> str:
    """
    Convert *s3://* or *wandb://* URI to a public browser URL.
    No-op for local paths.
    """
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    if path.startswith("wandb://"):
        return WandbURI.parse(path).http_url()
    return path


def is_public_uri(url: str | None) -> bool:
    """
    Check if a URL is a public HTTP/HTTPS URL.
    """
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# --------------------------------------------------------------------------- #
#  W&B URI handling                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WandbURI:
    """Parsed representation of a W&B artifact URI."""

    project: str
    artifact_path: str
    version: str = "latest"

    # ---------- factory ---------- #
    @classmethod
    def parse(cls, uri: str) -> "WandbURI":
        if not uri.startswith("wandb://"):
            raise ValueError("W&B URI must start with wandb://")

        body = uri[len("wandb://") :]
        if ":" in body:
            path_part, version = body.rsplit(":", 1)
        else:
            path_part, version = body, "latest"

        if "/" not in path_part:
            raise ValueError("Malformed W&B URI – expected wandb://<project>/<artifact_path>[:<version>]")

        project, artifact_path = path_part.split("/", 1)
        if not artifact_path:
            raise ValueError("Artifact path must be non-empty")

        return cls(project, artifact_path, version)

    # ---------- helpers ---------- #
    def qname(self) -> str:
        """`entity/project/artifact_path:version` – accepted by `wandb.Api().artifact()`."""
        return f"{WANDB_ENTITY}/{self.project}/{self.artifact_path}:{self.version}"

    def http_url(self) -> str:
        """Human-readable URL for this artifact version."""
        return f"https://wandb.ai/{WANDB_ENTITY}/{self.project}/artifacts/{self.artifact_path}/{self.version}"

    # pretty print
    def __str__(self) -> str:  # noqa: D401 (keep dunder)
        return f"wandb://{self.project}/{self.artifact_path}:{self.version}"


def upload_bytes_to_wandb(uri: WandbURI, blob: bytes, name: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(blob)
        tmpname = fh.name
    try:
        upload_file_to_wandb(uri, tmpname, name=name)
    finally:
        os.unlink(tmpname)


@contextmanager
def wandb_export_context(project: str, entity: str = WANDB_ENTITY) -> wandb.Run:
    """
    Context manager that ensures a wandb run exists for artifact exports.
    TODO: Refactor to use WandbContext without requiring passing a deep hydra config
    TODO: Remove this after switching to using wandb_context

    Args:
        project: wandb project name
        entity: wandb entity name

    Yields:
        The active wandb run object
    """
    # Check if there's already an active run
    active_run = wandb.run

    if active_run is not None:
        if active_run.project != project:
            raise ValueError(
                f"Wandb run already active: {active_run.name}; "
                "can't post files to a different project inside the same process"
            )
        run = active_run
    else:
        # Create a temporary run
        run = wandb.init(
            project=project,
            entity=entity,
            job_type="file-export",
            name="file-export",
            settings=wandb.Settings(
                disable_code=True,
                console="off",
                quiet=True,
            ),
        )

    try:
        yield run
    finally:
        # TODO: We don't want to finish the run becasue we want to be able to
        # reopen it later. However it would be good to "unset" wandb.run if we could.
        # I'm not sure the right way to do that
        pass


def upload_file_to_wandb(uri, local_file: str, name: str) -> None:
    """
    Upload *local_file* to W&B as the next version of
        wandb://{uri.project}/{uri.artifact_path}:latest    (type="file")
    • Re-uses the caller's active run if present.
    • Otherwise creates a temporary run just for this upload and finishes it

    Args:
        uri: A WandbURI object containing project, artifact_path, and version
        local_file: Path to the file to upload

    Returns:
        None

    Raises:
        ValueError: If uri.version is not "latest"
    """
    logger = logging.getLogger(__name__)

    if uri.version != "latest":
        raise ValueError(
            "Export only supports the implicit ':latest' alias when writing to W&B (explicit versions are immutable)."
        )

    try:
        with wandb_export_context(uri.project, WANDB_ENTITY) as run:
            # Create and log the artifact
            artifact = wandb.Artifact(uri.artifact_path, type="file")
            artifact.add_file(local_file, name=name)
            run.log_artifact(artifact)

            logger.info(
                "Uploaded %s → wandb://%s/%s  (size %d B, new version)",
                local_file,
                uri.project,
                uri.artifact_path,
                os.path.getsize(local_file),
            )
    except Exception as e:
        logger.error(f"Failed to upload file to wandb: {e}")
        raise
