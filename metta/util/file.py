"""
file.py
================
Read and write files to local, S3, or W&B.
Use EFS on AWS for shared filesystems.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import shutil
import socket
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import boto3
import wandb
from botocore.exceptions import NoCredentialsError

# --------------------------------------------------------------------------- #
#  Globals                                                                     #
# --------------------------------------------------------------------------- #

WANDB_ENTITY: str = os.getenv("WANDB_ENTITY", "metta-research")


# --------------------------------------------------------------------------- #
#  Public IO helpers                                                           #
# --------------------------------------------------------------------------- #


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
            logger.info("Wrote %d B → %s", len(data), path)
            return
        except NoCredentialsError as e:
            logger.error("AWS credentials not found; run 'aws sso login --profile softmax'")
            raise e

    # ---------- W&B ---------- #
    if path.startswith("wandb://"):
        uri = WandbURI.parse(path)
        _upload_bytes_to_wandb(uri, data)
        logger.info("Wrote %d B → %s", len(data), uri)
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
        logger.info("Uploaded %s → %s", local_file, path)
        return

    # ---------- W&B ---------- #
    if path.startswith("wandb://"):
        uri = WandbURI.parse(path)
        _upload_file_to_wandb(uri, local_file)
        logger.info("Uploaded %s → %s", local_file, uri)
        return

    # ---------- local -------- #
    dst = Path(path).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_file, dst)
    logger.info("Copied %s → %s", local_file, dst)


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
            data = files[0].read_bytes()
            logger.info("Read %d B from %s", len(data), uri)
            return data

    # ---------- local -------- #
    data = Path(path).expanduser().resolve().read_bytes()
    logger.info("Read %d B from %s", len(data), path)
    return data


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


def _upload_bytes_to_wandb(uri: WandbURI, blob: bytes) -> None:
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(blob)
        tmpname = fh.name
    try:
        _upload_file_to_wandb(uri, tmpname)
    finally:
        os.unlink(tmpname)


def _upload_file_to_wandb(uri: WandbURI, local_file: str) -> None:
    logger = logging.getLogger(__name__)
    art = wandb.Artifact(uri.artifact_path, type="file", incremental=True)
    art.add_file(local_file)
    art.save(entity=WANDB_ENTITY, project=uri.project)  # no run created
    logger.debug("Saved W&B artifact %s", uri.qname())


# --------------------------------------------------------------------------- #
#  Distributed lock for shared filesystems (unchanged)                         #
# --------------------------------------------------------------------------- #


class EfsLock:
    """Simple stale-file lock suitable for EFS / NFS-style shares."""

    def __init__(self, path: str, timeout: int = 300, retry_interval: int = 5, max_retries: int = 60):
        self.path = path
        self.timeout = timeout
        self.retry_interval = retry_interval
        self.max_retries = max_retries
        self.lock_acquired = False
        self.hostname = socket.gethostname()
        self.system = platform.system()

    # context-manager sugar
    def __enter__(self):  # noqa: D401
        self._acquire_lock()
        return self

    def __exit__(self, *_exc):
        self._release_lock()

    # ---------- internals ---------- #
    def _acquire_lock(self):
        retries = 0
        time.sleep(random.uniform(0.1, 1.0))  # soften thundering herd
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        while retries < self.max_retries:
            try:
                with open(self.path, "x") as f:
                    f.write(f"{time.time()}\n{os.getpid()}\n{self.hostname}\n{self.system}\n")
                self.lock_acquired = True
                logger.info("Lock acquired: %s", self.path)
                return
            except FileExistsError:
                if self._is_stale():
                    try:
                        os.remove(self.path)
                        continue  # retry immediately after removing stale lock
                    except OSError as e:
                        logger.warning("Could not remove stale lock: %s", e)
                time.sleep(self.retry_interval)
                retries += 1
        raise TimeoutError(f"Failed to acquire lock after {retries} retries: {self.path}")

    def _is_stale(self) -> bool:
        try:
            with open(self.path) as f:
                timestamp = float(f.readline().strip())
            return (time.time() - timestamp) > self.timeout
        except Exception:
            return True  # unreadable → treat as stale

    def _release_lock(self):
        if self.lock_acquired:
            try:
                os.remove(self.path)
                logger.info("Lock released: %s", self.path)
            except OSError as e:
                logger.warning("Failed to release lock %s: %s", self.path, e)
            finally:
                self.lock_acquired = False


@contextmanager
def efs_lock(path: str, timeout: int = 300, retry_interval: int = 5, max_retries: int = 60):
    """
    Convenience wrapper: ``with efs_lock("/shared/my.lock"):``
    """
    # shorten defaults for macOS dev (NFS-loopback is fast)
    if platform.system() == "Darwin":
        timeout = min(timeout, 60)
        retry_interval = min(retry_interval, 2)

    lock = EfsLock(path, timeout, retry_interval, max_retries)
    try:
        lock._acquire_lock()
        yield
    finally:
        lock._release_lock()
