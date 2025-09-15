"""
file.py
================
Read and write files to local, S3, or W&B.
Use EFS on AWS for shared filesystems.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union
from urllib.parse import urlparse

import boto3
import wandb
from botocore.exceptions import ClientError, NoCredentialsError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build as gdrive_build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from wandb.errors import CommError

# --------------------------------------------------------------------------- #
#  Globals                                                                     #
# --------------------------------------------------------------------------- #

def _get_wandb_entity() -> str:
    """Get WANDB_ENTITY, checking environment variable dynamically."""
    return os.getenv("WANDB_ENTITY") or ""

# Keep for backward compatibility, but use the function internally
WANDB_ENTITY: str = _get_wandb_entity()
GOOGLE_DRIVE_CREDENTIALS_FILE: str = os.getenv("GOOGLE_DRIVE_CREDENTIALS_FILE", "~/.config/gcloud/credentials.json")
GOOGLE_DRIVE_TOKEN_FILE: str = os.getenv("GOOGLE_DRIVE_TOKEN_FILE", "~/.config/gcloud/token.json")

# --------------------------------------------------------------------------- #
#  Public IO helpers                                                           #
# --------------------------------------------------------------------------- #


def exists(path: str) -> bool:
    """
    Return *True* if *path* points to an existing local file, S3 object,
    W&B artifact **version** (latest if omitted), or Google Drive file or folder.
    Network errors are propagated so callers can decide how to handle them.
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

    # ---------- Google Drive ---------- #
    if path.startswith("gdrive://") or path.startswith("https://drive.google.com/"):
        return _gdrive_exists(path)

    # ---------- local -------- #
    return Path(path).expanduser().exists()


def write_data(path: str, data: Union[str, bytes], *, content_type: str = "application/octet-stream") -> None:
    """
    Write in-memory bytes/str to *local*, *s3://*, *wandb://*, or *gdrive://* destinations.
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

    # ---------- Google Drive ---------- #
    if path.startswith("gdrive://") or path.startswith("https://drive.google.com/"):
        file_id = _gdrive_write_data(path, data, content_type)
        logger.info("Wrote %d B → %s (ID: %s)", len(data), http_url(path), file_id)
        return

    # ---------- local -------- #
    local_path = Path(path).expanduser().resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    logger.info("Wrote %d B → %s", len(data), local_path)


def write_file(path: str, local_file: str, *, content_type: str = "application/octet-stream") -> None:
    """
    Upload a file from disk to *s3://, *wandb://, *gdrive://* (or copy locally).
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

    # ---------- Google Drive ---------- #
    if path.startswith("gdrive://") or path.startswith("https://drive.google.com/"):
        file_id = _gdrive_write_file(path, local_file, content_type)
        logger.info(
            "Uploaded %s → %s (ID: %s, size %d B)", local_file, http_url(path), file_id, os.path.getsize(local_file)
        )
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
    if path.startswith("gdrive://") or path.startswith("https://drive.google.com/"):
        return GDriveURI.parse(path).http_url()
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
    """Parsed representation of a W&B artifact URI.

    Supports both formats:
    - wandb://project/artifact:version (uses environment WANDB_ENTITY)
    - wandb://entity/project/artifact:version (entity from URI)
    """

    project: str
    artifact_path: str
    version: str = "latest"
    entity: Optional[str] = None

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
            raise ValueError("Malformed W&B URI – expected wandb://<project>/<artifact_path> or wandb://<entity>/<project>/<artifact_path>")

        parts = path_part.split("/")
        if len(parts) == 2:
            # Format: wandb://project/artifact:version
            project, artifact_path = parts
            entity = None
        elif len(parts) >= 3:
            # Format: wandb://entity/project/artifact/...:version
            entity = parts[0]
            project = parts[1]
            artifact_path = "/".join(parts[2:])
        else:
            raise ValueError("Malformed W&B URI – expected wandb://<project>/<artifact_path> or wandb://<entity>/<project>/<artifact_path>")

        if not artifact_path:
            raise ValueError("Artifact path must be non-empty")

        return cls(project, artifact_path, version, entity)

    # ---------- helpers ---------- #
    def qname(self) -> str:
        """`entity/project/artifact_path:version` – accepted by `wandb.Api().artifact()`."""
        # Use entity from URI if provided, otherwise fall back to environment variable
        if self.entity:
            entity = self.entity
        else:
            entity = _get_wandb_entity()  # Use dynamic lookup
            if not entity:
                raise ValueError(
                    f"No wandb entity specified. Either:\n"
                    f"1. Use full URI format: wandb://your-entity/{self.project}/{self.artifact_path}:{self.version}\n"
                    f"2. Set WANDB_ENTITY environment variable: export WANDB_ENTITY=your-entity"
                )
        return f"{entity}/{self.project}/{self.artifact_path}:{self.version}"

    def http_url(self) -> str:
        """Human-readable URL for this artifact version."""
        entity = self.entity or _get_wandb_entity()  # Use dynamic lookup
        if not entity:
            raise ValueError("Cannot create HTTP URL without wandb entity")
        return f"https://wandb.ai/{entity}/{self.project}/artifacts/{self.artifact_path}/{self.version}"

    # pretty print
    def __str__(self) -> str:  # noqa: D401 (keep dunder)
        if self.entity:
            return f"wandb://{self.entity}/{self.project}/{self.artifact_path}:{self.version}"
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
def wandb_export_context(project: str, entity: str | None = None) -> wandb.Run:
    """
    Context manager that ensures a wandb run exists for artifact exports.
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
            entity=entity or _get_wandb_entity(),
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
        entity = _get_wandb_entity()
        with wandb_export_context(uri.project, entity) as run:
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


# --------------------------------------------------------------------------- #
#  GDrive URI handling                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GDriveURI:
    kind: Literal["file", "folder"]
    id: str  # file ID or folder ID
    name: Optional[str] = None  # required for folder uploads

    @classmethod
    def parse(cls, uri: str) -> "GDriveURI":
        # gdrive://file/<ID>
        if uri.startswith("gdrive://file/"):
            fid = uri.split("gdrive://file/", 1)[1].strip("/")
            if not fid:
                raise ValueError("Malformed: gdrive://file/<FILE_ID>")
            return cls("file", fid)

        # gdrive://folder/<FOLDER_ID>/<FILENAME>
        if uri.startswith("gdrive://folder/"):
            rest = uri.split("gdrive://folder/", 1)[1].strip("/")
            folder_id, *name = rest.split("/", 1)
            if not name:
                raise ValueError("Malformed: gdrive://folder/<FOLDER_ID>/<FILENAME>")
            return cls("folder", folder_id, name[0])

        # Drive file URLs (treat as file-by-ID)
        for pat in (r"https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)", r"[?&]id=([A-Za-z0-9_-]+)"):
            if m := re.search(pat, uri):
                return cls("file", m.group(1))

        # Bare fallback: gdrive://<ID>
        if uri.startswith("gdrive://") and (fid := uri.split("gdrive://", 1)[1].strip("/")):
            return cls("file", fid)

        raise ValueError("Not a recognized Google Drive URI")

    def http_url(self) -> str:
        if self.kind == "file":
            return f"https://drive.google.com/file/d/{self.id}/view"
        else:
            return f"https://drive.google.com/drive/folders/{self.id}"


_GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _load_gdrive_credentials() -> Credentials:
    token = Path(GOOGLE_DRIVE_TOKEN_FILE).expanduser()
    creds = Credentials.from_authorized_user_file(str(token), _GDRIVE_SCOPES) if token.exists() else None

    if not (creds and creds.valid):
        cred = Path(GOOGLE_DRIVE_CREDENTIALS_FILE).expanduser()
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not cred.exists():
                raise FileNotFoundError(f"Missing OAuth client secrets at {cred}")
            creds = InstalledAppFlow.from_client_secrets_file(str(cred), _GDRIVE_SCOPES).run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json())

    return creds


def _gdrive_service():
    return gdrive_build("drive", "v3", credentials=_load_gdrive_credentials(), cache_discovery=False)


def _gdrive_exists(uri_str: str) -> bool:
    """True if Drive file (by ID) or file-in-folder (by name) exists; False on 403/404; else raise."""
    uri, svc = GDriveURI.parse(uri_str), _gdrive_service()
    try:
        if uri.kind == "file":
            svc.files().get(fileId=uri.id, fields="id", supportsAllDrives=True).execute()
            return True
        name = (uri.name or "").replace("'", "\\'")
        resp = (
            svc.files()
            .list(
                q=f"'{uri.id}' in parents and name = '{name}' and trashed = false",
                fields="files(id)",
                pageSize=1,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        return bool(resp.get("files"))
    except HttpError as e:
        if getattr(getattr(e, "resp", None), "status", None) in (403, 404):
            return False
        raise


def _gdrive_write_file(uri_str: str, local_file: str, content_type: str) -> str:
    """Upload a local file to Google Drive and return the file-ID.

    • gdrive://file/<ID>         → update that file
    • gdrive://folder/<ID>/<fn>  → create (or overwrite-by-name) inside folder
    """
    uri = GDriveURI.parse(uri_str)
    svc = _gdrive_service()

    # Read the local file and create media upload
    media = MediaFileUpload(local_file, mimetype=content_type, resumable=False)

    try:
        if uri.kind == "file":  # update by ID
            return (
                svc.files().update(fileId=uri.id, media_body=media, supportsAllDrives=True, fields="id").execute()["id"]
            )

        # ───── kind == "folder" ─────
        meta = {"name": uri.name, "parents": [uri.id]}
        fileId = svc.files().create(body=meta, media_body=media, supportsAllDrives=True, fields="id").execute()["id"]

        # best-effort public permission; ignore failures
        try:
            svc.permissions().create(
                fileId=fileId,
                body={"type": "anyone", "role": "reader"},
                supportsAllDrives=True,
            ).execute()
        except HttpError:
            pass

        return fileId

    except HttpError as e:
        if getattr(e, "resp", None) and e.resp.status == 404:
            raise FileNotFoundError(f"Google Drive {uri.kind} not found: {uri.id}") from e
        raise


def _gdrive_write_data(uri_str: str, data: bytes, content_type: str) -> str:
    """Upload *data* to Google Drive and return the file-ID.

    • gdrive://file/<ID>         → update that file
    • gdrive://folder/<ID>/<fn>  → create (or overwrite-by-name) inside folder
    """
    uri = GDriveURI.parse(uri_str)
    svc = _gdrive_service()
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=content_type, resumable=False)

    try:
        if uri.kind == "file":  # update by ID
            return (
                svc.files().update(fileId=uri.id, media_body=media, supportsAllDrives=True, fields="id").execute()["id"]
            )

        # ───── kind == "folder" ─────
        meta = {"name": uri.name, "parents": [uri.id]}
        fileId = svc.files().create(body=meta, media_body=media, supportsAllDrives=True, fields="id").execute()["id"]

        # best-effort public permission; ignore failures
        try:
            svc.permissions().create(
                fileId=fileId,
                body={"type": "anyone", "role": "reader"},
                supportsAllDrives=True,
            ).execute()
        except HttpError:
            pass

        return fileId

    except HttpError as e:
        if getattr(e, "resp", None) and e.resp.status == 404:
            raise FileNotFoundError(f"Google Drive {uri.kind} not found: {uri.id}") from e
        raise
