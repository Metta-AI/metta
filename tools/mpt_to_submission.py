from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from rich.console import Console

from cogames.cli.client import TournamentServerClient
from cogames.cli.login import DEFAULT_COGAMES_SERVER
from cogames.cli.submit import DEFAULT_SUBMIT_SERVER, upload_submission


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert legacy .mpt to submission zip (optionally upload to S3/Observatory)."
    )
    parser.add_argument("--mpt", required=True, help="Path or s3:// URI to legacy .mpt file")
    parser.add_argument("--name", help="Submission policy name")
    parser.add_argument("--season", help="Tournament season to submit to (e.g., beta)")
    parser.add_argument(
        "--output",
        help="Output zip path (defaults to /tmp/<mpt_basename>.zip)",
    )
    parser.add_argument(
        "--s3-dest",
        help="Upload bundle to S3 at s3://bucket/prefix[/] or s3://bucket/key.zip",
    )
    parser.add_argument(
        "--replace-s3",
        action="store_true",
        help="Upload bundle next to source .mpt with same key and .zip suffix (requires --mpt s3://...)",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source .mpt after successful S3 upload (requires --mpt s3://...)",
    )
    parser.add_argument(
        "--no-copy-extras",
        action="store_true",
        help="Exclude extra files from the legacy .mpt (e.g., agent_codebase.yaml)",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SUBMIT_SERVER,
        help="Observatory API base URL",
    )
    parser.add_argument(
        "--login-server",
        default=DEFAULT_COGAMES_SERVER,
        help="Login/authentication server URL",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip Observatory upload/submit (S3 upload still allowed)",
    )
    return parser.parse_args()


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3 URI: {uri}")
    remainder = uri[5:]
    if "/" not in remainder:
        raise ValueError(f"Malformed s3 URI: {uri}")
    bucket, key = remainder.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"Malformed s3 URI: {uri}")
    return bucket, key


def _download_mpt(mpt: str, dest_dir: Path) -> Path:
    if mpt.startswith("s3://"):
        bucket, key = _parse_s3_uri(mpt)
        dest = dest_dir / Path(key).name
        s3 = boto3.client("s3")
        try:
            s3.download_file(bucket, key, str(dest))
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"Failed to download {mpt}: {exc}") from exc
        return dest
    local_path = Path(mpt).expanduser().resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Missing mpt file: {local_path}")
    return local_path


def _validate_entry_name(name: str) -> None:
    path = Path(name)
    if path.is_absolute():
        raise ValueError(f"Archive contains absolute path: {name}")
    if ".." in path.parts:
        raise ValueError(f"Archive contains path traversal: {name}")


def _build_submission_zip(mpt_path: Path, output_zip: Path, *, copy_extras: bool) -> Path:
    with zipfile.ZipFile(mpt_path, "r") as archive:
        names = set(archive.namelist())
        if "weights.safetensors" not in names or "modelarchitecture.txt" not in names:
            raise ValueError("Expected weights.safetensors and modelarchitecture.txt in legacy .mpt archive")
        architecture_spec = archive.read("modelarchitecture.txt").decode("utf-8")
        extra_names = []
        if copy_extras:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                _validate_entry_name(entry.filename)
                if entry.filename in {"weights.safetensors", "modelarchitecture.txt"}:
                    continue
                extra_names.append(entry.filename)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            archive.extract("weights.safetensors", tmp_path)
            if copy_extras:
                for name in extra_names:
                    archive.extract(name, tmp_path)

            policy_spec = {
                "class_path": "metta.agent.policy.CheckpointPolicy",
                "data_path": "weights.safetensors",
                "init_kwargs": {
                    "architecture_spec": architecture_spec,
                    "device": "cpu",
                },
            }
            (tmp_path / "policy_spec.json").write_text(json.dumps(policy_spec))
            (tmp_path / "modelarchitecture.txt").write_text(architecture_spec)

            output_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as out_zip:
                for name in ("policy_spec.json", "weights.safetensors", "modelarchitecture.txt"):
                    out_zip.write(tmp_path / name, arcname=name)
                if copy_extras:
                    for name in extra_names:
                        out_zip.write(tmp_path / name, arcname=name)

    return output_zip


def _default_output_path(mpt_path: Path) -> Path:
    return Path("/tmp") / f"{mpt_path.stem}.zip"


def _resolve_s3_destination(
    *,
    mpt_arg: str,
    output_zip: Path,
    s3_dest: Optional[str],
    replace_s3: bool,
) -> Optional[tuple[str, str]]:
    if s3_dest and replace_s3:
        raise ValueError("Use either --s3-dest or --replace-s3, not both.")
    if replace_s3:
        if not mpt_arg.startswith("s3://"):
            raise ValueError("--replace-s3 requires --mpt to be an s3:// URI.")
        bucket, key = _parse_s3_uri(mpt_arg)
        zip_key = f"{key[:-4]}.zip" if key.endswith(".mpt") else f"{key}.zip"
        return bucket, zip_key
    if s3_dest:
        bucket, key = _parse_s3_uri(s3_dest)
        if key.endswith(".zip"):
            return bucket, key
        if not key.endswith("/"):
            key = f"{key}/"
        return bucket, f"{key}{output_zip.name}"
    return None


def _upload_zip_to_s3(zip_path: Path, bucket: str, key: str) -> None:
    s3 = boto3.client("s3")
    try:
        s3.upload_file(
            str(zip_path),
            bucket,
            key,
            ExtraArgs={"ContentType": "application/zip"},
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload {zip_path} to s3://{bucket}/{key}: {exc}") from exc


def _delete_s3_object(bucket: str, key: str) -> None:
    s3 = boto3.client("s3")
    try:
        s3.delete_object(Bucket=bucket, Key=key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to delete s3://{bucket}/{key}: {exc}") from exc


def _upload_and_submit(
    zip_path: Path,
    name: str,
    season: Optional[str],
    server: str,
    login_server: str,
    console: Console,
) -> None:
    client = TournamentServerClient.from_login(server_url=server, login_server=login_server)
    if not client:
        raise SystemExit(1)
    with client:
        result = upload_submission(client, zip_path, name, console)
        if not result:
            raise SystemExit("Upload failed")
        console.print(f"Uploaded as {result.name}:v{result.version} id={result.policy_version_id}")
        if season:
            submit_resp = client.submit_to_season(season, result.policy_version_id)
            console.print(f"Submitted to season {season}; pools={submit_resp.pools}")


def main() -> None:
    args = _parse_args()
    console = Console()

    if not args.no_upload and not args.name:
        raise ValueError("--name is required unless --no-upload is set.")
    if args.season and args.no_upload:
        raise ValueError("--season requires Observatory upload; remove --no-upload.")
    if args.delete_source and not args.mpt.startswith("s3://"):
        raise ValueError("--delete-source requires --mpt to be an s3:// URI.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mpt_path = _download_mpt(args.mpt, tmp_path)

        output_zip = Path(args.output).expanduser().resolve() if args.output else _default_output_path(mpt_path)
        zip_path = _build_submission_zip(mpt_path, output_zip, copy_extras=not args.no_copy_extras)
        console.print(f"Built submission zip: {zip_path}")

        s3_dest = _resolve_s3_destination(
            mpt_arg=args.mpt,
            output_zip=zip_path,
            s3_dest=args.s3_dest,
            replace_s3=args.replace_s3,
        )
        if s3_dest:
            bucket, key = s3_dest
            console.print(f"Uploading bundle to s3://{bucket}/{key}")
            _upload_zip_to_s3(zip_path, bucket, key)
            if args.delete_source:
                src_bucket, src_key = _parse_s3_uri(args.mpt)
                console.print(f"Deleting source mpt s3://{src_bucket}/{src_key}")
                _delete_s3_object(src_bucket, src_key)

        if args.no_upload:
            return

        _upload_and_submit(
            zip_path=zip_path,
            name=args.name,
            season=args.season,
            server=args.server,
            login_server=args.login_server,
            console=console,
        )


if __name__ == "__main__":
    main()
