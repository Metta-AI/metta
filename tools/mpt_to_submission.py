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
    parser = argparse.ArgumentParser(description="Convert legacy .mpt to submission zip and submit to Observatory.")
    parser.add_argument("--mpt", required=True, help="Path or s3:// URI to legacy .mpt file")
    parser.add_argument("--name", required=True, help="Submission policy name")
    parser.add_argument("--season", help="Tournament season to submit to (e.g., beta)")
    parser.add_argument(
        "--output",
        help="Output zip path (defaults to /tmp/<mpt_basename>.zip)",
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
        help="Only build the zip; do not upload or submit",
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


def _build_submission_zip(mpt_path: Path, output_zip: Path) -> Path:
    with zipfile.ZipFile(mpt_path, "r") as archive:
        names = set(archive.namelist())
        if "weights.safetensors" not in names or "modelarchitecture.txt" not in names:
            raise ValueError("Expected weights.safetensors and modelarchitecture.txt in legacy .mpt archive")
        architecture_spec = archive.read("modelarchitecture.txt").decode("utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            archive.extract("weights.safetensors", tmp_path)

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

    return output_zip


def _default_output_path(mpt_path: Path) -> Path:
    return Path("/tmp") / f"{mpt_path.stem}.zip"


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

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mpt_path = _download_mpt(args.mpt, tmp_path)

        output_zip = Path(args.output).expanduser().resolve() if args.output else _default_output_path(mpt_path)
        zip_path = _build_submission_zip(mpt_path, output_zip)
        console.print(f"Built submission zip: {zip_path}")

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
