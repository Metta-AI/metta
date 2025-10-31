from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from metta.common.util.fs import get_repo_root
from metta.tools.pr_similarity import DEFAULT_CACHE_PATH, resolve_cache_paths


def download_cache_from_s3(cache_path: Path, bucket: str = "softmax-public", prefix: str = "pr-cache/") -> bool:
    """Download PR embedding cache from S3 if it doesn't exist locally.

    Returns True if cache was downloaded or already exists, False if download failed.
    """
    meta_path, vectors_path = resolve_cache_paths(cache_path)

    if meta_path.exists() and vectors_path.exists():
        print(f"Cache already exists at {cache_path}")
        return True

    try:
        session = boto3.session.Session()
        client = session.client("s3")
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        client.download_file(bucket, prefix + meta_path.name, str(meta_path))
        client.download_file(bucket, prefix + vectors_path.name, str(vectors_path))
        print(f"Downloaded PR similarity cache from s3://{bucket}/{prefix}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("Cache not found in S3 (this is expected on first run)")
        else:
            print(f"Error downloading cache from S3: {e}")
        return False
    except Exception as e:
        print(f"Unable to download PR similarity cache: {e}")
        return False


def upload_cache_to_s3(cache_path: Path, bucket: str = "softmax-public", prefix: str = "pr-cache/") -> None:
    """Upload PR embedding cache to S3."""
    meta_path, vectors_path = resolve_cache_paths(cache_path)

    if not meta_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(f"Cache files not found: {meta_path} / {vectors_path}")

    try:
        session = boto3.session.Session()
        client = session.client("s3")

        client.upload_file(str(meta_path), bucket, prefix + meta_path.name)
        client.upload_file(str(vectors_path), bucket, prefix + vectors_path.name)
        print(f"Uploaded PR similarity cache to s3://{bucket}/{prefix}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload cache to S3: {e}") from e


def main() -> None:
    """Download cache from S3, update it with new PRs, and upload back to S3."""
    repo_root = get_repo_root()
    cache_path_env = os.getenv("PR_SIMILARITY_CACHE_PATH")
    if cache_path_env:
        cache_path = Path(cache_path_env)
    else:
        cache_path = repo_root / DEFAULT_CACHE_PATH

    print("Downloading cache from S3...")
    download_succeeded = download_cache_from_s3(cache_path)

    meta_path, vectors_path = resolve_cache_paths(cache_path)
    cache_exists_locally = meta_path.exists() and vectors_path.exists()

    build_cache_script = repo_root / "mcp_servers" / "pr_similarity" / "build_cache.py"

    if not download_succeeded and not cache_exists_locally:
        raise FileNotFoundError(
            f"Failed to download cache from S3 and no local cache exists at {cache_path}. "
            "This would trigger a full rebuild of all PR embeddings, which is expensive. "
            "If this is intentional (e.g., first-time setup), manually run: "
            f"python {build_cache_script} --cache-path {cache_path} --allow-full-rebuild"
        )

    print("Building/updating cache...")
    if not build_cache_script.exists():
        raise FileNotFoundError(f"build_cache.py not found at {build_cache_script}")

    result = subprocess.run(
        [sys.executable, str(build_cache_script), "--cache-path", str(cache_path)],
        check=False,
    )

    if result.returncode != 0:
        print(f"Error building cache (exit code {result.returncode})")
        sys.exit(result.returncode)

    print("Uploading updated cache to S3...")
    upload_cache_to_s3(cache_path)
    print("Cache refresh complete")


if __name__ == "__main__":
    main()
