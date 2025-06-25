#!/usr/bin/env -S uv run
import argparse
import mimetypes
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from metta.common.util.uv_check import enforce_uv


def is_image_file(filename):
    """Check if a file is an image based on its mimetype."""
    mimetype, _ = mimetypes.guess_type(filename)
    return mimetype and mimetype.startswith("image/")


def get_proper_filename(filepath):
    """Ensure the file has an appropriate image extension."""
    path = Path(filepath)

    # If file already has an extension, use it
    if path.suffix:
        return path.name

    # Otherwise, try to determine extension from mimetype
    mimetype, _ = mimetypes.guess_type(filepath)
    if mimetype:
        ext = mimetypes.guess_extension(mimetype)
        if ext:
            return f"{path.name}{ext}"

    # Fallback to .jpg if we can't determine the type
    return f"{path.name}.jpg"


def upload_to_s3(file_path, s3_bucket, s3_prefix):
    """Upload a file to S3."""
    s3_client = boto3.client("s3")

    dest_filename = get_proper_filename(file_path)
    s3_key = f"{s3_prefix.rstrip('/')}/{dest_filename}"

    try:
        print(f"Uploading {file_path} to s3://{s3_bucket}/{s3_key}")
        s3_client.upload_file(file_path, s3_bucket, s3_key)
        return True
    except ClientError as e:
        print(f"Error uploading {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload images to S3")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without uploading files")
    args = parser.parse_args()

    # S3 destination details
    s3_bucket = "softmax-public"
    s3_prefix = "policydash/evals/img"

    uploaded_count = 0
    skipped_count = 0

    for filename in os.listdir("."):
        if os.path.isfile(filename) and is_image_file(filename):
            if args.dry_run:
                print(f"[DRY RUN] Would upload {filename} to s3://{s3_bucket}/{s3_prefix}")
                continue

            if upload_to_s3(filename, s3_bucket, s3_prefix):
                uploaded_count += 1
            else:
                skipped_count += 1

    print(f"\nSummary: {uploaded_count} images {'would be ' if args.dry_run else ''}uploaded, {skipped_count} skipped")


if __name__ == "__main__":
    enforce_uv()
    main()
