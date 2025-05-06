import os
from logging import Logger
from typing import Any, Dict, Optional

import boto3


def upload_file(
    file_path: str,
    s3_key: str,
    bucket_name: str = "softmax-public",
    content_type: Optional[str] = None,
    extra_args: Optional[Dict[str, Any]] = None,
    logger: Optional[Logger] = None,
    skip_if_ci: bool = True,
) -> Optional[str]:
    """
    Uploads a file to S3 with configurable parameters.

    Args:
        file_path: Local path to the file to upload
        s3_key: The S3 object key (path within the bucket)
        bucket_name: S3 bucket name (default: "softmax-public")
        content_type: Optional content type for the file
        extra_args: Optional extra arguments for S3 upload
        logger: Optional logger for logging messages
        skip_if_ci: If True, skips upload in CI environments (default: True)

    Returns:
        The URL of the uploaded file in S3, or None if upload was skipped or failed
    """
    # Check if running in CI environment - abort upload if in CI and skip_if_ci is True
    is_ci = os.environ.get("CI", "").lower() in ("1", "true", "yes")
    if is_ci and skip_if_ci:
        if logger:
            logger.info("Running in CI environment, skipping S3 upload")
        return None

    # Prepare extra arguments for the upload
    upload_args = {}
    if content_type:
        upload_args["ContentType"] = content_type

    # Add any additional arguments provided
    if extra_args:
        upload_args.update(extra_args)

    try:
        # Initialize S3 client
        s3_client = boto3.client("s3")

        # Upload the file
        s3_client.upload_file(
            Filename=file_path,
            Bucket=bucket_name,
            Key=s3_key,
            ExtraArgs=upload_args,
        )

        # Generate and return the S3 URL
        s3_url = f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{s3_key}"

        if logger:
            logger.info(f"Successfully uploaded file to S3: {s3_url}")

        return s3_url

    except Exception as e:
        if logger:
            logger.error(f"Failed to upload file to S3: {str(e)}")
        raise
