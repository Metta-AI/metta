import logging
import boto3
from typing import Tuple, Optional

def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Parse S3 path into bucket and key components"""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with s3://")
    
    s3_parts = s3_path[5:].split("/", 1)
    if len(s3_parts) < 2:
        raise ValueError(f"Invalid S3 path: {s3_path}. Must be in format s3://bucket/path")
    
    return s3_parts[0], s3_parts[1]

def download_from_s3(s3_path: str, local_path: str) -> bool:
    """Download file from S3 to local path. Returns True iff file was downloaded successfully."""
    logger = logging.getLogger(__name__)

    bucket, key = parse_s3_path(s3_path)
    try:
        s3_client = boto3.client('s3')
        # Check if file exists
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded file from s3://{bucket}/{key} to {local_path}")
            return True
        except Exception as e:
            logger.info(f"No existing file at s3://{bucket}/{key}: {e}")
            return False
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        return False

def upload_to_s3(local_path: str, s3_path: str) -> bool:
    """Upload file to S3"""
    logger = logging.getLogger(__name__)

    bucket, key = parse_s3_path(s3_path)
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)
        logger.info(f"Uploaded file to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return False

def upload_content_to_s3(content: str, s3_path: str, content_type: str = "text/html") -> bool:
    """Upload string content directly to S3"""
    logger = logging.getLogger(__name__)

    bucket, key = parse_s3_path(s3_path)
    try:
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Body=content,
            Bucket=bucket,
            Key=key,
            ContentType=content_type
        )
        logger.info(f"Uploaded content to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"Error uploading content to S3: {e}")
        return False

def check_s3_exists(s3_path: str) -> bool:
    """Check if a file exists in S3"""
    bucket, key = parse_s3_path(s3_path)
    try:
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False