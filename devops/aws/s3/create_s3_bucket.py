#!/usr/bin/env python3
"""
Script to create S3 bucket in us-east-1.

Usage:
    python create_s3_bucket.py mettagrid-cache-bucket
"""

import argparse
import json
import sys
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from metta.util.colorama import blue, bold, cyan, green, red, yellow


def create_bucket(bucket_name: str) -> bool:
    """Create S3 bucket in us-east-1."""
    try:
        s3_client = boto3.client("s3", region_name="us-east-1")

        s3_client.create_bucket(Bucket=bucket_name)

        print(green(f"✓ Created bucket: {bucket_name}"))
        return True

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "BucketAlreadyExists":
            print(red(f"✗ Bucket {bucket_name} already exists globally"))
        elif error_code == "BucketAlreadyOwnedByYou":
            print(green(f"✓ Bucket {bucket_name} already exists and is owned by you"))
            return True
        else:
            print(red(f"✗ Error creating bucket: {e}"))
        return False
    except Exception as e:
        print(red(f"✗ Unexpected error: {e}"))
        return False


def setup_lifecycle_policy(bucket_name: str) -> bool:
    """Set up lifecycle policy for cache cleanup."""
    try:
        s3_client = boto3.client("s3", region_name="us-east-1")

        lifecycle_config = {
            "Rules": [
                {
                    "ID": "CacheCleanup",
                    "Status": "Enabled",
                    "Filter": {},  # Empty filter applies to all objects
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                    "Expiration": {"Days": 30},
                }
            ]
        }

        s3_client.put_bucket_lifecycle_configuration(Bucket=bucket_name, LifecycleConfiguration=lifecycle_config)
        print(green(f"✓ Set up lifecycle policy on {bucket_name} (30-day expiration)"))
        return True

    except Exception as e:
        print(red(f"✗ Error setting up lifecycle policy: {e}"))
        return False


def setup_bucket_policy(bucket_name: str, account_id: Optional[str] = None) -> bool:
    """Set up bucket policy for team access."""
    try:
        if not account_id:
            sts_client = boto3.client("sts")
            account_id = sts_client.get_caller_identity()["Account"]

        s3_client = boto3.client("s3", region_name="us-east-1")

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "CacheAccess",
                    "Effect": "Allow",
                    "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
                    "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}", f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }

        s3_client.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(policy))
        print(green(f"✓ Set up bucket policy on {bucket_name}"))
        return True

    except Exception as e:
        print(red(f"✗ Error setting up bucket policy: {e}"))
        return False


def check_aws_credentials() -> bool:
    """Check if AWS credentials are configured."""
    try:
        sts_client = boto3.client("sts")
        identity = sts_client.get_caller_identity()
        print(green(f"✓ AWS credentials configured for: {identity.get('Arn', 'Unknown')}"))
        return True
    except NoCredentialsError:
        print(red("✗ AWS credentials not configured. Run 'aws configure' first."))
        return False
    except Exception as e:
        print(red(f"✗ Error checking AWS credentials: {e}"))
        return False


def main():
    parser = argparse.ArgumentParser(description="Create S3 bucket for MettaGrid cache in us-east-1")
    parser.add_argument("bucket_name", help="S3 bucket name (must be globally unique)")

    args = parser.parse_args()

    if not check_aws_credentials():
        sys.exit(1)

    bucket_name = args.bucket_name

    print(f"\n{bold('Creating S3 bucket:')} {cyan(bucket_name)} {bold('in us-east-1')}")
    print("-" * 50)

    # Create bucket
    if not create_bucket(bucket_name):
        sys.exit(1)

    success_count = 1  # Bucket creation succeeded
    total_operations = 3  # Bucket + lifecycle + policy

    # Always set up lifecycle and policy
    if setup_lifecycle_policy(bucket_name):
        success_count += 1

    if setup_bucket_policy(bucket_name):
        success_count += 1

    print("\n" + "=" * 50)
    print(bold("SUMMARY"))
    print("=" * 50)
    print(f"Operations completed: {green(str(success_count))}/{total_operations}")
    print(f"Bucket: {cyan(bucket_name)}")
    print(f"Region: {blue('us-east-1')}")
    print(f"S3 URL: {yellow(f's3://{bucket_name}')}")

    if success_count == total_operations:
        print(green("✓ All operations completed successfully!"))
        print(green("✓ Bucket configured as simple key-value store with 30-day cleanup"))
    else:
        print(yellow("⚠ Some operations failed. Check the output above."))

    print(f"\n{bold('Ready to use with:')} METTAGRID_CACHE_BUCKET={cyan(bucket_name)}")


if __name__ == "__main__":
    main()
