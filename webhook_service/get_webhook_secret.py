#!/usr/bin/env python3
"""Get GitHub webhook secret from AWS Secrets Manager."""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "softmax" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "common" / "src"))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ Error: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)


def get_secret():
    """Get GitHub webhook secret from AWS Secrets Manager."""
    secret_name = "github/webhook-secret"
    region_name = "us-east-1"

    try:
        session = boto3.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)

        response = client.get_secret_value(SecretId=secret_name)
        secret = response["SecretString"]
        print("✅ Retrieved secret from AWS Secrets Manager")
        print("\nSet it as:")
        print(f"export GITHUB_WEBHOOK_SECRET='{secret}'")
        return secret
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            print(f"❌ Secret '{secret_name}' not found in AWS Secrets Manager")
        else:
            print(f"❌ Error retrieving secret: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    get_secret()


