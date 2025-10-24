"""AWS Secrets Manager utilities for collectors."""

from functools import lru_cache

import boto3


@lru_cache(maxsize=128)
def get_secretsmanager_secret(secret_name: str) -> str:
    """Fetch a secret value from AWS Secrets Manager.

    Cached to avoid repeated API calls for the same secret.
    """
    client = boto3.client("secretsmanager", region_name="us-east-1")
    resp = client.get_secret_value(SecretId=secret_name)

    if "SecretString" in resp and resp["SecretString"] is not None:
        return resp["SecretString"]

    raise ValueError(f"Secret {secret_name} does not contain SecretString")
