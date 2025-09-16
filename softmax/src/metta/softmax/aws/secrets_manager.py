import json
from typing import Any

import boto3
from botocore.exceptions import ClientError


def get_secret(secret_name: str) -> dict[str, Any]:
    """Fetch a secret value from AWS Secrets Manager as JSON.

    Always expects SecretString to contain JSON. Raises ValueError if not JSON.
    """
    client = boto3.client("secretsmanager")
    try:
        resp = client.get_secret_value(SecretId=secret_name)
    except ClientError:
        raise

    if "SecretString" in resp and resp["SecretString"] is not None:
        try:
            return json.loads(resp["SecretString"])
        except Exception as exc:
            raise ValueError("SecretString is not valid JSON") from exc

    raise ValueError("Expected SecretString with JSON content")


def create_secret(
    secret_name: str,
    secret_value: dict[str, Any],
    *,
    allow_overwrite: bool = False,
) -> dict:
    """Create a secret (JSON value) or overwrite its current value."""
    client = boto3.client("secretsmanager")

    params: dict[str, Any] = {
        "Name": secret_name,
        "SecretString": json.dumps(secret_value),
    }
    try:
        resp = client.create_secret(**params)
        return resp["ARN"]
    except client.exceptions.ResourceExistsException:
        if not allow_overwrite:
            raise

        put_params: dict[str, Any] = {"SecretId": secret_name, "SecretString": json.dumps(secret_value)}
        return client.put_secret_value(**put_params)
