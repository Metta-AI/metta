from typing import Any, Literal, overload

import boto3

from metta.common.util.constants import METTA_AWS_REGION
from softmax.utils import memoize


@overload
def get_secretsmanager_secret(secret_name: str, require_exists: Literal[True] = True) -> str: ...


@overload
def get_secretsmanager_secret(secret_name: str, require_exists: Literal[False]) -> str | None: ...


@memoize(max_age=60 * 60)
def get_secretsmanager_secret(secret_name: str, require_exists: bool = True) -> str | None:
    """Fetch a secret value from AWS Secrets Manager"""
    client = boto3.client("secretsmanager", region_name=METTA_AWS_REGION)
    try:
        resp = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        if not require_exists:
            return None
        raise e

    if "SecretString" in resp and resp["SecretString"] is not None:
        try:
            return resp["SecretString"]
        except Exception as exc:
            raise ValueError("SecretString is not valid JSON") from exc

    raise ValueError("Expected SecretString with JSON content")


def create_secretsmanager_secret(
    secret_name: str,
    secret_value: str,
    *,
    allow_overwrite: bool = False,
) -> dict:
    """Create a secret (JSON value) or overwrite its current value."""
    client = boto3.client("secretsmanager", region_name=METTA_AWS_REGION)

    params: dict[str, Any] = {
        "Name": secret_name,
        "SecretString": secret_value,
    }
    try:
        return client.create_secret(**params)
    except client.exceptions.ResourceExistsException:
        if not allow_overwrite:
            raise

        put_params: dict[str, Any] = {"SecretId": secret_name, "SecretString": secret_value}
        return client.put_secret_value(**put_params)
