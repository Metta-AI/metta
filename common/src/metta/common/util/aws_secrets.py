"""AWS Secrets Manager utilities for Metta."""

import time
from functools import wraps
from typing import Any, Literal, overload

import boto3

from metta.common.util.constants import METTA_AWS_REGION


def memoize(max_age: int = 60):
    """Cache function results with time-based expiration."""

    def decorator(func):
        cache = {}
        cache_time = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = hash((args, tuple(sorted(kwargs.items()))))
            current_time = time.time()

            if key in cache and current_time - cache_time[key] < max_age:
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            cache_time[key] = current_time
            return result

        return wrapper

    return decorator


@overload
def get_secretsmanager_secret(secret_name: str, require_exists: Literal[True] = True) -> str: ...


@overload
def get_secretsmanager_secret(secret_name: str, require_exists: Literal[False]) -> str | None: ...


@memoize(max_age=60 * 60)
def get_secretsmanager_secret(secret_name: str, require_exists: bool = True) -> str | None:
    """Fetch a secret value from AWS Secrets Manager with 1-hour caching."""
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
    """Create a new secret or overwrite an existing one if allow_overwrite is True."""
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
