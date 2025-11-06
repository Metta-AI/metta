import typing

import boto3

import metta.common.util.constants
import softmax.utils


@typing.overload
def get_secretsmanager_secret(secret_name: str, require_exists: typing.Literal[True] = True) -> str: ...


@typing.overload
def get_secretsmanager_secret(secret_name: str, require_exists: typing.Literal[False]) -> str | None: ...


@softmax.utils.memoize(max_age=60 * 60)
def get_secretsmanager_secret(secret_name: str, require_exists: bool = True) -> str | None:
    """Fetch a secret value from AWS Secrets Manager"""
    client = boto3.client("secretsmanager", region_name=metta.common.util.constants.METTA_AWS_REGION)
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
    client = boto3.client("secretsmanager", region_name=metta.common.util.constants.METTA_AWS_REGION)

    params: dict[str, typing.Any] = {
        "Name": secret_name,
        "SecretString": secret_value,
    }
    try:
        return client.create_secret(**params)
    except client.exceptions.ResourceExistsException:
        if not allow_overwrite:
            raise

        put_params: dict[str, typing.Any] = {"SecretId": secret_name, "SecretString": secret_value}
        return client.put_secret_value(**put_params)
