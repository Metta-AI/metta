"""AWS Secrets Manager integration for API keys and tokens.

This module provides fallback to AWS Secrets Manager when environment variables
are not set.

Example usage:
    # Basic usage - try env var, then AWS, return None if not found
    token = get_github_token()

    # Require the secret - raise error if not found
    api_key = get_anthropic_api_key(required=True)

    # Generic secret retrieval with custom AWS name
    value = get_secret("MY_SECRET", "my-app/secret", required=False)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal, overload

import boto3

logger = logging.getLogger(__name__)

# Cache for secrets with 1-hour TTL
_SECRET_CACHE: dict[str, tuple[str | None, float]] = {}
_CACHE_TTL = 3600  # 1 hour in seconds


@overload
def get_secret(
    env_var: str,
    aws_secret_name: str | None = None,
    *,
    required: Literal[True] = True,
) -> str: ...


@overload
def get_secret(
    env_var: str,
    aws_secret_name: str | None = None,
    *,
    required: Literal[False],
) -> str | None: ...


def get_secret(
    env_var: str,
    aws_secret_name: str | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Get a secret from environment variable or AWS Secrets Manager.

    Checks environment variable first, then AWS Secrets Manager. AWS results are
    cached for 1 hour. If required=True, raises ValueError when not found.
    If aws_secret_name is None, defaults to env_var.lower().replace('_', '/').

    Examples:
        >>> token = get_secret("GITHUB_TOKEN")  # Optional, returns None if not found
        >>> api_key = get_secret("ANTHROPIC_API_KEY", required=True)  # Required
        >>> token = get_secret("GITHUB_TOKEN", "github/gitta-token")  # Custom AWS name
    """
    # Try environment variable first
    value = os.environ.get(env_var)
    if value:
        logger.debug(f"Using {env_var} from environment variable")
        return value

    # Try AWS Secrets Manager
    # Default AWS secret name: convert ENV_VAR to env/var
    if aws_secret_name is None:
        aws_secret_name = env_var.lower().replace("_", "/")

    # Check cache first
    cache_key = f"aws:{aws_secret_name}"
    if cache_key in _SECRET_CACHE:
        cached_value, cached_time = _SECRET_CACHE[cache_key]
        if time.time() - cached_time < _CACHE_TTL:
            logger.debug(f"Using {aws_secret_name} from cache")
            if cached_value is not None or not required:
                return cached_value

    # Fetch from AWS
    try:
        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)

        try:
            response = client.get_secret_value(SecretId=aws_secret_name)
            if "SecretString" in response:
                value = response["SecretString"].strip()
                logger.info(f"Using {aws_secret_name} from AWS Secrets Manager")
                _SECRET_CACHE[cache_key] = (value, time.time())
                return value
        except client.exceptions.ResourceNotFoundException:
            logger.debug(f"AWS secret {aws_secret_name} not found")
            _SECRET_CACHE[cache_key] = (None, time.time())
        except Exception as e:
            logger.debug(f"Failed to fetch AWS secret {aws_secret_name}: {e}")

    except Exception as e:
        logger.debug(f"AWS Secrets Manager unavailable: {e}")

    # Not found anywhere
    if required:
        raise ValueError(
            f"Secret not found: {env_var} environment variable not set or "
            f"AWS Secrets Manager secret '{aws_secret_name}' not found. "
            f"Please set {env_var} environment variable or configure AWS secret."
        )

    return None


def get_github_token(required: bool = False) -> str | None:
    """Get GitHub token from GITHUB_TOKEN env var or AWS secret 'github/token'."""
    return get_secret("GITHUB_TOKEN", "github/token", required=required)


def get_anthropic_api_key(required: bool = False) -> str | None:
    """Get Anthropic API key from ANTHROPIC_API_KEY env var or AWS secret 'anthropic/api-key'."""
    return get_secret("ANTHROPIC_API_KEY", "anthropic/api-key", required=required)


def clear_cache() -> None:
    """Clear the AWS secret cache (env vars are never cached, always checked first)."""
    _SECRET_CACHE.clear()
    logger.debug("Secret cache cleared")
