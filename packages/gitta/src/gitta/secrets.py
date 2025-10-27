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

logger = logging.getLogger(__name__)

# Cache for secrets with 1-hour TTL (matching softmax pattern)
_SECRET_CACHE: dict[str, tuple[str | None, float]] = {}
_CACHE_TTL = 3600  # 1 hour in seconds


def _has_boto3() -> bool:
    """Check if boto3 is available without importing it."""
    try:
        import boto3  # noqa: F401

        return True
    except ImportError:
        return False


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

    Tries sources in this order:
    1. Environment variable (if set)
    2. AWS Secrets Manager (if boto3 available and secret exists)
    3. None (if not required) or raise ValueError (if required)

    Results from AWS are cached for 1 hour to minimize API calls.

    Args:
        env_var: Environment variable name to check first
        aws_secret_name: AWS Secrets Manager secret name. If None, defaults to
            env_var.lower().replace('_', '/') (e.g., GITHUB_TOKEN -> github/token)
        required: If True, raise ValueError when secret not found anywhere

    Returns:
        Secret value or None if not found and not required

    Raises:
        ValueError: If required=True and secret not found anywhere

    Examples:
        >>> # Try env var first, then AWS, return None if not found
        >>> token = get_secret("GITHUB_TOKEN")

        >>> # Try env var first, then AWS, raise if not found
        >>> api_key = get_secret("ANTHROPIC_API_KEY", required=True)

        >>> # Use custom AWS secret name
        >>> token = get_secret("GITHUB_TOKEN", "github/gitta-token")
    """
    # Try environment variable first
    value = os.environ.get(env_var)
    if value:
        logger.debug(f"Using {env_var} from environment variable")
        return value

    # Try AWS Secrets Manager if boto3 is available
    if _has_boto3():
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
                # Cached None but required - fall through to raise error

        # Fetch from AWS
        try:
            import boto3

            # Use AWS_REGION env var or fall back to us-east-1
            region = os.environ.get("AWS_REGION", "us-east-1")
            client = boto3.client("secretsmanager", region_name=region)

            try:
                response = client.get_secret_value(SecretId=aws_secret_name)
                if "SecretString" in response:
                    value = response["SecretString"].strip()
                    logger.info(f"Using {aws_secret_name} from AWS Secrets Manager")
                    # Cache the value
                    _SECRET_CACHE[cache_key] = (value, time.time())
                    return value
            except client.exceptions.ResourceNotFoundException:
                logger.debug(f"AWS secret {aws_secret_name} not found")
                # Cache the miss to avoid repeated lookups
                _SECRET_CACHE[cache_key] = (None, time.time())
            except Exception as e:
                logger.debug(f"Failed to fetch AWS secret {aws_secret_name}: {e}")
                # Don't cache errors - might be transient (throttling, permissions, etc.)

        except Exception as e:
            logger.debug(f"AWS Secrets Manager unavailable: {e}")

    # Not found anywhere
    if required:
        aws_hint = ""
        if aws_secret_name is None:
            aws_secret_name = env_var.lower().replace("_", "/")

        if _has_boto3():
            aws_hint = f" or AWS Secrets Manager secret '{aws_secret_name}'"

        raise ValueError(
            f"Secret not found: {env_var} environment variable not set{aws_hint}. "
            f"Please set {env_var} environment variable or configure AWS secret."
        )

    return None


def get_github_token(required: bool = False) -> str | None:
    """Get GitHub token from environment or AWS Secrets Manager.

    Tries GITHUB_TOKEN environment variable first, then falls back to
    AWS Secrets Manager secret 'github/token' if boto3 is available.

    Args:
        required: If True, raise ValueError when token not found

    Returns:
        GitHub token or None if not found and not required

    Raises:
        ValueError: If required=True and token not found

    Examples:
        >>> # Optional token - returns None if not found
        >>> token = get_github_token()
        >>> if token:
        >>>     # Make authenticated API call
        >>>     pass

        >>> # Required token - raises error if not found
        >>> token = get_github_token(required=True)
        >>> # Guaranteed to have token here
    """
    return get_secret("GITHUB_TOKEN", "github/token", required=required)


def get_anthropic_api_key(required: bool = False) -> str | None:
    """Get Anthropic API key from environment or AWS Secrets Manager.

    Tries ANTHROPIC_API_KEY environment variable first, then falls back to
    AWS Secrets Manager secret 'anthropic/api-key' if boto3 is available.

    Args:
        required: If True, raise ValueError when API key not found

    Returns:
        Anthropic API key or None if not found and not required

    Raises:
        ValueError: If required=True and API key not found

    Examples:
        >>> # Optional key - returns None if not found
        >>> key = get_anthropic_api_key()

        >>> # Required key - raises error if not found
        >>> key = get_anthropic_api_key(required=True)
    """
    return get_secret("ANTHROPIC_API_KEY", "anthropic/api-key", required=required)


def clear_cache() -> None:
    """Clear the secret cache.

    This forces fresh retrieval from AWS Secrets Manager on the next call.
    Useful for testing or when you know secrets have been rotated.

    Example:
        >>> clear_cache()
        >>> # Next call will fetch fresh from AWS
        >>> token = get_github_token()
    """
    _SECRET_CACHE.clear()
    logger.debug("Secret cache cleared")
