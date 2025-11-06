"""Tests for secrets module with AWS Secrets Manager integration."""

import os
import unittest.mock

import botocore.exceptions
import pytest

import gitta.secrets


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    original_env = {}
    env_vars = ["GITHUB_TOKEN", "ANTHROPIC_API_KEY", "AWS_REGION", "TEST_SECRET"]
    for var in env_vars:
        original_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    gitta.secrets.clear_cache()
    yield

    for var, value in original_env.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]

    gitta.secrets.clear_cache()


class TestEnvironmentVariableFallback:
    """Test environment variable fallback (no AWS)."""

    def test_get_secret_from_env_var(self):
        """Should retrieve secret from environment variable."""
        os.environ["TEST_SECRET"] = "test_value_123"
        result = gitta.secrets.get_secret("TEST_SECRET", required=False)
        assert result == "test_value_123"

    def test_get_secret_env_var_takes_precedence_over_aws(self):
        """Environment variable should be checked before AWS."""
        os.environ["TEST_SECRET"] = "env_value"
        result = gitta.secrets.get_secret("TEST_SECRET", required=False)
        assert result == "env_value"

    def test_get_secret_missing_not_required(self):
        """Should return None when secret not found and not required."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.side_effect = Exception("Not found")

            result = gitta.secrets.get_secret("MISSING_SECRET", required=False)
            assert result is None

    def test_get_secret_missing_required(self):
        """Should raise ValueError when secret not found and required."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.side_effect = Exception("Not found")

            with pytest.raises(ValueError, match="Secret not found"):
                gitta.secrets.get_secret("MISSING_SECRET", required=True)


class TestAWSSecretsManagerFallback:
    """Test AWS Secrets Manager fallback."""

    def test_get_secret_from_aws(self):
        """Should retrieve secret from AWS when env var not set."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "aws_value"}

            result = gitta.secrets.get_secret("TEST_SECRET", "test/secret", required=False)

            assert result == "aws_value"
            mock_client_constructor.assert_called_once_with("secretsmanager", region_name="us-east-1")
            mock_client.get_secret_value.assert_called_once_with(SecretId="test/secret")

    def test_get_secret_aws_default_secret_name(self):
        """Should convert env var name to AWS secret name."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "value"}

            gitta.secrets.get_secret("GITHUB_TOKEN", required=False)
            mock_client.get_secret_value.assert_called_with(SecretId="github/token")

    def test_get_secret_aws_custom_region(self):
        """Should use AWS_REGION env var if set."""
        os.environ["AWS_REGION"] = "us-west-2"

        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "value"}

            gitta.secrets.get_secret("TEST_SECRET", required=False)
            mock_client_constructor.assert_called_once_with("secretsmanager", region_name="us-west-2")

    def test_get_secret_aws_not_found(self):
        """Should handle AWS secret not found gracefully."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client

            import botocore.exceptions

            mock_client.get_secret_value.side_effect = botocore.exceptions.ClientError(
                {"Error": {"Code": "ResourceNotFoundException"}}, "GetSecretValue"
            )
            mock_client.exceptions.ResourceNotFoundException = botocore.exceptions.ClientError

            result = gitta.secrets.get_secret("MISSING", "missing/secret", required=False)
            assert result is None

    def test_get_secret_aws_not_found_required(self):
        """Should raise error when AWS secret not found and required."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client

            mock_client.get_secret_value.side_effect = botocore.exceptions.ClientError(
                {"Error": {"Code": "ResourceNotFoundException"}}, "GetSecretValue"
            )
            mock_client.exceptions.ResourceNotFoundException = botocore.exceptions.ClientError

            with pytest.raises(ValueError, match="Secret not found"):
                gitta.secrets.get_secret("MISSING", "missing/secret", required=True)

    def test_get_secret_aws_strips_whitespace(self):
        """Should strip whitespace from AWS secret values."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "  value_with_spaces  "}

            result = gitta.secrets.get_secret("TEST", required=False)
            assert result == "value_with_spaces"


class TestCaching:
    """Test caching behavior."""

    def test_cache_aws_secret(self):
        """Should cache AWS secret values."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "cached_value"}

            # First call - should hit AWS
            result1 = gitta.secrets.get_secret("TEST", required=False)
            assert result1 == "cached_value"
            assert mock_client.get_secret_value.call_count == 1

            # Second call - should use cache
            result2 = gitta.secrets.get_secret("TEST", required=False)
            assert result2 == "cached_value"
            assert mock_client.get_secret_value.call_count == 1  # Still 1, not 2

    def test_cache_expiry(self):
        """Should refetch from AWS after cache expires."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "value"}

            with unittest.mock.patch("time.time") as mock_time:
                # First call at t=0
                mock_time.return_value = 0
                gitta.secrets.get_secret("TEST", required=False)
                assert mock_client.get_secret_value.call_count == 1

                # Second call at t=3601 (after 1 hour expiry)
                mock_time.return_value = 3601
                gitta.secrets.get_secret("TEST", required=False)
                assert mock_client.get_secret_value.call_count == 2

    def test_cache_miss_is_cached(self):
        """Should cache secret misses to avoid repeated lookups."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client

            mock_client.get_secret_value.side_effect = botocore.exceptions.ClientError(
                {"Error": {"Code": "ResourceNotFoundException"}}, "GetSecretValue"
            )
            mock_client.exceptions.ResourceNotFoundException = botocore.exceptions.ClientError

            # First call - should hit AWS
            result1 = gitta.secrets.get_secret("MISSING", required=False)
            assert result1 is None
            assert mock_client.get_secret_value.call_count == 1

            # Second call - should use cached None
            result2 = gitta.secrets.get_secret("MISSING", required=False)
            assert result2 is None
            assert mock_client.get_secret_value.call_count == 1

    def test_clear_cache(self):
        """Should clear cache and force refetch."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "value"}

            # First call
            gitta.secrets.get_secret("TEST", required=False)
            assert mock_client.get_secret_value.call_count == 1

            # Clear cache
            gitta.secrets.clear_cache()

            # Next call should refetch
            gitta.secrets.get_secret("TEST", required=False)
            assert mock_client.get_secret_value.call_count == 2


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_github_token_from_env(self):
        """Should get GitHub token from environment."""
        os.environ["GITHUB_TOKEN"] = "ghp_test"
        token = gitta.secrets.get_github_token()
        assert token == "ghp_test"

    def test_get_github_token_from_aws(self):
        """Should get GitHub token from AWS."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "ghp_aws"}

            token = gitta.secrets.get_github_token()
            assert token == "ghp_aws"
            mock_client.get_secret_value.assert_called_with(SecretId="github/token")

    def test_get_github_token_not_found_optional(self):
        """Should return None when token not found and optional."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.side_effect = Exception("Not found")

            token = gitta.secrets.get_github_token(required=False)
            assert token is None

    def test_get_github_token_not_found_required(self):
        """Should raise error when token not found and required."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.side_effect = Exception("Not found")

            with pytest.raises(ValueError, match="Secret not found"):
                gitta.secrets.get_github_token(required=True)

    def test_get_anthropic_api_key_from_env(self):
        """Should get Anthropic API key from environment."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        key = gitta.secrets.get_anthropic_api_key()
        assert key == "sk-test"

    def test_get_anthropic_api_key_from_aws(self):
        """Should get Anthropic API key from AWS."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.return_value = {"SecretString": "sk-aws"}

            key = gitta.secrets.get_anthropic_api_key()
            assert key == "sk-aws"
            mock_client.get_secret_value.assert_called_with(SecretId="anthropic/api-key")

    def test_get_anthropic_api_key_not_found_optional(self):
        """Should return None when key not found and optional."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.side_effect = Exception("Not found")

            key = gitta.secrets.get_anthropic_api_key(required=False)
            assert key is None

    def test_get_anthropic_api_key_not_found_required(self):
        """Should raise error when key not found and required."""
        with unittest.mock.patch("gitta.secrets.boto3.client") as mock_client_constructor:
            mock_client = unittest.mock.MagicMock()
            mock_client_constructor.return_value = mock_client
            mock_client.get_secret_value.side_effect = Exception("Not found")

            with pytest.raises(ValueError, match="Secret not found"):
                gitta.secrets.get_anthropic_api_key(required=True)
