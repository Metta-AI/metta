"""Tests for secrets module with AWS Secrets Manager integration."""

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from gitta.secrets import (
    clear_cache,
    get_anthropic_api_key,
    get_github_token,
    get_secret,
)


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test."""
    # Store original values
    original_env = {}
    env_vars = ["GITHUB_TOKEN", "ANTHROPIC_API_KEY", "AWS_REGION", "TEST_SECRET"]
    for var in env_vars:
        original_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    # Clear cache before each test
    clear_cache()

    yield

    # Restore original values
    for var, value in original_env.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]

    # Clear cache after each test
    clear_cache()


class TestEnvironmentVariableFallback:
    """Test environment variable fallback (no AWS)."""

    def test_get_secret_from_env_var(self):
        """Should retrieve secret from environment variable."""
        os.environ["TEST_SECRET"] = "test_value_123"
        result = get_secret("TEST_SECRET", required=False)
        assert result == "test_value_123"

    def test_get_secret_env_var_takes_precedence_over_aws(self):
        """Environment variable should be checked before AWS."""
        os.environ["TEST_SECRET"] = "env_value"

        # Even with AWS available, env var should win
        # No need to mock boto3 - env var takes precedence so boto3 never imported
        result = get_secret("TEST_SECRET", required=False)
        assert result == "env_value"

    def test_get_secret_missing_not_required(self):
        """Should return None when secret not found and not required."""
        result = get_secret("MISSING_SECRET", required=False)
        assert result is None

    def test_get_secret_missing_required(self):
        """Should raise ValueError when secret not found and required."""
        with pytest.raises(ValueError, match="Secret not found"):
            get_secret("MISSING_SECRET", required=True)


class TestAWSSecretsManagerFallback:
    """Test AWS Secrets Manager fallback."""

    def test_get_secret_from_aws(self):
        """Should retrieve secret from AWS when env var not set."""
        # Create mock boto3 module
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "aws_value"}

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                result = get_secret("TEST_SECRET", "test/secret", required=False)

                assert result == "aws_value"
                mock_boto3.client.assert_called_once_with("secretsmanager", region_name="us-east-1")
                mock_client.get_secret_value.assert_called_once_with(SecretId="test/secret")

    def test_get_secret_aws_default_secret_name(self):
        """Should convert env var name to AWS secret name."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                mock_client.get_secret_value.return_value = {"SecretString": "value"}

                # GITHUB_TOKEN -> github/token
                get_secret("GITHUB_TOKEN", required=False)
                mock_client.get_secret_value.assert_called_with(SecretId="github/token")

    def test_get_secret_aws_custom_region(self):
        """Should use AWS_REGION env var if set."""
        os.environ["AWS_REGION"] = "us-west-2"

        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                mock_client.get_secret_value.return_value = {"SecretString": "value"}

                get_secret("TEST_SECRET", required=False)

                mock_boto3.client.assert_called_once_with("secretsmanager", region_name="us-west-2")

    def test_get_secret_aws_not_found(self):
        """Should cache misses and return None when secret not found in AWS."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                # Simulate secret not found
                from botocore.exceptions import ClientError

                error_response = {"Error": {"Code": "ResourceNotFoundException"}}
                mock_client.exceptions.ResourceNotFoundException = ClientError
                mock_client.get_secret_value.side_effect = ClientError(error_response, "GetSecretValue")

                result = get_secret("MISSING_SECRET", required=False)

                assert result is None

    def test_get_secret_aws_not_found_required(self):
        """Should raise ValueError when AWS secret not found and required."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                from botocore.exceptions import ClientError

                error_response = {"Error": {"Code": "ResourceNotFoundException"}}
                mock_client.exceptions.ResourceNotFoundException = ClientError
                mock_client.get_secret_value.side_effect = ClientError(error_response, "GetSecretValue")

                with pytest.raises(ValueError, match="Secret not found"):
                    get_secret("MISSING_SECRET", required=True)

    def test_get_secret_aws_strips_whitespace(self):
        """Should strip whitespace from AWS secret values."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                mock_client.get_secret_value.return_value = {"SecretString": "  value_with_spaces  \n"}

                result = get_secret("TEST_SECRET", required=False)

                assert result == "value_with_spaces"


class TestBoto3NotAvailable:
    """Test graceful degradation when boto3 is not available."""

    def test_get_secret_without_boto3(self):
        """Should work without boto3 installed."""
        with patch("gitta.secrets._has_boto3", return_value=False):
            # Should return None since env var not set and boto3 not available
            result = get_secret("TEST_SECRET", required=False)
            assert result is None

    def test_get_secret_without_boto3_required(self):
        """Should raise error when boto3 not available and secret required."""
        with patch("gitta.secrets._has_boto3", return_value=False):
            with pytest.raises(ValueError, match="Secret not found"):
                get_secret("TEST_SECRET", required=True)

    def test_get_secret_without_boto3_with_env_var(self):
        """Should still use env var when boto3 not available."""
        os.environ["TEST_SECRET"] = "env_value"

        with patch("gitta.secrets._has_boto3", return_value=False):
            result = get_secret("TEST_SECRET", required=False)
            assert result == "env_value"


class TestCaching:
    """Test caching behavior."""

    def test_cache_aws_secret(self):
        """Should cache AWS secrets for 1 hour."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                mock_client.get_secret_value.return_value = {"SecretString": "cached_value"}

                # First call - should fetch from AWS
                result1 = get_secret("TEST_SECRET", required=False)
                assert result1 == "cached_value"
                assert mock_client.get_secret_value.call_count == 1

                # Second call - should use cache
                result2 = get_secret("TEST_SECRET", required=False)
                assert result2 == "cached_value"
                assert mock_client.get_secret_value.call_count == 1  # Still 1, from cache

    def test_cache_expiry(self):
        """Should re-fetch after cache expires."""
        # Create mock boto3 module
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.get_secret_value.return_value = {"SecretString": "value"}

        # Mock time to control cache expiry
        current_time = time.time()
        with patch("gitta.secrets.time.time") as mock_time:
            # First call - set current time
            mock_time.return_value = current_time

            with patch("gitta.secrets._has_boto3", return_value=True):
                with patch.dict(sys.modules, {"boto3": mock_boto3}):
                    get_secret("TEST_SECRET", required=False)
                    assert mock_client.get_secret_value.call_count == 1

                    # Advance time by more than 1 hour
                    mock_time.return_value = current_time + 3601

                    # Should re-fetch
                    get_secret("TEST_SECRET", required=False)
                    assert mock_client.get_secret_value.call_count == 2

    def test_cache_miss_is_cached(self):
        """Should cache misses to avoid repeated lookups."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                from botocore.exceptions import ClientError

                error_response = {"Error": {"Code": "ResourceNotFoundException"}}
                mock_client.exceptions.ResourceNotFoundException = ClientError
                mock_client.get_secret_value.side_effect = ClientError(error_response, "GetSecretValue")

                # First call - should try AWS
                result1 = get_secret("MISSING_SECRET", required=False)
                assert result1 is None
                assert mock_client.get_secret_value.call_count == 1

                # Second call - should use cached miss
                result2 = get_secret("MISSING_SECRET", required=False)
                assert result2 is None
                assert mock_client.get_secret_value.call_count == 1  # Still 1

    def test_clear_cache(self):
        """Should clear cache and re-fetch."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                mock_client.get_secret_value.return_value = {"SecretString": "value"}

                # First call
                get_secret("TEST_SECRET", required=False)
                assert mock_client.get_secret_value.call_count == 1

                # Clear cache
                clear_cache()

                # Should re-fetch
                get_secret("TEST_SECRET", required=False)
                assert mock_client.get_secret_value.call_count == 2


class TestHelperFunctions:
    """Test helper functions for specific secrets."""

    def test_get_github_token_from_env(self):
        """Should get GitHub token from environment variable."""
        os.environ["GITHUB_TOKEN"] = "ghp_test123"
        result = get_github_token(required=False)
        assert result == "ghp_test123"

    def test_get_github_token_from_aws(self):
        """Should get GitHub token from AWS Secrets Manager."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                mock_client.get_secret_value.return_value = {"SecretString": "ghp_from_aws"}

                result = get_github_token(required=False)

                assert result == "ghp_from_aws"
                mock_client.get_secret_value.assert_called_with(SecretId="github/token")

    def test_get_github_token_not_found_optional(self):
        """Should return None when GitHub token not found and not required."""
        result = get_github_token(required=False)
        assert result is None

    def test_get_github_token_not_found_required(self):
        """Should raise error when GitHub token not found and required."""
        with pytest.raises(ValueError, match="Secret not found"):
            get_github_token(required=True)

    def test_get_anthropic_api_key_from_env(self):
        """Should get Anthropic API key from environment variable."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test123"
        result = get_anthropic_api_key(required=False)
        assert result == "sk-ant-test123"

    def test_get_anthropic_api_key_from_aws(self):
        """Should get Anthropic API key from AWS Secrets Manager."""
        # Create mock boto3 module

        mock_boto3 = MagicMock()

        mock_client = MagicMock()

        mock_boto3.client.return_value = mock_client

        with patch("gitta.secrets._has_boto3", return_value=True):
            with patch.dict(sys.modules, {"boto3": mock_boto3}):
                mock_client.get_secret_value.return_value = {"SecretString": "sk-ant-from-aws"}

                result = get_anthropic_api_key(required=False)

                assert result == "sk-ant-from-aws"
                mock_client.get_secret_value.assert_called_with(SecretId="anthropic/api-key")

    def test_get_anthropic_api_key_not_found_optional(self):
        """Should return None when Anthropic API key not found and not required."""
        result = get_anthropic_api_key(required=False)
        assert result is None

    def test_get_anthropic_api_key_not_found_required(self):
        """Should raise error when Anthropic API key not found and required."""
        with pytest.raises(ValueError, match="Secret not found"):
            get_anthropic_api_key(required=True)
