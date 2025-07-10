from unittest.mock import MagicMock, patch

import httpx
import pytest

from app_backend.llm_client import LLMClient


class TestLLMClient:
    """Tests for LLMClient."""

    def test_llm_client_creation(self):
        """Test LLMClient creation with httpx dependency."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client)
            assert llm_client.http_client == http_client

    def test_is_available_no_keys(self):
        """Test is_available returns False when no API keys are present."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client)
            assert not llm_client.is_available()

    def test_is_available_with_openai_key(self):
        """Test is_available returns True when OpenAI key is present."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, openai_api_key="test-openai-key")
            assert llm_client.is_available()

    def test_is_available_with_anthropic_key(self):
        """Test is_available returns True when Anthropic key is present."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, anthropic_api_key="test-anthropic-key")
            assert llm_client.is_available()

    def test_is_available_with_openrouter_key(self):
        """Test is_available returns True when OpenRouter key is present."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, openrouter_api_key="test-openrouter-key")
            assert llm_client.is_available()

    def test_is_available_with_multiple_keys(self):
        """Test is_available returns True when multiple keys are present."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(
                http_client, openai_api_key="test-openai-key", anthropic_api_key="test-anthropic-key"
            )
            assert llm_client.is_available()

    def test_generate_text_with_messages_no_keys(self):
        """Test generate_text_with_messages raises error when no keys are available."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client)

            with pytest.raises(RuntimeError) as exc_info:
                llm_client.generate_text_with_messages([{"role": "user", "content": "test prompt"}])
            assert "No LLM API keys available" in str(exc_info.value)

    def test_generate_text_with_messages_openai_success(self):
        """Test successful text generation with OpenAI."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, openai_api_key="test-key")

            # Mock successful API response
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "Generated text response"}}]}

            with patch.object(http_client, "post", return_value=mock_response):
                result = llm_client.generate_text_with_messages([{"role": "user", "content": "test prompt"}])
                assert result == "Generated text response"

    def test_generate_text_with_messages_anthropic_success(self):
        """Test successful text generation with Anthropic."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, anthropic_api_key="test-key")

            # Mock successful API response
            mock_response = MagicMock()
            mock_response.json.return_value = {"content": [{"text": "Anthropic generated text"}]}

            with patch.object(http_client, "post", return_value=mock_response):
                result = llm_client.generate_text_with_messages([{"role": "user", "content": "test prompt"}])
                assert result == "Anthropic generated text"

    def test_generate_text_with_messages_http_error(self):
        """Test generate_text_with_messages handles HTTP errors properly."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, openai_api_key="test-key")

            # Mock HTTP error
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "Rate limit exceeded"

            with patch.object(
                http_client,
                "post",
                side_effect=httpx.HTTPStatusError("Rate limit", request=None, response=mock_response),
            ):
                with pytest.raises(Exception) as exc_info:
                    llm_client.generate_text_with_messages([{"role": "user", "content": "test prompt"}])
                assert "LLM API request failed: 429" in str(exc_info.value)

    def test_generate_text_with_messages_timeout_error(self):
        """Test generate_text_with_messages handles timeout errors properly."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, openai_api_key="test-key")

            with patch.object(http_client, "post", side_effect=httpx.TimeoutException("Timeout")):
                with pytest.raises(Exception) as exc_info:
                    llm_client.generate_text_with_messages([{"role": "user", "content": "test prompt"}])
                assert "LLM API request timed out" in str(exc_info.value)

    def test_generate_text_with_messages_openrouter_success(self):
        """Test successful text generation with OpenRouter (uses OpenAI format)."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(http_client, openrouter_api_key="test-key")

            # Mock successful API response (OpenAI format)
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "OpenRouter generated text"}}]}

            with patch.object(http_client, "post", return_value=mock_response):
                result = llm_client.generate_text_with_messages([{"role": "user", "content": "test prompt"}])
                assert result == "OpenRouter generated text"

                # Verify the API was called with the correct URL
                http_client.post.assert_called_once()
                call_args = http_client.post.call_args
                assert "openrouter.ai" in call_args[0][0]

    def test_provider_fallback(self):
        """Test that providers are tried in order until one succeeds."""
        with httpx.Client() as http_client:
            llm_client = LLMClient(
                http_client, openai_api_key="test-openai-key", anthropic_api_key="test-anthropic-key"
            )

            # Mock first provider fails, second succeeds
            mock_response = MagicMock()
            mock_response.json.return_value = {"content": [{"text": "Anthropic fallback response"}]}

            # First call raises error, second returns response
            with patch.object(
                http_client,
                "post",
                side_effect=[
                    httpx.HTTPStatusError(
                        "OpenAI failed", request=None, response=MagicMock(status_code=500, text="Error")
                    ),
                    mock_response,
                ],
            ):
                result = llm_client.generate_text_with_messages([{"role": "user", "content": "test prompt"}])
                assert result == "Anthropic fallback response"

                # Verify both providers were tried
                assert http_client.post.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
