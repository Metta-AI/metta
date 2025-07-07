#!/usr/bin/env -S uv run

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("llm_client")


@dataclass
class ModelPricing:
    """Pricing information for a single model."""

    prompt_cost_per_1k: float
    completion_cost_per_1k: float

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate total cost based on token usage.

        Args:
            prompt_tokens: Number of prompt/input tokens
            completion_tokens: Number of completion/output tokens

        Returns:
            Total cost in USD
        """
        prompt_cost = (prompt_tokens / 1000) * self.prompt_cost_per_1k
        completion_cost = (completion_tokens / 1000) * self.completion_cost_per_1k
        return prompt_cost + completion_cost


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, api_key: str, api_url: str, default_model: str):
        self.api_key = api_key
        self.api_url = api_url
        self.default_model = default_model

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API request."""
        pass

    @abstractmethod
    def encode_request(self, model: str, messages: List[Dict[str, str]], opts: Dict[str, Any]) -> Dict[str, Any]:
        """Encode request body for API."""
        pass

    @abstractmethod
    def decode_response(self, response_data: Dict[str, Any]) -> str:
        """Decode API response to get text."""
        pass

    @abstractmethod
    def extract_usage_info(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage and cost information from API response."""
        pass

    @abstractmethod
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate cost based on model and token usage.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD, or None if model pricing not available
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    # Pricing per 1K tokens (as of 2025)
    # Note: OpenAI now prices per 1M tokens, converted here to per 1K for consistency
    PRICING = {
        # GPT-4.1 series (latest)
        "gpt-4.1": ModelPricing(0.005, 0.02),  # Estimated based on performance improvements
        "gpt-4.1-mini": ModelPricing(0.00015, 0.0006),  # 83% cheaper than GPT-4o
        "gpt-4.1-nano": ModelPricing(0.0001, 0.0004),  # Estimated
        # GPT-4o series
        "gpt-4o": ModelPricing(0.005, 0.02),  # $5/1M input, $20/1M output
        "gpt-4o-mini": ModelPricing(0.00015, 0.0006),  # $0.15/1M input, $0.60/1M output
        "gpt-4o-audio": ModelPricing(0.1, 0.2),  # $100/1M input, $200/1M output
        # GPT-4 series
        "gpt-4-turbo": ModelPricing(0.01, 0.03),  # $10/1M input, $30/1M output
        "gpt-4-turbo-128k": ModelPricing(0.01, 0.03),
        "gpt-4": ModelPricing(0.03, 0.06),  # $30/1M input, $60/1M output
        "gpt-4-32k": ModelPricing(0.06, 0.12),  # $60/1M input, $120/1M output
        "gpt-4.5-preview": ModelPricing(0.075, 0.15),  # $75/1M input, $150/1M output (deprecated July 2025)
        # GPT-3.5 series
        "gpt-3.5-turbo": ModelPricing(0.0015, 0.002),
        "gpt-3.5-turbo-0125": ModelPricing(0.0005, 0.0015),
    }

    def get_headers(self) -> Dict[str, str]:
        return {"authorization": f"bearer {self.api_key}"}

    def encode_request(self, model: str, messages: List[Dict[str, str]], opts: Dict[str, Any]) -> Dict[str, Any]:
        return {"model": model, "messages": messages, **opts}

    def decode_response(self, response_data: Dict[str, Any]) -> str:
        return response_data["choices"][0]["message"]["content"]

    def extract_usage_info(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage and cost information from OpenAI API response."""
        usage_info = {}

        # Extract token usage
        usage = response_data.get("usage", {})
        if usage:
            usage_info.update(
                {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            )

        # OpenAI now often includes cost directly in the response
        if "cost" in response_data:
            usage_info["cost_usd"] = response_data["cost"]
        elif usage and "prompt_tokens_details" in usage:
            # Some OpenAI responses include detailed cost breakdown
            details = usage.get("prompt_tokens_details", {})
            if "cost" in details:
                usage_info["cost_usd"] = details["cost"]

        return usage_info

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate cost for OpenAI models."""
        # Find matching model pricing
        pricing = None
        model_lower = model.lower()

        # Direct match first
        if model_lower in self.PRICING:
            pricing = self.PRICING[model_lower]
        else:
            # Partial match for model variations
            for known_model, model_pricing in self.PRICING.items():
                if known_model in model_lower:
                    pricing = model_pricing
                    break

        if not pricing:
            logger.debug(f"No pricing found for OpenAI model: {model}")
            return None

        return pricing.calculate_cost(prompt_tokens, completion_tokens)


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter API provider (uses OpenAI-compatible format but different pricing)."""

    # OpenRouter has dynamic pricing based on the model
    # These are example prices - OpenRouter pricing varies by model
    PRICING = {
        "openrouter/auto": ModelPricing(0.0, 0.0),  # Auto-routing, price varies
        "gpt-4": ModelPricing(0.03, 0.06),
        "gpt-3.5-turbo": ModelPricing(0.001, 0.002),
        "claude-3-opus": ModelPricing(0.015, 0.075),
        "claude-3-sonnet": ModelPricing(0.003, 0.015),
        "claude-3-haiku": ModelPricing(0.00025, 0.00125),
        "llama-3-70b": ModelPricing(0.0007, 0.0009),
        "mixtral-8x7b": ModelPricing(0.0003, 0.0003),
    }

    def extract_usage_info(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage and cost information from OpenRouter API response."""
        usage_info = super().extract_usage_info(response_data)

        # OpenRouter often includes direct cost information
        if "cost" in response_data:
            usage_info["cost_usd"] = response_data["cost"]

        # OpenRouter may include model info in response
        if "model" in response_data:
            usage_info["model"] = response_data["model"]

        return usage_info


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    # Pricing per 1K tokens (as of 2025)
    PRICING = {
        # Claude 4 series (latest)
        "claude-opus-4": ModelPricing(0.015, 0.075),  # $15/1M input, $75/1M output
        "claude-sonnet-4": ModelPricing(0.003, 0.015),  # $3/1M input, $15/1M output
        # Claude 3.5 series
        "claude-3-5-sonnet-20241022": ModelPricing(0.003, 0.015),
        "claude-3-5-sonnet": ModelPricing(0.003, 0.015),
        "claude-3-7-sonnet": ModelPricing(0.003, 0.015),
        "claude-3-5-haiku": ModelPricing(0.00025, 0.00125),  # Pricing estimated
        # Claude 3 series
        "claude-3-opus-20240229": ModelPricing(0.015, 0.075),
        "claude-3-opus": ModelPricing(0.015, 0.075),
        "claude-3-sonnet-20240229": ModelPricing(0.003, 0.015),
        "claude-3-haiku-20240307": ModelPricing(0.00025, 0.00125),
        # Claude 2 series (legacy)
        "claude-2.1": ModelPricing(0.008, 0.024),
        "claude-2.0": ModelPricing(0.008, 0.024),
        "claude-instant-1.2": ModelPricing(0.0008, 0.0024),
    }

    def get_headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def encode_request(self, model: str, messages: List[Dict[str, str]], opts: Dict[str, Any]) -> Dict[str, Any]:
        # Anthropic requires max_tokens
        if "max_tokens" not in opts:
            opts["max_tokens"] = 1024

        # Extract system messages - Anthropic expects system as a top-level parameter
        system_content = None
        filtered_messages = []

        for message in messages:
            if message["role"] == "system":
                # Take the first system message content (combine multiple if needed)
                if system_content is None:
                    system_content = message["content"]
                else:
                    # If multiple system messages, combine them
                    system_content += "\n\n" + message["content"]
            else:
                filtered_messages.append(message)

        # Build request body
        request_body = {"model": model, "messages": filtered_messages, **opts}

        # Add system parameter if we found system messages
        if system_content is not None:
            request_body["system"] = system_content

        return request_body

    def decode_response(self, response_data: Dict[str, Any]) -> str:
        return response_data["content"][0]["text"]

    def extract_usage_info(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage and cost information from Anthropic API response."""
        usage_info = {}

        # Anthropic usage is in a different format
        usage = response_data.get("usage", {})
        if usage:
            usage_info.update(
                {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                }
            )

        return usage_info

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        """Calculate cost for Anthropic models."""
        # Find matching model pricing
        pricing = None
        model_lower = model.lower()

        # Direct match first
        if model in self.PRICING:
            pricing = self.PRICING[model]
        else:
            # Partial match for model variations
            for known_model, model_pricing in self.PRICING.items():
                if known_model in model_lower or model_lower in known_model:
                    pricing = model_pricing
                    break

        if not pricing:
            logger.debug(f"No pricing found for Anthropic model: {model}")
            return None

        return pricing.calculate_cost(prompt_tokens, completion_tokens)


class LLMClient:
    """
    Client for LLM API operations with OpenAI, OpenRouter, and Anthropic support.
    Follows the existing client pattern with dependency injection.
    """

    def __init__(
        self,
        http_client: httpx.Client,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        openrouter_api_base: Optional[str] = None,
        anthropic_api_url: Optional[str] = None,
        openai_model: Optional[str] = None,
        anthropic_model: Optional[str] = None,
        openrouter_model: Optional[str] = None,
    ):
        """
        Initialize the LLM client with configuration.

        Args:
            http_client: HTTP client for making API requests
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            openrouter_api_key: OpenRouter API key
            openai_api_base: OpenAI API endpoint
            openrouter_api_base: OpenRouter API endpoint
            anthropic_api_url: Anthropic API endpoint
            openai_model: Default model for OpenAI provider
            anthropic_model: Default model for Anthropic provider
            openrouter_model: Default model for OpenRouter provider
        """
        from app_backend import config

        self.http_client = http_client

        # Initialize providers with config defaults
        self.providers: List[LLMProvider] = []

        if openai_api_key:
            api_base = openai_api_base or config.openai_api_base
            model = openai_model or config.openai_model
            self.providers.append(OpenAIProvider(openai_api_key, api_base, model))

        if anthropic_api_key:
            api_url = anthropic_api_url or config.anthropic_api_url
            model = anthropic_model or config.anthropic_model
            self.providers.append(AnthropicProvider(anthropic_api_key, api_url, model))

        if openrouter_api_key:
            # OpenRouter uses OpenAI-compatible API but different pricing
            api_base = openrouter_api_base or config.openrouter_api_base
            model = openrouter_model or config.openrouter_model
            self.providers.append(OpenRouterProvider(openrouter_api_key, api_base, model))

    def is_available(self) -> bool:
        """
        Check if LLM client has at least one API key available.

        Returns:
            True if at least one API key is available, False otherwise
        """
        return len(self.providers) > 0

    def generate_text_with_messages(self, messages: List[Dict[str, str]], **opts) -> str:
        """
        Generate text using available LLM API with multi-message conversation support.

        Args:
            messages: List of messages in conversation format [{"role": "user/assistant/system", "content": "..."}]
            **opts: Additional options (model, max_tokens, temperature, etc.)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If no API keys are available
            Exception: For API call failures
        """
        if not self.is_available():
            raise RuntimeError("No LLM API keys available")

        # Try providers in order until one succeeds
        last_error = None
        logger.debug(f"Attempting LLM generation with {len(self.providers)} available provider(s)")

        for i, provider in enumerate(self.providers):
            provider_name = provider.__class__.__name__
            logger.debug(f"Trying provider {i + 1}/{len(self.providers)}: {provider_name} at {provider.api_url}")
            try:
                # Prepare request
                model = opts.pop("model", provider.default_model)
                body = provider.encode_request(model, messages, opts)
                headers = provider.get_headers()

                # Set timeout
                timeout = opts.get("timeout", 60)

                logger.debug(f"Making LLM API request to {provider.api_url} with model {model}")
                logger.debug(f"=== LLM REQUEST ({len(messages)} messages) ===")
                for i, msg in enumerate(messages):
                    logger.debug(f"Message {i + 1} [{msg['role']}]:")
                    logger.debug(msg["content"])
                    logger.debug("---")

                # Make API request
                response = self.http_client.post(provider.api_url, headers=headers, json=body, timeout=timeout)
                response.raise_for_status()

                # Decode response
                response_json = response.json()
                result = provider.decode_response(response_json)

                # Log usage and cost information if available
                self._log_usage_info(provider, model, response_json)

                logger.debug(f"=== LLM RESPONSE ({len(result)} chars) ===")
                logger.debug(result)

                return result

            except httpx.HTTPStatusError as e:
                error_text = (
                    f"Provider {provider_name} failed with HTTP error: {e.response.status_code} - {e.response.text}"
                )
                logger.error(error_text)
                last_error = RuntimeError(error_text)
                continue
            except httpx.TimeoutException as e:
                error_text = f"Provider {provider_name} failed with timeout: {e}"
                logger.error(error_text)
                last_error = RuntimeError(error_text)
                continue
            except KeyError as e:
                error_text = f"Provider {provider_name} failed with response parsing error: {e}"
                logger.error(error_text)
                last_error = RuntimeError(error_text)
                continue
            except Exception as e:
                error_text = f"Provider {provider_name} failed with error: {e}"
                logger.error(error_text)
                last_error = RuntimeError(error_text)
                continue

        # If we get here, all providers failed
        if last_error:
            raise last_error
        else:
            raise Exception("All LLM providers failed")

    def _log_usage_info(self, provider: LLMProvider, model: str, response_json: Dict[str, Any]) -> None:
        """
        Log usage and cost information from LLM API response.

        Args:
            provider: The provider instance used for the request
            model: Model used for the request
            response_json: Full API response JSON
        """
        try:
            provider_name = provider.__class__.__name__
            usage_info = provider.extract_usage_info(response_json)

            if not usage_info:
                return

            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            total_tokens = usage_info.get("total_tokens", 0)

            logger.info(
                f"LLM API Usage - Provider: {provider_name}, Model: {model}, "
                f"Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total"
            )

            # Use direct cost if available, otherwise calculate using provider pricing
            if "cost_usd" in usage_info:
                logger.info(f"LLM API Cost: ${usage_info['cost_usd']:.6f} USD")
            else:
                cost_estimate = provider.calculate_cost(model, prompt_tokens, completion_tokens)
                if cost_estimate:
                    logger.info(f"LLM API Cost Estimate: ~${cost_estimate:.6f} USD")

        except Exception as e:
            logger.warning(f"Could not extract usage info from response: {e}")
