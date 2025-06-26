#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-generativeai>=0.3.0",
# ]
# ///

import logging
import time
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

# Model configuration - three tiers for different use cases
MODEL_CONFIG = {
    "fast": "gemini-2.5-flash-lite-preview-06-17",  # Most cost-efficient, high throughput
    "default": "gemini-2.5-flash",  # Balanced cost and performance
    "best": "gemini-2.5-pro",  # Enhanced reasoning and complex analysis
}


class GeminiAIClient:
    """AI client optimized for Gemini 2.5 with rate limiting and retry logic."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rate_limit_delay = 0.3  # Fast model, minimal delay
        self.max_retries = 3
        self.last_request_time = 0

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Minimal safety settings for code analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        logging.info("Initialized AI client with Gemini 2.5 models")

    def _get_model(self, tier: str = "default"):
        """Get the appropriate model for the specified tier."""
        model_name = MODEL_CONFIG.get(tier, MODEL_CONFIG["default"])

        # Adjust token limits based on model tier
        max_tokens = {
            "fast": 4000,  # Lightweight analysis
            "default": 5000,  # Standard analysis
            "best": 10000,  # Comprehensive analysis
        }.get(tier, 5000)

        generation_config = genai.GenerationConfig(
            temperature=0.2,  # Consistent technical analysis
            top_p=0.8,
            top_k=40,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain",
        )

        return genai.GenerativeModel(
            model_name=model_name, generation_config=generation_config, safety_settings=self.safety_settings
        )

    def _wait_for_rate_limit(self):
        """Simple rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def generate_with_retry(self, prompt: str, tier: str = "default") -> Optional[str]:
        """Generate content with retry logic.

        Args:
            prompt: The input prompt
            tier: Model tier - "fast", "default", or "best"
        """
        model = self._get_model(tier)

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                response = model.generate_content(prompt)

                if response.text:
                    logging.info(f"AI generation successful on attempt {attempt + 1} ({MODEL_CONFIG[tier]})")
                    return response.text.strip()
                else:
                    logging.warning(f"Empty response on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2**attempt)
                        continue

            except Exception as e:
                logging.error(f"AI generation error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    logging.error("Max retries exceeded for AI generation")
                    return None

        return None
