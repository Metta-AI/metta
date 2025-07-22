#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
# ]
# ///
"""Discord webhook posting utility with automatic message splitting."""

import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

DISCORD_MESSAGE_CHARACTER_LIMIT = 2000
RATE_LIMIT_DELAY = 0.5  # Delay between messages to avoid rate limiting
MESSAGE_PREFIX = "...\r\n   \r\n"  # Prefix added to each chunk


def split_content(content: str, max_len: int = DISCORD_MESSAGE_CHARACTER_LIMIT) -> list[str]:
    """Split content into chunks that fit Discord's message limit.

    Attempts to split at natural boundaries (paragraphs, lines, words) to maintain readability.
    """
    if not content:
        return []

    chunks: list[str] = []
    remaining_content = content.strip()

    while remaining_content:
        if len(remaining_content) <= max_len:
            chunks.append(remaining_content)
            break

        # Try to find a double newline (paragraph break) to split
        split_at = remaining_content.rfind("\n\n", 0, max_len)

        if split_at != -1:
            # Split after the double newline
            chunk = remaining_content[: split_at + 2]
            remaining_content = remaining_content[split_at + 2 :]
        else:
            # No double newline, try to find a single newline
            split_at = remaining_content.rfind("\n", 0, max_len)
            if split_at != -1:
                # Split after the single newline
                chunk = remaining_content[: split_at + 1]
                remaining_content = remaining_content[split_at + 1 :]
            else:
                # No newline found, try to split at last space
                split_at = remaining_content.rfind(" ", 0, max_len)
                if split_at != -1:
                    chunk = remaining_content[:split_at]
                    remaining_content = remaining_content[split_at + 1 :]
                else:
                    # Hard split (worst case)
                    chunk = remaining_content[:max_len]
                    remaining_content = remaining_content[max_len:]

        chunks.append(chunk.strip())

    return [chunk for chunk in chunks if chunk]  # Filter out empty chunks


def sanitize_discord_content(content: str) -> str:
    """Sanitize content for Discord to prevent injection attacks."""
    # Remove @everyone and @here mentions
    content = content.replace("@everyone", "@\u200beveryone")
    content = content.replace("@here", "@\u200bhere")

    # Also handle variations with extra spaces
    content = content.replace("@ everyone", "@ \u200beveryone")
    content = content.replace("@ here", "@ \u200bhere")

    return content


def send_to_discord(webhook_url: str, content: str, suppress_embeds: bool = True) -> bool:
    """Send content to Discord webhook, handling message splitting.

    Args:
        webhook_url: Discord webhook URL
        content: Content to post
        suppress_embeds: Whether to suppress link embeds

    Returns:
        True if all messages sent successfully, False otherwise
    """
    # Sanitize content before splitting
    content = sanitize_discord_content(content)

    # Calculate effective max length accounting for prefix
    effective_max_len = DISCORD_MESSAGE_CHARACTER_LIMIT - len(MESSAGE_PREFIX)

    chunks = split_content(content, max_len=effective_max_len)

    if not chunks:
        print("No content to send to Discord.")
        return True

    print(f"Splitting message into {len(chunks)} chunk(s)...")

    for i, chunk in enumerate(chunks):
        # Prefix each chunk
        prefixed_chunk = MESSAGE_PREFIX + chunk

        # Safety check: ensure we don't exceed Discord's limit
        if len(prefixed_chunk) > DISCORD_MESSAGE_CHARACTER_LIMIT:
            print(f"Warning: Chunk {i + 1} exceeds Discord limit after prefix. Truncating...", file=sys.stderr)
            # Truncate to fit
            max_chunk_len = DISCORD_MESSAGE_CHARACTER_LIMIT - len(MESSAGE_PREFIX)
            chunk = chunk[:max_chunk_len]
            prefixed_chunk = MESSAGE_PREFIX + chunk

        payload: dict[str, Any] = {"content": prefixed_chunk}
        if suppress_embeds:
            payload["flags"] = 4  # SUPPRESS_EMBEDS flag

        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"Successfully sent chunk {i + 1}/{len(chunks)} to Discord.")

            # Rate limit protection
            if i < len(chunks) - 1:
                time.sleep(RATE_LIMIT_DELAY)

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error sending message to Discord: {e}", file=sys.stderr)
            if e.response is not None:
                print(f"Discord API response: {e.response.text}", file=sys.stderr)
                # Check for rate limiting
                if e.response.status_code == 429:
                    retry_after = e.response.json().get("retry_after", 60)
                    print(f"Rate limited. Retry after {retry_after} seconds.", file=sys.stderr)
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error sending message to Discord: {e}", file=sys.stderr)
            return False

    return True


def get_content() -> str:
    """Get content from either direct input or file."""
    # Try direct content first
    content = os.getenv("DISCORD_CONTENT")
    if content:
        return content

    # Try content file
    content_file = os.getenv("DISCORD_CONTENT_FILE")
    if content_file:
        content_path = Path(content_file)
        if content_path.exists():
            try:
                return content_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Error reading content file '{content_file}': {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: Content file '{content_file}' does not exist", file=sys.stderr)
            sys.exit(1)

    print("Error: Neither DISCORD_CONTENT nor DISCORD_CONTENT_FILE provided", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """Main entry point for the Discord posting action."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    suppress_embeds = os.getenv("DISCORD_SUPPRESS_EMBEDS", "true").lower() == "true"

    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL not provided", file=sys.stderr)
        sys.exit(1)

    # Get content from either direct input or file
    content = get_content()

    if not content.strip():
        print("Error: Content is empty", file=sys.stderr)
        sys.exit(1)

    # Validate webhook URL format
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        print("Warning: Webhook URL doesn't match expected Discord format", file=sys.stderr)

    success = send_to_discord(webhook_url, content, suppress_embeds)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
