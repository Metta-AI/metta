#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.31.0",
#   "metta-common @ file:///${GITHUB_WORKSPACE}/common",
# ]
# ///
"""Discord webhook posting action script using metta-common utility."""

import os
import sys
from pathlib import Path

from metta.common.util.discord import send_to_discord


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
