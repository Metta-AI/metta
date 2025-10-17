#!/usr/bin/env -S uv run

"""CLI authentication script for CoGames platform."""

import argparse
import sys
from urllib.parse import urlparse

from metta.common.util.auth import BaseCLIAuthenticator

# Default CoGames server URL
DEFAULT_COGAMES_SERVER = "https://beta.softmax.com"


class CoGamesAuthenticator(BaseCLIAuthenticator):
    """CLI Authenticator for CoGames, storing tokens in cogames.yaml under 'login_tokens' key."""

    def __init__(self, auth_server_url: str):
        super().__init__(
            auth_server_url=auth_server_url,
            token_file_name="cogames.yaml",
            token_storage_key="login_tokens",  # Nested under 'login_tokens' key
            extra_uris={},  # No extra URIs for CoGames
        )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Authenticate with CoGames")
    parser.add_argument(
        "auth_server_url",
        nargs="?",
        default=DEFAULT_COGAMES_SERVER,
        help=f"CoGames server URL (default: {DEFAULT_COGAMES_SERVER})",
    )
    parser.add_argument("--force", action="store_true", help="Get a new token even if one already exists")
    parser.add_argument("--timeout", type=int, default=300, help="Authentication timeout in seconds (default: 300)")

    args = parser.parse_args()

    # Create authenticator
    authenticator = CoGamesAuthenticator(auth_server_url=args.auth_server_url)

    # Check if we already have a token
    if authenticator.has_saved_token() and not args.force:
        print(f"Found existing token for {urlparse(args.auth_server_url).hostname}")
        sys.exit(0)

    # Perform authentication
    print(f"Authenticating with {args.auth_server_url}")
    if authenticator.authenticate(timeout=args.timeout):
        print("Authentication successful!")
    else:
        print("Authentication failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
