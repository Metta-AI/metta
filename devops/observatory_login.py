#!/usr/bin/env -S uv run

# NOTE: when moving this file, make sure to update ObservatoryKeySetup.login_script_location

import argparse
import sys
from urllib.parse import urlparse

from cogames.auth import BaseCLIAuthenticator
from metta.common.util.constants import (
    DEV_STATS_SERVER_URI,
    PROD_OBSERVATORY_FRONTEND_URL,
    PROD_STATS_SERVER_URI,
)

_EXTRA_URIS: dict[str, list[str]] = {
    f"{PROD_OBSERVATORY_FRONTEND_URL}/api": [PROD_STATS_SERVER_URI],
}


class CLIAuthenticator(BaseCLIAuthenticator):
    """CLI Authenticator for Observatory, storing tokens in observatory_tokens.yaml."""

    def __init__(self, auth_server_url: str):
        super().__init__(
            auth_server_url=auth_server_url,
            token_file_name="observatory_tokens.yaml",
            token_storage_key=None,  # Top-level storage
            extra_uris=_EXTRA_URIS,
        )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Authenticate with Observatory")
    parser.add_argument(
        "auth_server_url",
        help=f"Stats server API URI (e.g., {PROD_STATS_SERVER_URI} or {DEV_STATS_SERVER_URI})",
    )
    parser.add_argument("--force", action="store_true", help="Get a new token even if one already exists")
    parser.add_argument("--timeout", type=int, default=300, help="Authentication timeout in seconds (default: 300)")

    args = parser.parse_args()

    # Create authenticator
    authenticator = CLIAuthenticator(auth_server_url=args.auth_server_url)

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
