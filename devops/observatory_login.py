#!/usr/bin/env -S uv run
# ruff: noqa: E501

# NOTE: when moving this file, make sure to update ObservatoryKeySetup.login_script_location

import argparse
import sys
import urllib.parse

import cogames.auth
import metta.common.util.constants


class CLIAuthenticator(cogames.auth.BaseCLIAuthenticator):
    """CLI Authenticator for Observatory, storing tokens in observatory_tokens.yaml."""

    def __init__(self):
        super().__init__(
            token_file_name="config.yaml",
            token_storage_key="observatory_tokens",
        )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Authenticate with Observatory")
    parser.add_argument(
        "auth_server_url",
        help=(
            "Stats server API URI (e.g., "
            f"{metta.common.util.constants.OBSERVATORY_AUTH_SERVER_URL} or "
            f"{metta.common.util.constants.DEV_STATS_SERVER_URI})"
        ),
    )
    parser.add_argument(
        "token_key",
        help=(
            f"Key to store the token under in the YAML file (e.g. {metta.common.util.constants.PROD_STATS_SERVER_URI})"
        ),
    )
    parser.add_argument("--force", action="store_true", help="Get a new token even if one already exists")
    parser.add_argument("--timeout", type=int, default=300, help="Authentication timeout in seconds (default: 300)")

    args = parser.parse_args()

    # Create authenticator
    authenticator = CLIAuthenticator()

    # Check if we already have a token
    if authenticator.has_saved_token(args.token_key) and not args.force:
        print(f"Found existing token for {urllib.parse.urlparse(args.auth_server_url).hostname}")
        sys.exit(0)

    # Perform authentication
    print(f"Authenticating with {args.auth_server_url}")
    if authenticator.authenticate(auth_server_url=args.auth_server_url, token_key=args.token_key, timeout=args.timeout):
        print("Authentication successful!")
    else:
        print("Authentication failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
