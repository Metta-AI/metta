#!/usr/bin/env python3
"""Validate that all required secrets and environment variables are configured.

This script checks:
1. All required AWS Secrets Manager secrets exist and are accessible
2. All required environment variables are set
3. Provides clear feedback on what's missing or misconfigured

Usage:
    uv run python devops/datadog/scripts/validate_secrets.py
    uv run python devops/datadog/scripts/validate_secrets.py --verbose
"""

import argparse
import os
import sys
from pathlib import Path


def load_env_file() -> None:
    """Load .env file from devops/datadog/ if it exists."""
    # Try to find .env file
    env_paths = [
        Path("devops/datadog/.env"),  # From repo root
        Path(".env"),  # From devops/datadog/
        Path("../.env"),  # From devops/datadog/scripts/
    ]

    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            # Only set if not already in environment
                            if key.strip() not in os.environ:
                                os.environ[key.strip()] = value.strip()
            return


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def check_aws_secret(secret_name: str, verbose: bool = False) -> tuple[bool, str | None]:
    """Check if an AWS secret exists and can be retrieved.

    Returns:
        Tuple of (success, error_message)
    """
    try:
        from softmax.aws.secrets_manager import get_secretsmanager_secret

        secret_value = get_secretsmanager_secret(secret_name)
        if secret_value:
            if verbose:
                print_success(f"Secret '{secret_name}' found and accessible")
            return True, None
        else:
            return False, "Secret exists but is empty"
    except ImportError:
        return False, "AWS secrets manager module not available (softmax.aws.secrets_manager)"
    except Exception as e:
        return False, str(e)


def check_env_var(var_name: str, required: bool = True, verbose: bool = False) -> tuple[bool, str | None]:
    """Check if an environment variable is set.

    Returns:
        Tuple of (success, value or error_message)
    """
    value = os.getenv(var_name)
    if value:
        if verbose:
            # Mask sensitive values
            display_value = (
                value if var_name.endswith("_GID") or var_name in ["GITHUB_ORG", "GITHUB_REPO", "DD_SITE"] else "***"
            )
            print_success(f"Environment variable '{var_name}' is set: {display_value}")
        return True, value
    else:
        if required:
            return False, "Not set"
        else:
            return False, "Not set (optional)"


def validate_secrets(verbose: bool = False) -> bool:
    """Validate all required secrets are configured.

    Returns:
        True if all validation passes, False otherwise
    """
    all_valid = True

    # Check AWS Secrets Manager secrets
    print_header("Checking AWS Secrets Manager")

    required_secrets = [
        ("datadog/api-key", "Datadog API Key", "https://app.datadoghq.com/organization-settings/api-keys"),
        (
            "datadog/app-key",
            "Datadog Application Key",
            "https://app.datadoghq.com/organization-settings/application-keys",
        ),
        ("github/dashboard-token", "GitHub Personal Access Token", "https://github.com/settings/tokens"),
        ("asana/access-token", "Asana Personal Access Token", "Asana → My Settings → Apps → Personal access tokens"),
        ("asana/workspace-gid", "Asana Workspace GID", "From Asana workspace URL"),
        ("asana/bugs-project-gid", "Asana Bugs Project GID (optional)", "From Asana project URL"),
    ]

    secrets_status = {}
    for secret_id, description, get_from in required_secrets:
        success, error = check_aws_secret(secret_id, verbose)
        secrets_status[secret_id] = success

        if success:
            print_success(f"{description} ({secret_id})")
        else:
            print_error(f"{description} ({secret_id}): {error}")
            print(f"         Get from: {get_from}")
            all_valid = False

    # Check environment variables
    print_header("Checking Environment Variables")

    required_env_vars = [
        ("DD_SITE", False, "Datadog site (defaults to datadoghq.com)"),
        ("GITHUB_ORG", True, "GitHub organization"),
        ("GITHUB_REPO", True, "GitHub repository"),
        ("ASANA_WORKSPACE_GID", False, "Asana workspace ID (or from AWS: asana/workspace-gid)"),
        ("ASANA_BUGS_PROJECT_GID", False, "Asana bugs project ID (or from AWS: asana/bugs-project-gid)"),
    ]

    env_status = {}
    for var_name, required, description in required_env_vars:
        success, result = check_env_var(var_name, required, verbose)
        env_status[var_name] = success

        if success:
            print_success(f"{description} ({var_name})")
        elif required:
            print_error(f"{description} ({var_name}): {result}")
            all_valid = False
        else:
            print_warning(f"{description} ({var_name}): {result}")

    # Check for fallback environment variables (secrets in .env)
    print_header("Checking Environment Variable Fallbacks")

    fallback_vars = [
        ("DD_API_KEY", "datadog/api-key"),
        ("DD_APP_KEY", "datadog/app-key"),
        ("GITHUB_TOKEN", "github/dashboard-token"),
        ("ASANA_ACCESS_TOKEN", "asana/access-token"),
    ]

    print("These environment variables can override AWS Secrets Manager:")
    print("(Useful for local development)\n")

    for var_name, secret_name in fallback_vars:
        env_set = os.getenv(var_name) is not None
        secret_ok = secrets_status.get(secret_name, False)

        if env_set:
            print_success(f"{var_name} is set (will be used instead of {secret_name})")
        elif secret_ok:
            print(f"  {var_name} not set (will use AWS secret: {secret_name})")
        else:
            print_warning(f"{var_name} not set and AWS secret {secret_name} is not accessible")

    # Print summary
    print_header("Validation Summary")

    secrets_ok = sum(secrets_status.values())
    secrets_total = len(secrets_status)
    env_ok = sum(1 for var_name, required, _ in required_env_vars if required and env_status.get(var_name, False))
    env_total = sum(1 for _, required, _ in required_env_vars if required)

    print(f"AWS Secrets Manager: {secrets_ok}/{secrets_total} secrets accessible")
    print(f"Environment Variables: {env_ok}/{env_total} required variables set")

    if all_valid:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All validations passed!{Colors.RESET}")
        print("\nYou can now run collectors:")
        print("  uv run python devops/datadog/scripts/run_collector.py github --verbose")
        print("  uv run python devops/datadog/scripts/run_collector.py skypilot --verbose")
        print("  uv run python devops/datadog/scripts/run_collector.py asana --verbose")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Validation failed!{Colors.RESET}")
        print("\nPlease fix the errors above. See SECRETS_SETUP.md for detailed instructions:")
        print("  cat devops/datadog/SECRETS_SETUP.md")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Datadog collector secrets and environment configuration")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about each check",
    )

    args = parser.parse_args()

    # Load .env file if it exists
    load_env_file()

    print(f"{Colors.BOLD}Datadog Collectors - Secrets Validation{Colors.RESET}")
    print("This script validates your AWS Secrets Manager and environment configuration.\n")

    success = validate_secrets(verbose=args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
