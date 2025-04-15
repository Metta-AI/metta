#!/usr/bin/env python3
import argparse
import getpass
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_status(message):
    """Print a status message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Colors.BLUE}[{timestamp}] STATUS: {message}{Colors.ENDC}")


def print_success(message):
    """Print a success message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Colors.GREEN}[{timestamp}] SUCCESS: {message}{Colors.ENDC}")


def print_error(message):
    """Print an error message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Colors.RED}[{timestamp}] ERROR: {message}{Colors.ENDC}", file=sys.stderr)


def print_warning(message):
    """Print a warning message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"{Colors.YELLOW}[{timestamp}] WARNING: {message}{Colors.ENDC}")


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and handle errors."""
    print_status(f"Running command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=capture_output, text=True)
        if result.returncode == 0:
            print_status("Command completed successfully")
        else:
            print_warning(f"Command exited with code {result.returncode}")

        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Error executing command: {cmd}")
        print_error(f"Error message: {e}")
        if capture_output and e.stderr:
            print_error(f"Error details: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def setup_root_profile():
    """Set up AWS configuration for root account under stem-root profile."""
    print_status("Setting up AWS configuration for root account...")

    # Prompt for AWS access key and secret
    print_status("Please enter your AWS root account credentials:")
    aws_access_key_id = input("AWS Access Key ID: ")
    aws_secret_access_key = getpass.getpass("AWS Secret Access Key: ")

    if not aws_access_key_id or not aws_secret_access_key:
        print_error("AWS credentials cannot be empty")
        sys.exit(1)

    # Create AWS credentials file
    aws_dir = Path.home() / ".aws"
    aws_dir.mkdir(exist_ok=True)

    credentials_path = aws_dir / "credentials"

    # Read existing credentials if file exists
    print_status(f"Checking for existing credentials file at {credentials_path}")
    existing_credentials = {}
    if credentials_path.exists():
        print_status("Existing credentials file found, reading contents...")
        try:
            with open(credentials_path, "r") as f:
                current_profile = None
                for line in f:
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        current_profile = line[1:-1]
                        existing_credentials[current_profile] = []
                    elif current_profile:
                        existing_credentials[current_profile].append(line)
            print_status(f"Found {len(existing_credentials)} existing profile(s)")
        except Exception as e:
            print_error(f"Error reading credentials file: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Add or update stem-root profile
    print_status("Updating credentials file with stem-root profile...")
    try:
        with open(credentials_path, "w") as f:
            for profile, lines in existing_credentials.items():
                if profile != "stem-root":  # Skip stem-root as we'll rewrite it
                    f.write(f"[{profile}]\n")
                    for line in lines:
                        f.write(f"{line}\n")

            # Write stem-root profile
            f.write("[stem-root]\n")
            f.write(f"aws_access_key_id = {aws_access_key_id}\n")
            f.write(f"aws_secret_access_key = {aws_secret_access_key}\n")
            f.write("region = us-east-1\n")
        print_success("Successfully updated credentials file")
    except Exception as e:
        print_error(f"Error writing to credentials file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Update AWS config file
    config_path = aws_dir / "config"

    # Read existing config if file exists
    print_status(f"Checking for existing config file at {config_path}")
    existing_config = {}
    if config_path.exists():
        print_status("Existing config file found, reading contents...")
        try:
            with open(config_path, "r") as f:
                current_section = None
                for line in f:
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1]
                        existing_config[current_section] = []
                    elif current_section:
                        existing_config[current_section].append(line)
            print_status(f"Found {len(existing_config)} existing section(s)")
        except Exception as e:
            print_error(f"Error reading config file: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Add or update stem-root profile in config
    print_status("Updating config file with stem-root profile...")
    try:
        with open(config_path, "w") as f:
            for section, lines in existing_config.items():
                if section != "stem-root" and section != "profile stem-root":  # Skip as we'll rewrite it
                    f.write(f"[{section}]\n")
                    for line in lines:
                        f.write(f"{line}\n")

            # Write stem-root profile
            f.write("[profile stem-root]\n")
            f.write("region = us-east-1\n")
            f.write("output = json\n")
        print_success("Successfully updated config file")
    except Exception as e:
        print_error(f"Error writing to config file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Test the configuration
    print_status("Testing AWS access with root credentials...")
    result = run_command("aws s3 ls --profile stem-root", check=False)
    if result.returncode != 0:
        print_warning("Could not verify AWS access with the stem-root profile.")
        print_warning("Please check your credentials and try again.")
    else:
        print_success("Successfully verified AWS access with stem-root profile.")

    # Create a helper script to open the AWS console
    print_status("Creating AWS console helper script...")
    console_script = aws_dir / "console.sh"
    try:
        with open(console_script, "w") as f:
            f.write("""#!/bin/bash
# Helper script to open the AWS console in the browser
aws_signin_url="https://signin.aws.amazon.com/console"
open "$aws_signin_url" 2>/dev/null || \
xdg-open "$aws_signin_url" 2>/dev/null || \
echo "Could not open browser. Please visit $aws_signin_url manually."
""")
        print_success("Successfully created console helper script")
    except Exception as e:
        print_error(f"Error creating console script: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Make the script executable
    try:
        os.chmod(console_script, 0o755)
        print_status("Made console script executable")
    except Exception as e:
        print_error(f"Error making console script executable: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Add an alias for the console command to shell config files
    print_status("Adding aws-console alias to shell configuration files...")
    for rc_file in [".bashrc", ".zshrc"]:
        rc_path = Path.home() / rc_file
        if rc_path.exists():
            try:
                with open(rc_path, "r") as f:
                    content = f.read()

                if "alias aws-console=" not in content:
                    with open(rc_path, "a") as f:
                        f.write(f"\nalias aws-console='{console_script}'\n")
                    print_status(f"Added aws-console alias to {rc_file}")
                else:
                    print_status(f"aws-console alias already exists in {rc_file}")
            except Exception as e:
                print_error(f"Error updating {rc_file}: {e}")
                traceback.print_exc()

    # Set AWS_PROFILE in the current shell
    os.environ["AWS_PROFILE"] = "stem-root"
    print_status("Set AWS_PROFILE=stem-root in current shell")

    # Update shell config files to use stem-root profile
    print_status("Updating shell configuration files to use stem-root profile...")
    for rc_file in [".bashrc", ".zshrc"]:
        rc_path = Path.home() / rc_file
        if rc_path.exists():
            try:
                with open(rc_path, "r") as f:
                    content = f.read()

                # Remove any existing AWS_PROFILE export
                lines = content.split("\n")
                new_lines = [line for line in lines if not line.strip().startswith("export AWS_PROFILE=")]

                # Add the new export
                new_lines.append("export AWS_PROFILE=stem-root")

                with open(rc_path, "w") as f:
                    f.write("\n".join(new_lines))

                print_status(f"Updated AWS_PROFILE in {rc_file} to use stem-root")
            except Exception as e:
                print_error(f"Error updating {rc_file}: {e}")
                traceback.print_exc()

    print_success("Root account setup complete!")
    print_status("You can open the AWS Console in your browser by typing 'aws-console'")


def setup_sso_profile():
    """Set up AWS configuration for SSO access."""
    print_status("Setting up AWS configuration for SSO access...")

    # Create AWS directory if it doesn't exist
    aws_dir = Path.home() / ".aws"
    try:
        aws_dir.mkdir(exist_ok=True)
        print_status(f"Created AWS config directory at {aws_dir}")
    except Exception as e:
        print_error(f"Failed to create AWS config directory: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Create AWS config file with SSO configuration
    config_path = aws_dir / "config"

    # Read existing config if file exists to preserve other profiles
    print_status(f"Checking for existing config file at {config_path}")
    existing_config = {}
    if config_path.exists():
        print_status("Existing config file found, reading contents...")
        try:
            with open(config_path, "r") as f:
                current_section = None
                for line in f:
                    line = line.strip()
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1]
                        existing_config[current_section] = []
                    elif current_section:
                        existing_config[current_section].append(line)
            print_status(f"Found {len(existing_config)} existing section(s)")
        except Exception as e:
            print_error(f"Error reading config file: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Add or update SSO profiles
    print_status("Updating config file with SSO profiles...")
    try:
        with open(config_path, "w") as f:
            # Preserve existing profiles except the ones we're updating
            for section, lines in existing_config.items():
                if section not in ["stem", "profile stem", "sso-session stem-sso"]:
                    f.write(f"[{section}]\n")
                    for line in lines:
                        f.write(f"{line}\n")

            # Write stem profile
            f.write("""[stem]
region = us-east-1
output = json

[profile stem]
sso_session = stem-sso
sso_account_id = 767406518141
sso_role_name = PowerUserAccess
region = us-east-1

[sso-session stem-sso]
sso_start_url = https://softmaxx.awsapps.com/start/
sso_region = us-east-1
sso_registration_scopes = sso:account:access
""")
        print_success("Successfully updated config file with SSO profiles")
    except Exception as e:
        print_error(f"Error writing to config file: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Log in to AWS SSO
    print_status("Logging in to AWS SSO...")
    print_status("This will open a browser window. Please complete the login process there.")
    result = run_command("aws sso login --profile stem", check=False)
    if result.returncode != 0:
        print_error("Could not log in to AWS SSO.")
        print_error("Please check your network connection and try again.")
        sys.exit(1)

    # Verify the profile works
    print_status("Testing AWS access with stem profile...")
    result = run_command("aws s3 ls --profile stem", check=False)
    if result.returncode != 0:
        print_error("Could not access AWS with the stem profile.")
        print_error("Please check your SSO configuration and try again.")
        sys.exit(1)
    else:
        print_success("Successfully verified AWS access with stem profile.")

    # Set AWS_PROFILE in the current shell
    os.environ["AWS_PROFILE"] = "stem"
    print_status("Set AWS_PROFILE=stem in current shell")

    # Add profile to shell config files
    print_status("Updating shell configuration files...")
    for rc_file in [".bashrc", ".zshrc"]:
        rc_path = Path.home() / rc_file
        if rc_path.exists():
            try:
                with open(rc_path, "r") as f:
                    content = f.read()

                # Remove any existing AWS_PROFILE export
                lines = content.split("\n")
                new_lines = [line for line in lines if not line.strip().startswith("export AWS_PROFILE=")]

                # Add the new export
                new_lines.append("export AWS_PROFILE=stem")

                with open(rc_path, "w") as f:
                    f.write("\n".join(new_lines))

                print_status(f"Updated AWS_PROFILE in {rc_file}")
            except Exception as e:
                print_error(f"Error updating {rc_file}: {e}")
                traceback.print_exc()

    print_success("SSO setup complete!")


def main():
    try:
        parser = argparse.ArgumentParser(description="Set up AWS SSO or root account access")
        parser.add_argument("--root", action="store_true", help="Set up root account access only (skips SSO)")
        args = parser.parse_args()

        print_status("Starting AWS setup script (Python version)")

        if args.root:
            print_status("Mode: Root Account Only")
            print_status("Root account setup requested, skipping SSO setup")
            setup_root_profile()
        else:
            print_status("Mode: SSO Only")
            setup_sso_profile()
            print_status("Root account setup not requested (use --root flag to set up root credentials)")

        print_success("AWS setup completed successfully!")
        print_status("IMPORTANT: To apply changes immediately, run: source ~/.bashrc (or source ~/.zshrc)")
    except KeyboardInterrupt:
        print_error("\nSetup interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
