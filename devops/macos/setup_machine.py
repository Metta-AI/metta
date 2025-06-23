#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, check=True):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(command, check=check, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{' '.join(command)}': {e.stderr}")
        sys.exit(1)


def install_homebrew():
    """Install Homebrew if it's not already installed."""
    if not os.path.exists("/opt/homebrew/bin/brew"):
        print("Installing Homebrew...")
        run_command(["curl", "-fsSL", "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"])
        run_command(["/bin/bash", "install.sh"])
        os.remove("install.sh")
    else:
        print("Homebrew is already installed")


def run_brew_bundle(name="Brewfile", force=False, no_fail=False):
    """Run brew bundle with the Brewfile."""
    brewfile_path = Path(__file__).parent / name
    if not brewfile_path.exists():
        print(f"Error: Brewfile not found at {brewfile_path}")
        sys.exit(1)

    print(f"Running brew bundle with {brewfile_path}...")
    command = ["brew", "bundle", "--file", str(brewfile_path)]
    if force:
        command.append("--force")
    if no_fail:
        command.append("--no-fail")

    try:
        run_command(command)
    except subprocess.CalledProcessError:
        if not force and not no_fail:
            print("\nSome packages are already installed but not managed by Homebrew.")
            print("To proceed, run the script with one of these options:")
            print("  --brew-force: Let Homebrew take over existing installations")
            print("  --brew-no-fail: Skip packages that are already installed")
            sys.exit(1)
        raise


def install_skypilot():
    """Install Skypilot."""
    print("Installing Skypilot...")
    run_command(["./devops/skypilot/install.sh"])


def main():
    parser = argparse.ArgumentParser(description="Setup developer machine with Homebrew and required packages")
    parser.add_argument("--brew-force", action="store_true", help="Let Homebrew take over existing installations")
    parser.add_argument("--brew-no-fail", action="store_true", help="Skip packages that are already installed")
    parser.add_argument("--devops", action="store_true", help="Install devops tools")
    args = parser.parse_args()

    install_homebrew()
    run_brew_bundle(force=args.brew_force, no_fail=args.brew_no_fail)
    if args.devops:
        run_brew_bundle(name="Brewfile.devops", force=args.brew_force, no_fail=args.brew_no_fail)
    install_skypilot()

    print("Machine setup complete!")


if __name__ == "__main__":
    main()
