#!/usr/bin/env -S uv run

import argparse
import shutil
import subprocess
import sys

from metta.common.util.cli import get_user_confirmation
from metta.common.util.fs import cd_repo_root


def main():
    cd_repo_root()

    parser = argparse.ArgumentParser(description="Build the metta image")
    parser.add_argument("--image-name", default="mettaai/metta:latest")
    args = parser.parse_args()

    if not get_user_confirmation("Images should normally be built by CI. Do you want to proceed?"):
        sys.exit(0)

    if shutil.which("docker") is None:
        print(
            "Docker is not installed!\n\n"
            "To install Docker on macOS:\n"
            "  1. brew install --cask docker   # (recommended)\n"
            "  2. OR download Docker Desktop: https://www.docker.com/products/docker-desktop/\n\n"
            "After installation:\n"
            "  • Open Docker Desktop\n  • Wait for the whale icon\n  • Run this script again",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        subprocess.check_output(["docker", "info"])
    except subprocess.CalledProcessError:
        print("Error: Docker is not installed!")
        print("")
        print("To install Docker on macOS:")
        print("  1. Using Homebrew (recommended):")
        print("     brew install --cask docker")
        print("")
        print("  2. Or download Docker Desktop from:")
        print("     https://www.docker.com/products/docker-desktop/")
        print("")
        print("After installation:")
        print("  1. Open Docker Desktop from your Applications folder")
        print("  2. Wait for Docker to start (you'll see a whale icon in the menu bar)")
        print("  3. Run this script again")
        print("")
        sys.exit(1)

    platform = "linux/amd64"
    print(f"Building {args.image_name} for {platform} (AWS compatible)...")
    subprocess.check_call(
        [
            "docker",
            "build",
            "--platform",
            platform,
            "-f",
            "devops/docker/Dockerfile",
            "-t",
            args.image_name,
            ".",
        ]
    )

    print(f"✓ Successfully built {args.image_name}")


if __name__ == "__main__":
    main()
