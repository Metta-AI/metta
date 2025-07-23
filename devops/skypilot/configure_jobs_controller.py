#!/usr/bin/env -S uv run

"""
This script is used to configure the custom crontab on the jobs controller host.

We need the crontab to clean up some files on the jobs controller host that cause it to run out of disk space.
"""

import shlex
import subprocess
from pathlib import Path

from devops.skypilot.utils import get_jobs_controller_name
from metta.common.util.text_styles import bold, green, yellow


def main():
    controller_name = get_jobs_controller_name()
    print(f"Jobs controller: {yellow(controller_name)}")

    # Read the crontab file
    crontab_file = Path(__file__).parent / "files" / "controller.crontab"
    if not crontab_file.exists():
        raise FileNotFoundError(f"Crontab file not found: {crontab_file}")

    print(f"Reading crontab from: {crontab_file}")
    crontab_content = crontab_file.read_text()

    # Filter out comments and empty lines for display
    active_lines = [
        line for line in crontab_content.strip().split("\n") if line.strip() and not line.strip().startswith("#")
    ]

    print("")
    print(f"Crontab entries to install ({len(active_lines)} jobs):")
    for line in active_lines:
        print(f"  {line}")

    # Install crontab on remote host via SSH
    remote_command = "printf %s | crontab -" % shlex.quote(crontab_content)

    ssh_command = shlex.join(["ssh", controller_name, remote_command])
    print("")
    print(f"Installing crontab with: {bold(ssh_command)}")

    result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(green("✓ Crontab installed successfully"))

        # Verify installation by listing the crontab
        verify_command = shlex.join(["ssh", controller_name, "crontab -l"])
        print(f"Verifying with: {bold(verify_command)}")
        verify_result = subprocess.run(verify_command, shell=True, capture_output=True, text=True)

        if verify_result.returncode == 0:
            print("")
            print("Current crontab on controller:")
            for line in verify_result.stdout.split("\n"):
                print(f"  {line}")
        else:
            print(f"Warning: Could not verify crontab installation: {verify_result.stderr}")
    else:
        print(f"Error installing crontab: {result.stderr}")
        exit(1)


if __name__ == "__main__":
    main()
