#!/usr/bin/env python3
"""
Example of how skypilot launch.py can use the unified configuration.

This shows how to:
1. Load configuration from ~/.metta/config.yaml
2. Export as environment variables for cloud instances
3. Pass configuration to training scripts
"""

import subprocess


# Add to your existing launch.py
def get_metta_config_env():
    """Get Metta configuration as environment variables."""
    # Use metta export-env command
    result = subprocess.run(["metta", "export-env"], capture_output=True, text=True, check=True)

    # Parse the output into a dict
    env_vars = {}
    for line in result.stdout.strip().split("\n"):
        if line.startswith("export "):
            line = line.replace("export ", "")
            key, value = line.split("=", 1)
            # Remove quotes
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            env_vars[key] = value

    return env_vars


def launch_with_config(task_config):
    """
    Launch a skypilot task with Metta configuration.

    This function shows how to:
    1. Get configuration from the unified config file
    2. Add it to the task's environment variables
    3. Ensure remote instances have the right settings
    """
    # Get Metta configuration
    config_env = get_metta_config_env()

    # Add to task environment
    if "envs" not in task_config:
        task_config["envs"] = {}

    # Merge with existing environment variables
    task_config["envs"].update(config_env)

    # Example: Also set up the config file on remote instance
    if "setup" not in task_config:
        task_config["setup"] = []

    # Add command to export config on remote
    task_config["setup"].append(
        "# Export Metta configuration\n"
        "cat > ~/.metta_env << 'EOF'\n" + "\n".join(f"{k}={v}" for k, v in config_env.items()) + "\nEOF\n"
        "source ~/.metta_env"
    )

    return task_config


# Example usage in your launch.py
if __name__ == "__main__":
    # Your existing task configuration
    task = {
        "name": "metta-training",
        "resources": {
            "accelerators": "A100:8",
        },
        "setup": [
            "pip install -e .",
        ],
        "run": [
            "python -m metta.tools.train",
        ],
    }

    # Add Metta configuration
    task = launch_with_config(task)

    # Now task includes all configuration from ~/.metta/config.yaml
    print("Task configuration with Metta config:")
    print(f"Environment variables: {task.get('envs', {})}")

    # Launch with skypilot
    # sky.launch(task, ...)
