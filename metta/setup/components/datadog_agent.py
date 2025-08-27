import json
import os
import subprocess

import boto3

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning


@register_module
class DatadogAgentSetup(SetupModule):
    install_once = True

    @property
    def name(self) -> str:
        return "datadog-agent"

    def dependencies(self) -> list[str]:
        return ["aws"]

    @property
    def description(self) -> str:
        return "Datadog agent for system monitoring and log aggregation"

    def _is_applicable(self) -> bool:
        # Only install this within the docker containers that are running on EC2 instances
        return os.path.exists("/.dockerenv")

    def check_installed(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "datadog-agent"],
                capture_output=True,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _get_dd_api_key(self) -> str:
        api_key = os.environ.get("DD_API_KEY")
        if api_key:
            return api_key

        client = boto3.client("secretsmanager", region_name="us-east-1")
        try:
            secret_json = client.get_secret_value(SecretId="datadog/api-key")
        except client.exceptions.ResourceNotFoundException as e:
            raise Exception("Datadog API key not found in AWS Secrets Manager.") from e

        if not (secret := secret_json.get("SecretString")):
            raise Exception("Datadog API key not found in datadog/api-key secret.")
        return secret

    def install(self) -> None:
        info("Getting Datadog API key...")
        try:
            api_key = self._get_dd_api_key()
        except Exception as e:
            warning(f"Could not get Datadog API key: {e}")
            warning("Skipping Datadog agent installation.")
            return

        # Set environment variables for the install script
        env = os.environ.copy()
        env["DD_API_KEY"] = api_key
        env["DD_SITE"] = os.environ.get("DD_SITE", "datadoghq.com")
        env["DD_VERSION"] = os.environ.get("DD_VERSION", os.environ.get("METTA_GIT_REF", "unknown"))
        env["DD_TRACE_ENABLED"] = os.environ.get("DD_TRACE_ENABLED", "true")
        env["DD_LOGS_ENABLED"] = os.environ.get("DD_LOGS_ENABLED", "true")

        # Set tags from SkyPilot environment variables
        tags = env.get("DD_TAGS", "").split(" ")
        for env_var, tag in [
            ("METTA_RUN_ID", "metta_run_id"),
            ("SKYPILOT_TASK_ID", "skypilot_task_id"),
            ("SKYPILOT_NODE_RANK", "node_rank"),
            ("SKYPILOT_NUM_NODES", "num_nodes"),
        ]:
            if value := os.environ.get(env_var):
                tags.append(f"{tag}:{value}")

        if tags:
            env["DD_TAGS"] = " ".join(tags)

        info("Installing Datadog agent...")

        # For containers, we need to install without systemd
        # Use DD_INSTALL_ONLY to prevent the script from starting the service
        env["DD_INSTALL_ONLY"] = "true"
        info("Container environment detected - installing without systemd.")

        install_cmd = 'bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script_agent7.sh)"'
        result = subprocess.run(
            install_cmd,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error(f"Failed to install Datadog agent: {result.stdout}\n{result.stderr}")
            return

        success("Datadog agent installed successfully.")

        # Start the agent in background for containers
        info("Starting Datadog agent in background...")
        try:
            # Write configuration to the agent config file
            config_dir = "/etc/datadog-agent"
            os.makedirs(config_dir, exist_ok=True)

            # Start the agent process in background
            subprocess.Popen(
                ["/opt/datadog-agent/bin/agent/agent", "run"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            success("Datadog agent started in background with env vars:")
            info(json.dumps({k: v for k, v in env.items() if k.startswith("DD_")}, indent=2))
        except Exception as e:
            warning(f"Failed to start Datadog agent in background: {e}")
