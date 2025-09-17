import os
import platform
import subprocess

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning
from softmax.aws.secrets_manager import get_secretsmanager_secret


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
        # Only applicable on Linux systems (EC2 instances)
        return platform.system() == "Linux"

    def check_installed(self) -> bool:
        # Check if datadog-agent service exists
        try:
            result = subprocess.run(
                ["systemctl", "status", "datadog-agent"],
                capture_output=True,
                text=True,
                check=False,
            )
            # Service exists if systemctl can find it (even if not running)
            return result.returncode != 4  # 4 means service not found
        except FileNotFoundError:
            # systemctl not available
            return False

    def _get_dd_api_key(self) -> str | None:
        return os.environ.get("DD_API_KEY") or get_secretsmanager_secret("datadog/api-key", require_exists=False)

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        info("Getting Datadog API key...")
        try:
            api_key = self._get_dd_api_key()
        except Exception as e:
            warning(f"Could not get Datadog API key: {e}")
            warning("Skipping Datadog agent installation.")
            return
        if not api_key:
            warning("No Datadog API key found. Skipping Datadog agent installation.")
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

        # Check if already installed
        if self.check_installed():
            info("Datadog agent already installed.")
            # Just restart with new config/tags if needed
            try:
                subprocess.run(["sudo", "systemctl", "restart", "datadog-agent"], check=True)
                success("Datadog agent restarted with updated configuration.")
            except subprocess.CalledProcessError:
                warning("Failed to restart Datadog agent.")
            return

        info("Installing Datadog agent...")

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
