import os
import platform
import subprocess
from shutil import which

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
        # Check if datadog-agent binary exists (works in Docker and regular systems)
        agent_binary = "/opt/datadog-agent/bin/agent/agent"
        if os.path.exists(agent_binary):
            return True

        # Fallback: check for systemd service (for non-Docker environments)
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
            # systemctl not available (e.g., in Docker containers)
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

        # Check if already installed - verify binary exists (not just systemd service)
        agent_binary = "/opt/datadog-agent/bin/agent/agent"
        binary_exists = os.path.exists(agent_binary)

        if binary_exists:
            info("Datadog agent already installed (binary found).")
            # Just restart with new config/tags if needed
            restart_cmd = ["systemctl", "restart", "datadog-agent"]
            if which("sudo"):
                restart_cmd = ["sudo", *restart_cmd]
            try:
                subprocess.run(restart_cmd, check=True)
                success(f"Datadog agent restarted with updated configuration (cmd: {' '.join(restart_cmd)}).")
            except FileNotFoundError:
                warning("systemctl not available; skipping Datadog agent restart.")
            except subprocess.CalledProcessError:
                warning(f"Failed to restart Datadog agent using {' '.join(restart_cmd)}.")
            return
        elif self.check_installed() and not binary_exists:
            # Systemd service exists but binary doesn't - need to reinstall
            info("Datadog service found but binary missing. Reinstalling...")

        info("Installing Datadog agent...")

        # The install script needs to run with proper permissions
        install_cmd = 'bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script_agent7.sh)"'

        # Try with sudo if available
        if which("sudo"):
            install_cmd = f"sudo {install_cmd}"

        result = subprocess.run(
            install_cmd,
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        # Check if binary exists even if return code is non-zero (install script may return non-zero on warnings)
        agent_binary = "/opt/datadog-agent/bin/agent/agent"
        if os.path.exists(agent_binary):
            success("Datadog agent installed successfully (binary found).")
            return

        if result.returncode != 0:
            error(f"Failed to install Datadog agent (exit code {result.returncode}): {result.stdout}\n{result.stderr}")
            return

        success("Datadog agent installed successfully.")

    def _start_agent_process(self) -> subprocess.Popen | None:
        """Start Datadog agent as a background process (Docker-compatible).

        Returns:
            Popen process object if started successfully, None otherwise.
        """
        agent_binary = "/opt/datadog-agent/bin/agent/agent"

        if not os.path.exists(agent_binary):
            warning("Datadog agent binary not found. Cannot start agent.")
            return None

        try:
            # Start agent in background
            process = subprocess.Popen(
                [agent_binary, "run"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
            )
            info(f"Started Datadog agent process (PID: {process.pid})")
            return process
        except Exception as e:
            warning(f"Failed to start Datadog agent: {e}")
            return None

    def _verify_agent_running(self) -> bool:
        """Verify that Datadog agent is actually running.

        Returns:
            True if agent is running, False otherwise.
        """
        agent_binary = "/opt/datadog-agent/bin/agent/agent"

        if not os.path.exists(agent_binary):
            return False

        try:
            # Check agent status
            result = subprocess.run(
                [agent_binary, "status"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            # Agent is running if status command succeeds
            return result.returncode == 0
        except Exception:
            return False
