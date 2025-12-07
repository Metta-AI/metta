import os
import platform
import subprocess
from shutil import which

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning
from softmax.aws.secrets_manager import get_secretsmanager_secret

LOG_CONFIG = """\
logs:
  - type: file
    path: /tmp/datadog-training.log
    service: skypilot-training
    source: training
"""


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
        return platform.system() == "Linux"

    def check_installed(self) -> bool:
        if os.path.exists("/opt/datadog-agent/bin/agent/agent"):
            return True
        try:
            result = subprocess.run(
                ["systemctl", "status", "datadog-agent"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode != 4
        except FileNotFoundError:
            return False

    def _get_dd_api_key(self) -> str | None:
        return os.environ.get("DD_API_KEY") or get_secretsmanager_secret("datadog/api-key", require_exists=False)

    def _build_tags(self) -> list[str]:
        tags = []
        for env_var, tag in [
            ("METTA_RUN_ID", "metta_run_id"),
            ("SKYPILOT_TASK_ID", "skypilot_task_id"),
            ("SKYPILOT_NODE_RANK", "node_rank"),
            ("SKYPILOT_NUM_NODES", "num_nodes"),
        ]:
            if value := os.environ.get(env_var):
                tags.append(f"{tag}:{value}")
        return tags

    def _setup_log_config(self) -> None:
        conf_dir = "/etc/datadog-agent/conf.d/skypilot_training.d"
        try:
            os.makedirs(conf_dir, exist_ok=True)
            config_path = os.path.join(conf_dir, "conf.yaml")
            with open(config_path, "w") as f:
                f.write(LOG_CONFIG)
            os.chmod(config_path, 0o644)
        except Exception as e:
            warning(f"Could not create Datadog log config: {e}")

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        try:
            api_key = self._get_dd_api_key()
        except Exception as e:
            warning(f"Could not get Datadog API key: {e}")
            return
        if not api_key:
            warning("No Datadog API key found. Skipping Datadog agent installation.")
            return

        env = os.environ.copy()
        env["DD_API_KEY"] = api_key
        env["DD_SITE"] = os.environ.get("DD_SITE", "datadoghq.com")
        env["DD_VERSION"] = os.environ.get("DD_VERSION", os.environ.get("METTA_GIT_REF", "unknown"))
        env["DD_TRACE_ENABLED"] = os.environ.get("DD_TRACE_ENABLED", "true")
        env["DD_LOGS_ENABLED"] = os.environ.get("DD_LOGS_ENABLED", "true")

        tags = [t for t in env.get("DD_TAGS", "").split(" ") if t.strip()]
        tags.extend(self._build_tags())
        if tags:
            env["DD_TAGS"] = " ".join(tags)

        if self.check_installed():
            info("Datadog agent already installed.")
            self._setup_log_config()
            restart_cmd = ["systemctl", "restart", "datadog-agent"]
            if which("sudo"):
                restart_cmd = ["sudo", *restart_cmd]
            try:
                subprocess.run(restart_cmd, check=True)
                success("Datadog agent restarted.")
            except (FileNotFoundError, subprocess.CalledProcessError):
                warning("Could not restart Datadog agent via systemctl.")
            return

        info("Installing Datadog agent...")
        install_cmd = 'bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script_agent7.sh)"'
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

        if result.returncode != 0:
            error(f"Failed to install Datadog agent: {result.stdout}\n{result.stderr}")
            return

        self._setup_log_config()
        success("Datadog agent installed successfully.")
