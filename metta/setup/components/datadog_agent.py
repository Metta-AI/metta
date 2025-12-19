import os
import platform
import socket
import subprocess
import time

import yaml

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.utils import error, info, success, warning
from softmax.aws.secrets_manager import get_secretsmanager_secret

AGENT_BINARY = "/opt/datadog-agent/bin/agent/agent"


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
        return os.path.exists(AGENT_BINARY)

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
        log_file = os.environ.get("METTA_DD_LOG_FILE")
        if not log_file:
            warning("METTA_DD_LOG_FILE not set, skipping log config")
            return
        info(f"Configuring DD to tail: {log_file}")

        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            open(log_file, "a").close()
            os.chmod(log_file, 0o666)
        except Exception as e:
            warning(f"Could not create log file {log_file}: {e}")
            return

        log_entry: dict = {
            "type": "file",
            "path": log_file,
            "service": "skypilot-training",
            "source": "training",
        }
        tags = self._build_tags()
        if tags:
            log_entry["tags"] = tags

        config = {"logs": [log_entry]}
        conf_dir = "/etc/datadog-agent/conf.d/skypilot_training.d"
        try:
            os.makedirs(conf_dir, exist_ok=True)
            config_path = os.path.join(conf_dir, "conf.yaml")
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            os.chmod(config_path, 0o644)
            info(f"Created DD log config at {config_path}")
        except Exception as e:
            warning(f"Could not create Datadog log config: {e}")

    def _configure_agent(self, api_key: str) -> None:
        defaults_path = self.repo_root / "devops" / "datadog" / "defaults.yaml"
        try:
            with open(defaults_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            warning(f"Could not load DD defaults from {defaults_path}: {e}")
            config = {}

        config["api_key"] = api_key
        config["hostname"] = socket.gethostname() or "skypilot-node"

        config_file = "/etc/datadog-agent/datadog.yaml"
        try:
            with open(config_file, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            info(f"Wrote DD config to {config_file}")
        except Exception as e:
            warning(f"Could not write DD config: {e}")

    def _stop_agent(self) -> None:
        subprocess.run(["pkill", "-f", AGENT_BINARY], capture_output=True)

    def _start_agent(self) -> None:
        if not os.path.exists(AGENT_BINARY):
            return
        self._stop_agent()
        log_path = "/tmp/dd-agent-startup.log"
        with open(log_path, "w") as log_file:
            subprocess.Popen(
                [AGENT_BINARY, "run"],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        time.sleep(3)
        result = subprocess.run(["pgrep", "-f", AGENT_BINARY], capture_output=True)
        if result.returncode == 0:
            info("Started Datadog agent.")
        else:
            warning("Datadog agent not running after 3s")
            try:
                with open(log_path) as f:
                    for line in f.readlines()[-20:]:
                        warning(f"  {line.rstrip()}")
            except Exception:
                pass

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
            self._configure_agent(api_key)
            self._setup_log_config()
            self._start_agent()
            return

        info("Installing Datadog agent...")
        env["DD_INSTALL_ONLY"] = "true"
        subprocess.run(
            "curl -fsSL https://s3.amazonaws.com/dd-agent/scripts/install_script_agent7.sh | bash",
            shell=True,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        if not os.path.exists(AGENT_BINARY):
            error("Datadog agent binary not found after install")
            return

        self._configure_agent(api_key)
        self._setup_log_config()
        self._start_agent()
        success("Datadog agent installed successfully.")
