import os
import platform
import subprocess
import time
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

    def _sanitize_hostname(self) -> str:
        """Sanitize hostname to be RFC1123 compliant."""
        raw_hostname = (
            os.environ.get("SKYPILOT_TASK_ID")
            or os.environ.get("METTA_RUN_ID")
            or os.environ.get("HOSTNAME")
            or "skypilot-job"
        )
        hostname = raw_hostname.lower().replace("_", "-")[:63]
        hostname = hostname.rstrip("-")
        if not hostname or not hostname[0].isalnum():
            hostname = "skypilot-" + hostname
        hostname = hostname.rstrip("-")
        if not hostname or len(hostname) < 3:
            hostname = "skypilot-job"
        return hostname

    def _build_tags(self) -> list[str]:
        """Build tags list from environment variables."""
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

    def update_log_config_and_start_agent(self) -> None:
        """Update Datadog log config with runtime tags and start agent."""
        agent_binary = "/opt/datadog-agent/bin/agent/agent"
        if not os.path.exists(agent_binary):
            return  # Agent not installed, skip silently

        try:
            # Update existing log config with runtime tags
            conf_d_dir = "/etc/datadog-agent/conf.d"
            if os.path.exists(conf_d_dir):
                custom_logs_dir = os.path.join(conf_d_dir, "skypilot_training.d")
                log_config_file = os.path.join(custom_logs_dir, "conf.yaml")

                if not os.path.exists(log_config_file):
                    # Config should have been created at setup, but create it if missing
                    os.makedirs(custom_logs_dir, exist_ok=True)
                    log_config_template = """# Custom log collection for SkyPilot jobs
logs:
  - type: file
    path: /tmp/datadog-agent.log
    service: datadog-agent
    source: datadog-agent
    sourcecategory: monitoring
  - type: file
    path: /tmp/datadog-training.log
    service: skypilot-training
    source: training
    sourcecategory: application
"""
                    with open(log_config_file, "w") as f:
                        f.write(log_config_template)

                # Read existing config and add tags
                with open(log_config_file, "r") as f:
                    config_content = f.read()

                log_tags = self._build_tags()
                if log_tags:
                    tags_lines = "\n".join([f'      - "{tag}"' for tag in log_tags])
                    tags_yaml = f"    tags:\n{tags_lines}\n"

                    # Add tags to each log entry if not already present
                    if "tags:" not in config_content:
                        # Insert tags after each log entry's sourcecategory line
                        updated_config = config_content
                        for log_entry in ["datadog-agent", "skypilot-training"]:
                            pattern = f"sourcecategory: {log_entry}\n"
                            replacement = f"sourcecategory: {log_entry}\n{tags_yaml}"
                            updated_config = updated_config.replace(pattern, replacement, 1)

                        with open(log_config_file, "w") as f:
                            f.write(updated_config)
                        os.chmod(log_config_file, 0o644)

            # Restart agent to pick up config
            subprocess.run(["pkill", "-f", "datadog-agent.*run"], check=False, capture_output=True)
            time.sleep(2)

            # Start agent with DD_LOGS_ENABLED=true
            env = os.environ.copy()
            env["DD_LOGS_ENABLED"] = "true"
            with open("/tmp/datadog-agent.log", "a") as log_file:
                subprocess.Popen(
                    [agent_binary, "run"],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
        except Exception:
            pass  # Non-fatal

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
        env["DD_HOSTNAME"] = self._sanitize_hostname()

        tags = [t for t in env.get("DD_TAGS", "").split(" ") if t.strip()]
        tags.extend(self._build_tags())
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
            # Configure datadog.yaml with hostname, logs, and tags
            config_file = "/etc/datadog-agent/datadog.yaml"
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config_content = f.read()

                    config_updates = []

                    if "hostname:" not in config_content:
                        hostname = self._sanitize_hostname()
                        config_updates.append(f"hostname: {hostname}")
                        info(f"Set hostname in Datadog config: {hostname}")

                    # Enable logs if not already configured
                    if "logs_enabled:" not in config_content:
                        config_updates.append("logs_enabled: true")
                        info("Enabled log collection in Datadog config")

                    # Add logs_config section for better reliability (recommended by Datadog docs)
                    if "logs_config:" not in config_content:
                        config_updates.append("logs_config:")
                        config_updates.append("  auto_multi_line_detection: true")
                        config_updates.append("  force_use_http: true")
                        info("Added logs_config section to Datadog config")

                    # Set tags if not already configured
                    if tags and "tags:" not in config_content:
                        # Filter out empty tags and format as YAML list
                        valid_tags = [tag for tag in tags if tag and tag.strip()]
                        if valid_tags:
                            tags_list = "[" + ", ".join([f'"{tag}"' for tag in valid_tags]) + "]"
                            config_updates.append(f"tags: {tags_list}")
                            info(f"Set tags in Datadog config: {tags_list}")

                    # Append any updates to config file
                    if config_updates:
                        with open(config_file, "a") as f:
                            f.write("\n# Metta SkyPilot job configuration\n")
                            for update in config_updates:
                                f.write(f"{update}\n")

                        # If we enabled logs, we need to restart the agent to pick up the change
                        # But we can't restart here (no systemd in Docker), so the run-phase script will handle it
                        if "logs_enabled: true" in config_updates:
                            info("logs_enabled set - agent will be restarted in run phase to pick up changes")
                except Exception as e:
                    warning(f"Could not update Datadog config file: {e}")

            # Create log collection config at setup (template, tags added at runtime)
            try:
                conf_d_dir = "/etc/datadog-agent/conf.d"
                if os.path.exists(conf_d_dir):
                    custom_logs_dir = os.path.join(conf_d_dir, "skypilot_training.d")
                    os.makedirs(custom_logs_dir, exist_ok=True)

                    log_config_file = os.path.join(custom_logs_dir, "conf.yaml")
                    # Write template config (tags will be added at runtime)
                    log_config = """# Custom log collection for SkyPilot jobs
# Tags will be added at runtime when environment variables are available
logs:
  - type: file
    path: /tmp/datadog-agent.log
    service: datadog-agent
    source: datadog-agent
    sourcecategory: monitoring
  - type: file
    path: /tmp/datadog-training.log
    service: skypilot-training
    source: training
    sourcecategory: application
"""
                    with open(log_config_file, "w") as f:
                        f.write(log_config)
                    os.chmod(log_config_file, 0o644)
                    info(f"Created Datadog log collection config at {log_config_file} (tags will be added at runtime)")
            except Exception as e:
                warning(f"Could not create Datadog log collection config: {e}")

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
