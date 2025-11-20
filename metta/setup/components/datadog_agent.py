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

        # Set hostname for Docker environments (required for agent to start)
        # Use SKYPILOT_TASK_ID if available, otherwise use METTA_RUN_ID, fallback to hostname
        # Must be RFC1123 compliant: lowercase, numbers, hyphens only, max 63 chars, no underscores
        raw_hostname = (
            os.environ.get("SKYPILOT_TASK_ID")
            or os.environ.get("METTA_RUN_ID")
            or os.environ.get("HOSTNAME")
            or "skypilot-job"
        )
        # Sanitize hostname: replace underscores with hyphens, lowercase, truncate to 63 chars
        hostname = raw_hostname.lower().replace("_", "-")[:63]
        # Ensure it starts with alphanumeric
        if not hostname[0].isalnum():
            hostname = "skypilot-" + hostname
        env["DD_HOSTNAME"] = hostname

        # Set tags from SkyPilot environment variables
        tags = [t for t in env.get("DD_TAGS", "").split(" ") if t.strip()]  # Filter empty strings
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
            # Configure datadog.yaml with hostname, logs, and tags
            config_file = "/etc/datadog-agent/datadog.yaml"
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config_content = f.read()

                    config_updates = []

                    # Set hostname if not already configured
                    if "hostname:" not in config_content:
                        raw_hostname = (
                            os.environ.get("SKYPILOT_TASK_ID")
                            or os.environ.get("METTA_RUN_ID")
                            or os.environ.get("HOSTNAME")
                            or "skypilot-job"
                        )
                        # Sanitize hostname: replace underscores with hyphens, lowercase, truncate to 63 chars
                        hostname = raw_hostname.lower().replace("_", "-")[:63]
                        # Ensure it starts with alphanumeric
                        if not hostname[0].isalnum():
                            hostname = "skypilot-" + hostname
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

                    # Build log tags for log collection config (same tags as host tags)
                    log_tags_for_config = []
                    for env_var, tag in [
                        ("METTA_RUN_ID", "metta_run_id"),
                        ("SKYPILOT_TASK_ID", "skypilot_task_id"),
                        ("SKYPILOT_NODE_RANK", "node_rank"),
                        ("SKYPILOT_NUM_NODES", "num_nodes"),
                    ]:
                        if value := os.environ.get(env_var):
                            log_tags_for_config.append(f"{tag}:{value}")

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

                    # Also add log collection config directly to main datadog.yaml for reliability
                    # This ensures logs are collected even if the separate config file isn't picked up
                    try:
                        with open(config_file, "r") as f:
                            main_config = f.read()

                        # Check if logs section already exists in main config
                        # Note: Datadog agent reads logs from conf.d/*.d/conf.yaml files, not from main datadog.yaml
                        # So we don't add logs here - they're in the separate config file
                        # But we ensure logs_enabled is set above
                        pass
                    except Exception as e:
                        warning(f"Could not add logs to main config: {e}")
                except Exception as e:
                    warning(f"Could not update Datadog config file: {e}")

            # Configure log collection paths
            try:
                conf_d_dir = "/etc/datadog-agent/conf.d"
                if os.path.exists(conf_d_dir):
                    # Use standard Datadog integration directory format: <name>.d
                    # The .d suffix is required for Datadog to recognize the integration
                    custom_logs_dir = os.path.join(conf_d_dir, "skypilot_training.d")
                    os.makedirs(custom_logs_dir, exist_ok=True)

                    # Build tags list for log configuration
                    log_tags = []
                    for env_var, tag in [
                        ("METTA_RUN_ID", "metta_run_id"),
                        ("SKYPILOT_TASK_ID", "skypilot_task_id"),
                        ("SKYPILOT_NODE_RANK", "node_rank"),
                        ("SKYPILOT_NUM_NODES", "num_nodes"),
                    ]:
                        if value := os.environ.get(env_var):
                            log_tags.append(f"{tag}:{value}")

                    # Format tags as YAML list
                    tags_yaml = ""
                    if log_tags:
                        tags_lines = "\n".join([f'      - "{tag}"' for tag in log_tags])
                        tags_yaml = f"    tags:\n{tags_lines}\n"

                    log_config_file = os.path.join(custom_logs_dir, "conf.yaml")
                    # Use explicit file paths (Datadog doesn't reliably support wildcards)
                    # Create empty log files to ensure they exist when agent starts
                    log_config = f"""# Custom log collection for SkyPilot jobs
logs:
  - type: file
    path: /tmp/datadog-agent.log
    service: datadog-agent
    source: datadog-agent
    sourcecategory: monitoring
{tags_yaml}
  - type: file
    path: /tmp/training_logs/training_combined.log
    service: skypilot-training
    source: training
    sourcecategory: application
{tags_yaml}
"""
                    # Ensure log directory and files exist before writing config
                    training_log_dir = "/tmp/training_logs"
                    os.makedirs(training_log_dir, exist_ok=True)
                    # Create empty log files so Datadog agent can start collecting immediately
                    for log_file in ["training_combined.log"]:
                        log_path = os.path.join(training_log_dir, log_file)
                        if not os.path.exists(log_path):
                            with open(log_path, "a") as f:
                                f.write("")  # Create empty file
                            os.chmod(log_path, 0o666)  # Ensure readable/writable by everyone
                    with open(log_config_file, "w") as f:
                        f.write(log_config)
                    # Set proper permissions so agent can read it
                    os.chmod(log_config_file, 0o644)
                    info(f"Created custom log collection configuration at {log_config_file}")
                    info(f"Config file size: {os.path.getsize(log_config_file)} bytes")
                    # Verify the YAML is valid by checking basic structure
                    if "logs:" in log_config and "type: file" in log_config:
                        info("Log collection config appears valid (contains 'logs:' and 'type: file')")
                    else:
                        warning("Log collection config might be malformed!")
            except Exception as e:
                warning(f"Could not create log collection config: {e}")

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
