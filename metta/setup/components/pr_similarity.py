from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import boto3

from metta.setup.components.base import SetupModule
from metta.setup.registry import register_module
from metta.setup.saved_settings import UserType, get_saved_settings
from metta.setup.utils import debug, info, warning
from metta.tools.pr_similarity import DEFAULT_CACHE_PATH, resolve_cache_paths
from softmax.aws.secrets_manager import get_secretsmanager_secret


def _resolve_cache_paths(repo_root: Path) -> tuple[Path, Path]:
    cache_base = repo_root / DEFAULT_CACHE_PATH
    return resolve_cache_paths(cache_base)


@register_module
class PrSimilaritySetup(SetupModule):
    @property
    def name(self) -> str:  # type: ignore[override]
        return "pr-similarity"

    @property
    def description(self) -> str:
        return "Configure PR similarity MCP server (Codex & Claude)"

    def dependencies(self) -> list[str]:
        return ["uv", "aws"]

    def check_installed(self) -> bool:
        meta_path, vectors_path = _resolve_cache_paths(self.repo_root)
        return meta_path.exists() and vectors_path.exists() and shutil.which("metta-pr-similarity-mcp") is not None

    def install(self, non_interactive: bool = False, force: bool = False) -> None:
        api_key = get_secretsmanager_secret("GEMINI-API-KEY", require_exists=False)

        self._ensure_cache(force)
        self._configure_claude(api_key=api_key, force=force)
        self._configure_codex(api_key=api_key)
        self._configure_cursor(api_key=api_key)

    def _ensure_cache(self, force: bool) -> None:
        meta_path, vectors_path = _resolve_cache_paths(self.repo_root)
        need_download = force or not meta_path.exists() or not vectors_path.exists()

        if not need_download:
            threshold = datetime.now(timezone.utc) - timedelta(days=2)
            meta_mtime = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=timezone.utc)
            vectors_mtime = datetime.fromtimestamp(vectors_path.stat().st_mtime, tz=timezone.utc)
            if meta_mtime < threshold or vectors_mtime < threshold:
                need_download = True

        if not need_download:
            debug(f"PR similarity cache already present at {meta_path.parent}")
            return

        bucket = "softmax-public"
        prefix = "pr-cache/"

        try:
            session = boto3.session.Session()
            client = session.client("s3")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, prefix + meta_path.name, str(meta_path))
            client.download_file(bucket, prefix + vectors_path.name, str(vectors_path))
            info(f"Downloaded PR similarity cache from s3://{bucket}/{prefix}")
        except Exception as error:  # pragma: no cover - external dependency
            warning(f"Unable to download PR similarity cache: {error}")

    def _configure_claude(self, *, api_key: Optional[str], force: bool) -> None:
        if shutil.which("claude") is None:
            debug("Claude CLI not found on PATH. Skipping Claude MCP configuration.")
            return

        if shutil.which("metta-pr-similarity-mcp") is None:
            warning("metta-pr-similarity-mcp is not available on PATH; skipping Claude MCP registration.")
            return

        # Keep local import: slow loading (~1000ms)
        from metta.tools.pr_similarity import API_KEY_ENV

        saved_settings = get_saved_settings()
        if saved_settings.user_type not in {UserType.SOFTMAX, UserType.SOFTMAX_DOCKER}:
            debug(
                f"Skipping Claude MCP registration for non-employee user type {saved_settings.user_type.value}",
            )
            return

        command_path = shutil.which("metta-pr-similarity-mcp")
        if command_path is None:
            warning("metta-pr-similarity-mcp is not available on PATH; skipping Claude MCP registration.")
            return

        if not force:
            self.run_command(
                ["claude", "mcp", "remove", "metta-pr-similarity"],
                check=False,
                capture_output=False,
            )

        if not api_key:
            warning("Skipping Claude MCP registration: no GEMINI API key available from Secrets Manager.")
            return

        command = [
            "claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            "metta-pr-similarity",
            "--env",
            f"{API_KEY_ENV}={api_key}",
            "--",
            command_path,
        ]

        try:
            self.run_command(command, capture_output=False)
            info("Configured Claude MCP server 'metta-pr-similarity'.")
        except Exception as error:  # pragma: no cover - external dependency
            warning(f"Failed to configure Claude MCP server: {error}")

    def _configure_codex(self, *, api_key: Optional[str]) -> None:
        codex_executable = shutil.which("codex")
        if not codex_executable:
            debug("Codex CLI not found on PATH. Skipping PR similarity MCP registration.")
            return

        command_path = shutil.which("metta-pr-similarity-mcp")
        if not command_path:
            warning(
                "Unable to locate 'metta-pr-similarity-mcp' on PATH. Install the MCP package before configuring Codex.",
            )
            return

        if not api_key:
            warning("Skipping Codex MCP registration: no GEMINI API key available from Secrets Manager.")
            return

        for name in ("metta-pr-similarity", "metta-pr-similarity-mcp"):
            self.run_command(
                [codex_executable, "mcp", "remove", name],
                check=False,
                capture_output=False,
            )

        try:
            self.run_command(
                [
                    codex_executable,
                    "mcp",
                    "add",
                    "--env",
                    f"GEMINI_API_KEY={api_key}",
                    "metta-pr-similarity",
                    command_path,
                ],
                capture_output=False,
            )
            info("Configured Codex MCP server 'metta-pr-similarity'.")
        except Exception as error:
            warning(
                f"Failed to configure Codex MCP server 'metta-pr-similarity'. {error}",
            )

    def _configure_cursor(self, *, api_key: Optional[str]) -> None:
        if not api_key:
            warning("Skipping Cursor MCP registration: no GEMINI API key available from Secrets Manager.")
            return

        command_path = shutil.which("metta-pr-similarity-mcp")
        if not command_path:
            warning(
                "Unable to locate 'metta-pr-similarity-mcp' on PATH. "
                "Install the MCP package before configuring Cursor.",
            )
            return

        cursor_root = Path.home() / ".cursor"
        if not cursor_root.exists():
            debug("Cursor directory not found. Skipping Cursor MCP configuration.")
            return
        config_path = cursor_root / "mcp.json"

        raw_config = ""
        if config_path.exists():
            try:
                raw_config = config_path.read_text(encoding="utf-8")
            except Exception as error:
                warning(f"Unable to read Cursor MCP configuration: {error}")
                return

        config_data: dict[str, Any] = {}
        if raw_config.strip():
            try:
                parsed = json.loads(raw_config)
            except json.JSONDecodeError as error:
                warning(f"Cursor MCP configuration is not valid JSON: {error}")
                return
            if isinstance(parsed, dict):
                config_data = parsed
            else:
                warning("Cursor MCP configuration has unexpected structure; leaving it unchanged.")
                return

        servers = config_data.get("mcpServers")
        if not isinstance(servers, dict):
            servers = {}
        config_data["mcpServers"] = servers

        existing_entry = servers.get("metta-pr-similarity")
        entry: dict[str, Any]
        if isinstance(existing_entry, dict):
            entry = dict(existing_entry)
        else:
            entry = {}

        entry["command"] = command_path
        entry["type"] = "stdio"
        entry["name"] = "metta-pr-similarity"
        entry["alwaysAllow"] = True
        args = entry.get("args")
        if not isinstance(args, list):
            args = []
        entry["args"] = args

        env: dict[str, str]
        existing_env = entry.get("env")
        if isinstance(existing_env, dict):
            env = {str(key): str(value) for key, value in existing_env.items()}
        else:
            env = {}
        env["GEMINI_API_KEY"] = api_key
        entry["env"] = env

        servers["metta-pr-similarity"] = entry

        try:
            config_path.write_text(json.dumps(config_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            info("Configured Cursor MCP server 'metta-pr-similarity'.")
        except Exception as error:
            warning(f"Failed to write Cursor MCP configuration: {error}")
