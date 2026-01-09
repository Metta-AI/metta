"""Policy submission command for CoGames."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import typer
from rich.console import Console

from cogames.cli.base import console
from cogames.cli.login import DEFAULT_COGAMES_SERVER
from cogames.cli.policy import PolicySpec, get_policy_spec

if TYPE_CHECKING:
    from cogames.cli.client import TournamentServerClient

from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec

DEFAULT_SUBMIT_SERVER = "https://api.observatory.softmax-research.net"


@dataclass
class UploadResult:
    policy_version_id: uuid.UUID
    name: str
    version: int


def validate_paths(paths: list[str], console: Console) -> list[Path]:
    """Validate that all paths are relative and within CWD.

    Ensures paths don't escape the current working directory using '../'.
    Returns list of resolved Path objects.
    """
    cwd = Path.cwd()
    validated_paths = []

    for path_str in paths:
        path = Path(path_str)

        # Check if path is absolute
        if path.is_absolute():
            console.print(f"[red]Error:[/red] Path must be relative: {path_str}")
            raise ValueError(f"Absolute paths not allowed: {path_str}")

        # Resolve the path and check it's within CWD
        try:
            resolved = (cwd / path).resolve()
            # Check if resolved path is under CWD
            resolved.relative_to(cwd)
        except ValueError:
            console.print(f"[red]Error:[/red] Path escapes current directory: {path_str}")
            raise ValueError(f"Path escapes CWD: {path_str}") from None

        # Check if path exists
        if not resolved.exists():
            console.print(f"[red]Error:[/red] Path does not exist: {path_str}")
            raise FileNotFoundError(f"Path not found: {path_str}")

        validated_paths.append(path)

    return validated_paths


def get_latest_cogames_version() -> str:
    """Get the latest published cogames version."""
    response = httpx.get("https://pypi.org/pypi/cogames/json")
    response.raise_for_status()
    return response.json()["info"]["version"]


def create_temp_validation_env() -> Path:
    """Create a temporary directory with a minimal pyproject.toml.

    The pyproject.toml depends on the latest published cogames package.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="cogames_submit_"))

    latest_cogames_version = get_latest_cogames_version()

    pyproject_content = f"""[project]
name = "cogames-submission-validator"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["cogames=={latest_cogames_version}"]

[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"
"""

    pyproject_path = temp_dir / "pyproject.toml"
    pyproject_path.write_text(pyproject_content)

    return temp_dir


def copy_files_maintaining_structure(files: list[Path], dest_dir: Path) -> None:
    """Copy files to destination, maintaining directory structure.

    If a file is 'train_dir/model.pt', it will be copied to 'dest_dir/train_dir/model.pt'.
    """
    for file_path in files:
        dest_path = dest_dir / file_path

        if file_path.is_dir():
            shutil.copytree(file_path, dest_path, dirs_exist_ok=True)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_path)


def validate_policy_spec(policy_spec: PolicySpec) -> None:
    """Validate policy works.

    Loads the policy and runs a single episode (up to 10 steps) using the same
    multi_episode_rollout flow as `cogames eval`.
    """
    from cogames.cli.mission import get_mission
    from mettagrid.simulator.multi_episode.rollout import multi_episode_rollout

    _, env_cfg, _ = get_mission("machina_1")
    policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
    policy = initialize_or_load_policy(policy_env_info, policy_spec)

    # Run 1 episode for up to 10 steps to validate the policy works
    env_cfg.game.max_steps = 10
    multi_episode_rollout(
        env_cfg=env_cfg,
        policies=[policy],
        episodes=1,
        seed=42,
    )


def validate_policy_in_isolation(
    policy_spec: PolicySpec,
    include_files: list[Path],
    console: Console,
    setup_script: str | None = None,
) -> bool:
    """Validate policy works in isolated environment.

    Creates a temp environment with published cogames package,
    copies include-files, and runs cogames eval for 1 episode with max 10 steps.

    Returns True if validation succeeds, False otherwise.
    """

    def _format_policy_arg(spec: PolicySpec) -> str:
        parts = [f"class={spec.class_path}"]
        if spec.data_path:
            parts.append(f"data={spec.data_path}")
        for key, value in spec.init_kwargs.items():
            parts.append(f"kw.{key}={value}")
        return ",".join(parts)

    console.print("[dim]Testing policy can run 10 steps on machina_1...[/dim]")

    temp_dir = None
    try:
        temp_dir = create_temp_validation_env()
        copy_files_maintaining_structure(include_files, temp_dir)

        policy_arg = _format_policy_arg(policy_spec)

        def _run_from_tmp_dir(cmd: list[str]) -> subprocess.CompletedProcess:
            env = os.environ.copy()
            env["UV_NO_CACHE"] = "1"
            res = subprocess.run(
                cmd,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                env=env,
            )
            if not res.returncode == 0:
                console.print("[red]Validation failed[/red]")
                console.print(f"\n[red]Error:[/red]\n{res.stderr}")
                if res.stdout:
                    console.print(f"\n[dim]Output:[/dim]\n{res.stdout}")
                raise Exception("Validation failed")
            return res

        _run_from_tmp_dir(["uv", "run", "cogames", "version"])

        validate_cmd = [
            "uv",
            "run",
            "cogames",
            "validate-policy",
            policy_arg,
        ]
        if setup_script:
            validate_cmd.extend(["--setup-script", setup_script])

        _run_from_tmp_dir(validate_cmd)

        console.print("[green]Validation passed[/green]")
        return True

    except subprocess.TimeoutExpired:
        console.print("[red]Validation timed out after 5 minutes[/red]")
        return False
    except Exception:
        return False
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)


def create_submission_zip(
    include_files: list[Path],
    policy_spec: PolicySpec,
    setup_script: str | None = None,
) -> Path:
    """Create a zip file containing all include-files.

    Maintains directory structure exactly as provided.
    Returns path to created zip file.
    """
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cogames_submission_")
    os.close(zip_fd)

    submission_spec = SubmissionPolicySpec(
        class_path=policy_spec.class_path,
        data_path=policy_spec.data_path,
        init_kwargs=policy_spec.init_kwargs,
        setup_script=setup_script,
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr(data=submission_spec.model_dump_json(), zinfo_or_arcname=POLICY_SPEC_FILENAME)

        for file_path in include_files:
            if file_path.is_dir():
                for root, _, files in os.walk(file_path):
                    for file in files:
                        file_full_path = Path(root) / file
                        zipf.write(file_full_path, arcname=file_full_path)
            else:
                zipf.write(file_path, arcname=file_path)

    return Path(zip_path)


def upload_submission(
    client: TournamentServerClient,
    zip_path: Path,
    submission_name: str,
    console: Console,
) -> UploadResult | None:
    """Upload submission to CoGames backend using a presigned S3 URL."""
    console.print("[bold]Uploading[/bold]")

    try:
        presigned_data = client.get_presigned_upload_url()
        upload_url = presigned_data.get("upload_url")
        upload_id = presigned_data.get("upload_id")

        if not upload_url or not upload_id:
            console.print("[red]Upload URL missing from response[/red]")
            return None
    except httpx.TimeoutException:
        console.print("[red]Timed out while requesting upload URL[/red]")
        return None
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]Failed to get upload URL ({exc.response.status_code})[/red]")
        console.print(f"[dim]{exc.response.text}[/dim]")
        return None
    except Exception as e:
        console.print(f"[red]Error requesting upload URL: {e}[/red]")
        return None

    console.print("[dim]Uploading to storage...[/dim]")

    try:
        with open(zip_path, "rb") as f:
            upload_response = httpx.put(
                upload_url,
                content=f,
                headers={"Content-Type": "application/zip"},
                timeout=600.0,
            )
        upload_response.raise_for_status()
    except httpx.TimeoutException:
        console.print("[red]Upload timed out after 10 minutes[/red]")
        return None
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]Upload failed with status {exc.response.status_code}[/red]")
        console.print(f"[dim]{exc.response.text}[/dim]")
        return None
    except Exception as e:
        console.print(f"[red]Upload error: {e}[/red]")
        return None

    console.print("[dim]Registering policy...[/dim]")

    try:
        result = client.complete_policy_upload(upload_id, submission_name)
        submission_id = result.get("id")
        name = result.get("name")
        version = result.get("version")
        if submission_id is not None and name is not None and version is not None:
            try:
                return UploadResult(
                    policy_version_id=uuid.UUID(str(submission_id)),
                    name=name,
                    version=version,
                )
            except ValueError:
                console.print(f"[red]Invalid submission ID returned: {submission_id}[/red]")
                return None

        console.print("[red]Missing fields in response[/red]")
        return None
    except httpx.TimeoutException:
        console.print("[red]Registration timed out[/red]")
        return None
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]Registration failed with status {exc.response.status_code}[/red]")
        console.print(f"[dim]{exc.response.text}[/dim]")
        return None
    except Exception as e:
        console.print(f"[red]Registration error: {e}[/red]")
        return None


def upload_policy(
    ctx: typer.Context,
    policy: str,
    name: str,
    include_files: list[str] | None = None,
    login_server: str = DEFAULT_COGAMES_SERVER,
    server: str = DEFAULT_SUBMIT_SERVER,
    init_kwargs: dict[str, str] | None = None,
    dry_run: bool = False,
    skip_validation: bool = False,
    setup_script: str | None = None,
) -> UploadResult | None:
    """Upload a policy to CoGames (without submitting to a tournament).

    Returns UploadResult with policy_version_id, name, and version on success.
    Returns None on failure.
    """
    from cogames.cli.client import TournamentServerClient

    if dry_run:
        console.print("[dim]Dry run mode - no upload[/dim]\n")

    client = TournamentServerClient.from_login(server_url=server, login_server=login_server)
    if not client:
        return None

    try:
        policy_spec = get_policy_spec(ctx, policy)
    except Exception as e:
        console.print(f"[red]Error parsing policy:[/red] {e}")
        return None

    if init_kwargs:
        merged_kwargs = {**policy_spec.init_kwargs, **init_kwargs}
        policy_spec = PolicySpec(
            class_path=policy_spec.class_path,
            data_path=policy_spec.data_path,
            init_kwargs=merged_kwargs,
        )

    files_to_include = []
    if policy_spec.data_path:
        files_to_include.append(policy_spec.data_path)
    if setup_script:
        files_to_include.append(setup_script)
    if include_files:
        files_to_include.extend(include_files)

    validated_paths: list[Path] = []
    if files_to_include:
        try:
            validated_paths = validate_paths(files_to_include, console)
        except (ValueError, FileNotFoundError):
            return None

    if not skip_validation:
        if not validate_policy_in_isolation(policy_spec, validated_paths, console, setup_script=setup_script):
            console.print("\n[red]Upload aborted due to validation failure.[/red]")
            return None
    else:
        console.print("[dim]Skipping validation[/dim]")

    try:
        zip_path = create_submission_zip(validated_paths, policy_spec, setup_script=setup_script)
    except Exception as e:
        console.print(f"[red]Error creating zip:[/red] {e}")
        return None

    if dry_run:
        console.print("[green]Dry run complete[/green]")
        if zip_path.exists():
            zip_path.unlink()
        return None

    try:
        with client:
            result = upload_submission(client, zip_path, name, console)
        if not result:
            console.print("\n[red]Upload failed.[/red]")
            return None
        return result
    finally:
        if zip_path.exists():
            zip_path.unlink()
