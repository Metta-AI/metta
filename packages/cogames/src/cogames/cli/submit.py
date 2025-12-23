"""Policy submission command for CoGames."""

import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path

import httpx
import typer
from rich.console import Console

from cogames.cli.base import console
from cogames.cli.login import DEFAULT_COGAMES_SERVER, CoGamesAuthenticator
from cogames.cli.policy import PolicySpec, get_policy_spec
from mettagrid.policy.checkpoint_policy import CheckpointPolicy
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.submission import POLICY_SPEC_FILENAME, SubmissionPolicySpec

DEFAULT_SUBMIT_SERVER = "https://api.observatory.softmax-research.net"
SUBMISSION_TAGS = {"cogames-submitted": "true"}


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


def create_temp_validation_env(console: Console) -> Path:
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

    console.print(f"[dim]Created validation environment: {temp_dir}[/dim]")
    return temp_dir


def copy_files_maintaining_structure(files: list[Path], dest_dir: Path, console: Console) -> None:
    """Copy files to destination, maintaining directory structure.

    If a file is 'train_dir/model.pt', it will be copied to 'dest_dir/train_dir/model.pt'.
    """
    for file_path in files:
        dest_path = dest_dir / file_path

        if file_path.is_dir():
            # Copy entire directory
            console.print(f"[dim]Copying directory: {file_path}[/dim]")
            shutil.copytree(file_path, dest_path, dirs_exist_ok=True)
        else:
            # Copy single file
            console.print(f"[dim]Copying file: {file_path}[/dim]")
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
    policy = (
        CheckpointPolicy.from_policy_spec(policy_env_info, policy_spec).wrapped_policy
        if policy_spec.class_path == CheckpointPolicy.CLASS_PATH
        else initialize_or_load_policy(policy_env_info, policy_spec)
    )

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

    temp_dir = None
    try:
        # Create temp validation environment
        temp_dir = create_temp_validation_env(console)

        # Copy include files maintaining structure
        console.print("[yellow]Copying files to validation environment...[/yellow]")
        copy_files_maintaining_structure(include_files, temp_dir, console)

        # Build cogames eval command
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
                console.print("[red]✗ Setting up validation environment failed![/red]")
                console.print("\n[red]Error output:[/red]")
                console.print(res.stderr)
                if res.stdout:
                    console.print("\n[dim]Standard output:[/dim]")
                    console.print(res.stdout)
                raise Exception("Setting up validation environment failed")
            return res

        console.print("[yellow] Validating policy...[/yellow]")
        result = _run_from_tmp_dir(["uv", "run", "cogames", "version"])
        console.print(f"[dim]Cogames version: {result.stdout.strip()}[/dim]")

        validate_cmd = [
            "uv",
            "run",
            "cogames",
            "validate-policy",
            policy_arg,
        ]
        if setup_script:
            validate_cmd.extend(["--setup-script", setup_script])

        result = _run_from_tmp_dir(validate_cmd)

        console.print(f"[dim]Validation result: {result.stdout.strip()}[/dim]")

        console.print("[green]✓ Validation passed![/green]")
        return True

    except subprocess.TimeoutExpired:
        console.print("[red]✗ Validation timed out after 5 minutes[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Validation error: {e}[/red]")
        return False
    finally:
        # Clean up temp directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
            console.print("[dim]Cleaned up validation environment[/dim]")


def create_submission_zip(
    include_files: list[Path],
    policy_spec: PolicySpec,
    console: Console,
    setup_script: str | None = None,
) -> Path:
    """Create a zip file containing all include-files.

    Maintains directory structure exactly as provided.
    Returns path to created zip file.
    """
    # Create temp zip file
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cogames_submission_")
    os.close(zip_fd)

    console.print("[yellow]Creating submission zip...[/yellow]")

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
                # Add all files in directory recursively
                for root, _, files in os.walk(file_path):
                    for file in files:
                        file_full_path = Path(root) / file
                        arcname = file_full_path
                        console.print(f"[dim]  Adding: {arcname}[/dim]")
                        zipf.write(file_full_path, arcname=arcname)
            else:
                # Add single file
                console.print(f"[dim]  Adding: {file_path}[/dim]")
                zipf.write(file_path, arcname=file_path)

    console.print(f"[green]✓ Created zip: {zip_path}[/green]")
    return Path(zip_path)


def upload_submission(
    zip_path: Path,
    submission_name: str,
    token: str,
    submit_server_url: str,
    console: Console,
) -> uuid.UUID | None:
    """Upload submission to CoGames backend using a presigned S3 URL."""
    console.print("[yellow]Uploading submission...[/yellow]")

    headers = {"X-Auth-Token": token}

    try:
        presigned_response = httpx.post(
            f"{submit_server_url}/stats/policies/submit/presigned-url",
            headers=headers,
            timeout=60.0,
        )

        if presigned_response.status_code != 200:
            console.print(f"[red]✗ Failed to get upload URL ({presigned_response.status_code})[/red]")
            console.print(f"[red]Response: {presigned_response.text}[/red]")
            return None

        presigned_data = presigned_response.json()
        upload_url = presigned_data.get("upload_url")
        upload_id = presigned_data.get("upload_id")

        if not upload_url or not upload_id:
            console.print("[red]✗ Upload URL missing from response[/red]")
            return None
    except httpx.TimeoutException:
        console.print("[red]✗ Timed out while requesting presigned URL[/red]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Error requesting presigned URL: {e}[/red]")
        return None

    console.print(f"[dim]  Upload ID: {upload_id}[/dim]")

    try:
        with open(zip_path, "rb") as f:
            upload_response = httpx.put(
                upload_url,
                content=f,
                headers={"Content-Type": "application/zip"},
                timeout=600.0,  # Allow large uploads
            )
        upload_response.raise_for_status()
    except httpx.TimeoutException:
        console.print("[red]✗ Upload timed out after 10 minutes[/red]")
        return None
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]✗ Upload failed with status {exc.response.status_code}[/red]")
        console.print(f"[red]Response: {exc.response.text}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Upload error: {e}[/red]")
        return None

    console.print("[dim]  Uploaded successfully. Registering...[/dim]")

    try:
        complete_response = httpx.post(
            f"{submit_server_url}/stats/policies/submit/complete",
            json={"upload_id": upload_id, "name": submission_name},
            headers=headers,
            timeout=120.0,
        )

        if complete_response.status_code == 200:
            result = complete_response.json()
            console.print("[green]✓ Submitted successfully![/green]")
            submission_id = result.get("id") or result.get("submission_id")
            if submission_id:
                try:
                    policy_version_id = uuid.UUID(str(submission_id))
                    console.print(f"[dim]Submission ID: {policy_version_id}[/dim]")
                    return policy_version_id
                except ValueError:
                    console.print(f"[red]✗ Invalid submission ID returned: {submission_id}[/red]")
                    return None

            console.print("[red]✗ Submission ID missing from response[/red]")
            return None

        console.print(f"[red]✗ Submission finalize failed with status {complete_response.status_code}[/red]")
        console.print(f"[red]Response: {complete_response.text}[/red]")
        return None

    except httpx.TimeoutException:
        console.print("[red]✗ Finalizing submission timed out[/red]")
        return None
    except Exception as e:
        console.print(f"[red]✗ Submission finalize error: {e}[/red]")
        return None


def update_submission_tags(policy_version_id: uuid.UUID, token: str, submit_server_url: str, console: Console) -> bool:
    """Upsert submission tags on the policy version."""
    console.print(f"[yellow]Updating submission tags for {policy_version_id}...[/yellow]")

    try:
        response = httpx.put(
            f"{submit_server_url}/stats/policies/versions/{policy_version_id}/tags",
            json=SUBMISSION_TAGS,
            headers={"X-Auth-Token": token},
            timeout=60.0,
        )

        if response.status_code == 200:
            console.print("[green]✓ Tags updated successfully[/green]")
            return True

        console.print(f"[red]✗ Failed to update tags ({response.status_code})[/red]")
        console.print(f"[red]Response: {response.text}[/red]")
        return False

    except httpx.TimeoutException:
        console.print("[red]✗ Tag update timed out[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Tag update error: {e}[/red]")
        return False


def submit_command(
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
) -> None:
    """Submit a policy to CoGames competitions.

    This command:
    1. Validates authentication
    2. Tests the policy in an isolated environment (unless --skip-validation)
    3. Creates a submission zip with included files
    4. Uploads to the backend API

    Args:
        ctx: Typer context
        policy: Policy specification as comma-separated key=value pairs
        name: Optional name for the submission
        include_files: List of files/directories to include in submission
        login_server: Login/authentication server URL
        server: Submission server URL
        dry_run: If True, run validation only without submitting
        skip_validation: If True, skip policy validation in isolated environment
        setup_script: Optional path to a Python setup script to run before loading the policy
    """
    if dry_run:
        console.print("[bold]CoGames Policy Submission (DRY RUN)[/bold]\n")
        console.print("[yellow]Running in dry-run mode - validation only, no submission[/yellow]\n")
    else:
        console.print("[bold]CoGames Policy Submission[/bold]\n")

    if skip_validation:
        console.print("[yellow]⚠ Skipping policy validation (--skip-validation)[/yellow]\n")

    # Check authentication first
    authenticator = CoGamesAuthenticator()
    if not authenticator.has_saved_token(login_server):
        console.print("[red]Error:[/red] Not authenticated.")
        console.print("Please run: [cyan]cogames login[/cyan]")
        return

    token = authenticator.load_token(login_server)
    if not token:
        console.print(f"[red]Error:[/red] Token not found for {login_server}")
        return

    # Parse policy spec
    try:
        policy_spec = get_policy_spec(ctx, policy)
    except Exception as e:
        console.print(f"[red]Error parsing policy:[/red] {e}")
        return

    # Merge init_kwargs into policy_spec
    if init_kwargs:
        merged_kwargs = {**policy_spec.init_kwargs, **init_kwargs}
        policy_spec = PolicySpec(
            class_path=policy_spec.class_path,
            data_path=policy_spec.data_path,
            init_kwargs=merged_kwargs,
        )
        console.print("\n[bold]Policy init_kwargs:[/bold]")
        for key, value in init_kwargs.items():
            console.print(f"  {key}: {value}")
        console.print()

    # Validate and collect all files to include
    files_to_include = []

    # Always include policy data file if specified
    if policy_spec.data_path:
        files_to_include.append(policy_spec.data_path)

    # Include setup script if specified
    if setup_script:
        files_to_include.append(setup_script)

    # Add user-specified include files
    if include_files:
        files_to_include.extend(include_files)

    # Validate all paths
    validated_paths: list[Path] = []
    if files_to_include:
        try:
            validated_paths = validate_paths(files_to_include, console)
        except (ValueError, FileNotFoundError):
            return

        console.print("\n[bold]Files to include:[/bold]")
        for path in validated_paths:
            console.print(f"  • {path}")
        console.print()

    # Validate policy in isolated environment (unless skipped)
    if not skip_validation:
        if not validate_policy_in_isolation(policy_spec, validated_paths, console, setup_script=setup_script):
            console.print("\n[red]Submission aborted due to validation failure.[/red]")
            return
    else:
        console.print("[yellow]⚠ Policy validation skipped[/yellow]")

    # Create submission zip
    try:
        zip_path = create_submission_zip(validated_paths, policy_spec, console, setup_script=setup_script)
    except Exception as e:
        console.print(f"[red]Error creating zip:[/red] {e}")
        return

    # If dry-run, skip upload and clean up
    if dry_run:
        if skip_validation:
            console.print("\n[green]✓ Dry run complete - validation skipped, zip created![/green]")
        else:
            console.print("\n[green]✓ Dry run complete - validation passed, zip created![/green]")
        console.print("[dim]Skipping upload (dry-run mode)[/dim]")
        # Clean up zip file in dry-run mode
        if zip_path.exists():
            zip_path.unlink()
            console.print("[dim]Cleaned up temporary zip file[/dim]")
        return

    # Upload submission
    try:
        policy_version_id = upload_submission(zip_path, name, token, server, console)
        if not policy_version_id:
            console.print("\n[red]Submission failed.[/red]")
            return

        tag_success = update_submission_tags(policy_version_id, token, server, console)
        if not tag_success:
            console.print("\n[red]Submission succeeded, but updating tags failed.[/red]")
    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
            console.print("[dim]Cleaned up temporary zip file[/dim]")
