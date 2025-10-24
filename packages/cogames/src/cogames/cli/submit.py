"""Policy submission command for CoGames."""

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import httpx
import typer
from rich.console import Console

from cogames.cli.base import console
from cogames.cli.login import DEFAULT_COGAMES_SERVER, CoGamesAuthenticator
from cogames.cli.policy import PolicySpec, get_policy_spec

DEFAULT_SUBMIT_SERVER = "https://api.observatory.softmax-research.net"
DEFAULT_VALIDATION_MISSION = "vanilla"
VALIDATION_EPISODES = 1
VALIDATION_MAX_STEPS = 10


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


def create_temp_validation_env(console: Console) -> Path:
    """Create a temporary directory with a minimal pyproject.toml.

    The pyproject.toml depends on the latest published cogames package.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="cogames_submit_"))

    pyproject_content = """[project]
name = "cogames-submission-validator"
version = "0.1.0"
requires-python = ">=3.11.7"
dependencies = ["cogames"]

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


def validate_policy_in_isolation(
    policy_spec: PolicySpec,
    include_files: list[Path],
    console: Console,
) -> bool:
    """Validate policy works in isolated environment.

    Creates a temp environment with published cogames package,
    copies include-files, and runs cogames eval for 1 episode with max 10 steps.

    Returns True if validation succeeds, False otherwise.
    """
    temp_dir = None
    try:
        # Create temp validation environment
        temp_dir = create_temp_validation_env(console)

        # Copy include files maintaining structure
        console.print("[yellow]Copying files to validation environment...[/yellow]")
        copy_files_maintaining_structure(include_files, temp_dir, console)

        # Build cogames eval command
        # Policy spec format: CLASS:DATA:PROPORTION
        policy_arg = policy_spec.policy_class_path
        if policy_spec.policy_data_path:
            policy_arg += f":{policy_spec.policy_data_path}"

        console.print(f"[yellow]Running validation with mission '{DEFAULT_VALIDATION_MISSION}'...[/yellow]")

        cmd = [
            "uv",
            "run",
            "cogames",
            "eval",
            "--mission",
            DEFAULT_VALIDATION_MISSION,
            "--policy",
            policy_arg,
            "--episodes",
            str(VALIDATION_EPISODES),
        ]

        # Run in temp directory
        result = subprocess.run(
            cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            console.print("[red]✗ Validation failed![/red]")
            console.print("\n[red]Error output:[/red]")
            console.print(result.stderr)
            if result.stdout:
                console.print("\n[dim]Standard output:[/dim]")
                console.print(result.stdout)
            return False

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


def create_submission_zip(include_files: list[Path], console: Console) -> Path:
    """Create a zip file containing all include-files.

    Maintains directory structure exactly as provided.
    Returns path to created zip file.
    """
    # Create temp zip file
    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="cogames_submission_")
    os.close(zip_fd)

    console.print("[yellow]Creating submission zip...[/yellow]")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
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
    submission_name: str | None,
    login_server_url: str,
    submit_server_url: str,
    console: Console,
) -> bool:
    """Upload submission to CoGames backend.

    Reads auth token from login server and POSTs to submit server's /cogames/submit_policy endpoint.
    Returns True on success, False otherwise.
    """
    # Get auth token from login server
    authenticator = CoGamesAuthenticator(login_server_url)
    if not authenticator.has_saved_token():
        console.print("[red]Error:[/red] Not authenticated. Please run: [cyan]cogames login[/cyan]")
        return False

    # Load token
    token = authenticator.load_token()
    if not token:
        console.print(f"[red]Error:[/red] Token not found for {login_server_url}")
        return False

    # Prepare multipart form data
    console.print(f"[yellow]Uploading submission to {submit_server_url}...[/yellow]")

    try:
        with open(zip_path, "rb") as f:
            files = {"file": ("submission.zip", f, "application/zip")}
            data = {}
            if submission_name:
                data["name"] = submission_name

            headers = {"X-Auth-Token": token}

            response = httpx.post(
                f"{submit_server_url}/cogames/submit_policy",
                files=files,
                data=data,
                headers=headers,
                timeout=300.0,  # 5 minute timeout for upload
            )

        if response.status_code == 200:
            result = response.json()
            console.print("[green]✓ Submitted successfully![/green]")
            if "submission_id" in result:
                console.print(f"[dim]Submission ID: {result['submission_id']}[/dim]")
            return True
        else:
            console.print(f"[red]✗ Upload failed with status {response.status_code}[/red]")
            console.print(f"[red]Response: {response.text}[/red]")
            return False

    except httpx.TimeoutException:
        console.print("[red]✗ Upload timed out after 5 minutes[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Upload error: {e}[/red]")
        return False


def submit_command(
    ctx: typer.Context,
    policy: str,
    name: str | None = None,
    include_files: list[str] | None = None,
    login_server: str = DEFAULT_COGAMES_SERVER,
    server: str = DEFAULT_SUBMIT_SERVER,
    dry_run: bool = False,
    skip_validation: bool = False,
) -> None:
    """Submit a policy to CoGames competitions.

    This command:
    1. Validates authentication
    2. Tests the policy in an isolated environment (unless --skip-validation)
    3. Creates a submission zip with included files
    4. Uploads to the backend API

    Args:
        ctx: Typer context
        policy: Policy specification in format CLASS[:DATA[:PROPORTION]]
        name: Optional name for the submission
        include_files: List of files/directories to include in submission
        login_server: Login/authentication server URL
        server: Submission server URL
        dry_run: If True, run validation only without submitting
        skip_validation: If True, skip policy validation in isolated environment
    """
    if dry_run:
        console.print("[bold]CoGames Policy Submission (DRY RUN)[/bold]\n")
        console.print("[yellow]Running in dry-run mode - validation only, no submission[/yellow]\n")
    else:
        console.print("[bold]CoGames Policy Submission[/bold]\n")

    if skip_validation:
        console.print("[yellow]⚠ Skipping policy validation (--skip-validation)[/yellow]\n")

    # Check authentication first
    authenticator = CoGamesAuthenticator(login_server)
    if not authenticator.has_saved_token():
        console.print("[red]Error:[/red] Not authenticated.")
        console.print("Please run: [cyan]cogames login[/cyan]")
        return

    # Parse policy spec
    try:
        policy_spec = get_policy_spec(ctx, policy)
    except Exception as e:
        console.print(f"[red]Error parsing policy:[/red] {e}")
        return

    # Validate and collect all files to include
    files_to_include = []

    # Always include policy data file if specified
    if policy_spec.policy_data_path:
        files_to_include.append(policy_spec.policy_data_path)

    # Add user-specified include files
    if include_files:
        files_to_include.extend(include_files)

    if not files_to_include:
        console.print(
            "[red]Error:[/red] No files to include. Please specify --include-files or provide a policy checkpoint path."
        )
        return

    # Validate all paths
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
        if not validate_policy_in_isolation(policy_spec, validated_paths, console):
            console.print("\n[red]Submission aborted due to validation failure.[/red]")
            return
    else:
        console.print("[yellow]⚠ Policy validation skipped[/yellow]")

    # Create submission zip
    try:
        zip_path = create_submission_zip(validated_paths, console)
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
        success = upload_submission(zip_path, name, login_server, server, console)
        if not success:
            console.print("\n[red]Submission failed.[/red]")
    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()
            console.print("[dim]Cleaned up temporary zip file[/dim]")
