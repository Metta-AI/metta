import re
import subprocess
from enum import StrEnum
from typing import Annotated, Optional

import typer

import gitta as git
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from metta.common.util.discord import send_to_discord
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, info, success, warning
from softmax.aws.secrets_manager import get_secretsmanager_secret

VERSION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+(?:\.\d+)?)$")
DEFAULT_INITIAL_VERSION = "0.0.0.1"
EXPECTED_REMOTE_URL = f"git@github.com:{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}.git"
DISCORD_CHANNEL_WEBHOOK_URL_SECRET_NAME = "discord/channel-webhook/updates"


class Package(StrEnum):
    COGAMES = "cogames"
    METTAGRID = "mettagrid"


_RELEASE_WORKFLOW_URL_FOR_PACKAGE = {
    Package.COGAMES: "https://github.com/Metta-AI/metta/actions/workflows/release-cogames.yml",
    Package.METTAGRID: "https://github.com/Metta-AI/metta/actions/workflows/release-mettagrid.yml",
}


def _get_metta_remote() -> str:
    try:
        remotes_output = git.run_git("remote", "-v")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to list remotes: {exc}")
        raise typer.Exit(exc.returncode) from exc

    for line in remotes_output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            remote_name, remote_url = parts[0], parts[1]
            if METTA_GITHUB_ORGANIZATION in remote_url and METTA_GITHUB_REPO in remote_url:
                return remote_name

    error(f"No remote found pointing to {METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}")
    info("Available remotes:")
    for line in remotes_output.splitlines():
        info(f"  {line}")
    raise typer.Exit(1)


def _bump_version(version: str) -> str:
    parts = version.split(".")
    bumped = parts[:-1] + [str(int(parts[-1]) + 1)]
    return ".".join(bumped)


def _validate_version_format(version: str) -> None:
    if not VERSION_PATTERN.match(version):
        error(f"Invalid version '{version}'. Expected numeric segments like '1.2.3' or '1.2.3.4'.")
        raise typer.Exit(1)


def _ensure_tag_unique(package: str, version: str) -> None:
    tag_name = f"{package}-v{version}"
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag_name}"],
        cwd=get_repo_root(),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        error(f"Tag '{tag_name}' already exists.")
        raise typer.Exit(1)


def _push_child_repo(package: str, dry_run: bool) -> None:
    if dry_run:
        info(f"  [dry-run] Would push {package} to child repo")
        return
    info(f"Pushing {package} to child repo...")
    try:
        subprocess.run(
            [f"{get_repo_root()}/devops/git/push_child_repo.py", package, "-y"],
            check=True,
        )
        success(f"Pushed {package} to child repo.")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to push child repo: {exc}")
        raise typer.Exit(exc.returncode) from exc


def _create_and_push_tag(package: str, version: str, remote: str, dry_run: bool) -> str:
    tag_name = f"{package}-v{version}"
    if dry_run:
        info(f"  [dry-run] Would create and push tag {tag_name}")
        return tag_name
    info(f"Tagging {package} {version}...")
    git.run_git("tag", "-a", tag_name, "-m", f"Release {package} {version}")
    git.run_git("push", remote, tag_name)
    success(f"Published {tag_name} to {remote}.")
    return tag_name


def _get_next_version(package: str, version_override: Optional[str], remote: str) -> tuple[str, Optional[str]]:
    prefix = f"{package}-v"

    try:
        info(f"Fetching tags from {remote}...")
        git.run_git("fetch", remote, "--tags", "--force")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to fetch tags from {remote}: {exc}")
        raise typer.Exit(exc.returncode) from exc

    try:
        tag_list_output = git.run_git("tag", "--list", f"{prefix}*", "--sort=-v:refname")
    except subprocess.CalledProcessError:
        tag_list_output = ""

    tags = [line for line in tag_list_output.splitlines() if line.strip()]
    latest_tag = tags[0] if tags else None

    if version_override is None:
        if latest_tag:
            previous_version = latest_tag[len(prefix) :]
            _validate_version_format(previous_version)
            target_version = _bump_version(previous_version)
        else:
            target_version = DEFAULT_INITIAL_VERSION
    else:
        _validate_version_format(version_override)
        target_version = version_override

    _validate_version_format(target_version)
    _ensure_tag_unique(package, target_version)

    return target_version, latest_tag


def _check_git_state(force: bool) -> tuple[str, str]:
    try:
        status_output = git.run_git("status", "--porcelain")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to read git status: {exc}")
        raise typer.Exit(exc.returncode) from exc

    if status_output.strip() and not force:
        error("Working tree is not clean. Commit, stash, or clean changes before publishing (use --force to override).")
        raise typer.Exit(1)

    try:
        current_branch = git.run_git("rev-parse", "--abbrev-ref", "HEAD")
        current_commit = git.run_git("rev-parse", "HEAD")
    except subprocess.CalledProcessError as exc:
        error(f"Failed to determine git state: {exc}")
        raise typer.Exit(exc.returncode) from exc

    if current_branch not in {"main"} and not force:
        error("Publishing is only supported from the main branch. Switch to 'main' or pass --force to override.")
        raise typer.Exit(1)

    return current_branch, current_commit


def _post_to_discord(
    package: Package,
    version: str,
    tag_name: str,
    commit: str,
    dry_run: bool,
) -> None:
    # Get webhook URL from AWS Secrets Manager.
    webhook_url = get_secretsmanager_secret(DISCORD_CHANNEL_WEBHOOK_URL_SECRET_NAME, require_exists=False)
    if not webhook_url:
        warning("Discord webhook URL not configured")
        return

    # Validate webhook URL format
    if not webhook_url.startswith("https://discord.com/api/webhooks/"):
        warning("Discord webhook URL doesn't match expected format. Skipping Discord notification.")
        return

    # Format release message
    tag_url = f"https://github.com/{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}/releases/tag/{tag_name}"
    commit_url = f"https://github.com/{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}/commit/{commit}"

    message = f"🚀 **{package.value} v{version}** released!\n\n"
    message += f"Tag: `{tag_name}`\n"
    message += f"Commit: [{commit[:7]}]({commit_url})\n"
    message += f"Release: [View on GitHub]({tag_url})\n"

    if release_workflow_url := _RELEASE_WORKFLOW_URL_FOR_PACKAGE.get(package):
        message += f"\n📦 Triggered [release workflow]({release_workflow_url}) to publish to PyPi (takes ~5-10 min)."

    if dry_run:
        info("Would post following message to Discord:")
        print(message)
        return

    # Actually post to Discord.
    info("Posting release announcement to Discord...")
    success_flag = send_to_discord(webhook_url, message, suppress_embeds=True)

    if not success_flag:
        warning("Failed to post to Discord.")
        return

    success("Posted release announcement to Discord.")


def cmd_publish(
    package: Annotated[Package, typer.Argument(help="Package to publish")],
    version: Annotated[
        Optional[str],
        typer.Option("--version", "-v", help="Explicit version to tag (digits separated by dots)"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview actions without executing")] = False,
    repo_only: Annotated[bool, typer.Option("--repo-only", help="Only push to child repo")] = False,
    tag_only: Annotated[bool, typer.Option("--tag-only", help="Only create/push tag (skip child repo)")] = False,
    force: Annotated[bool, typer.Option("--force", help="Bypass branch and clean checks")] = False,
):
    if repo_only and tag_only:
        error("Cannot use --repo-only and --tag-only together")
        raise typer.Exit(1)

    remote = _get_metta_remote()

    _publish(
        package=package,
        version=version,
        dry_run=dry_run,
        repo_only=repo_only,
        tag_only=tag_only,
        remote=remote,
        force=force,
    )


def _publish(
    package: Package,
    version: Optional[str],
    dry_run: bool,
    repo_only: bool,
    tag_only: bool,
    remote: str,
    force: bool,
) -> None:
    current_branch, current_commit = _check_git_state(force)

    tag_name = None
    target_version = None
    latest_tag = None
    if not repo_only:
        target_version, latest_tag = _get_next_version(package, version, remote)
        tag_name = f"{package}-v{target_version}"

    info("Release summary:\n")
    info(f"  Package: {package}")
    info(f"  Branch: {current_branch}")
    info(f"  Commit: {current_commit}")
    if not repo_only:
        info(f"  Tag: {tag_name}")
        info(f"  Previous tag: {latest_tag or 'none'}")

    if force:
        warning("Force mode enabled: branch and clean checks were bypassed.")
    info("")

    if package == Package.COGAMES:
        if typer.confirm("Cogames depends on mettagrid. Publish mettagrid first?", default=True):
            info("")
            info("Starting mettagrid publish flow...")
            _publish(
                package=Package.METTAGRID,
                version=None,
                dry_run=dry_run,
                repo_only=repo_only,
                tag_only=tag_only,
                remote=remote,
                force=force,
            )
            info("")
            info("Continuing with cogames publish...")

    if not dry_run and not typer.confirm("Proceed?", default=True):
        info("Publishing aborted.")
        return

    if not repo_only:
        assert target_version is not None
        tag_name = _create_and_push_tag(package, target_version, remote, dry_run)

        # Post to Discord after successful tag creation
        try:
            _post_to_discord(
                package=package,
                version=target_version,
                tag_name=tag_name,
                commit=current_commit,
                dry_run=dry_run,
            )
        except Exception as exc:
            warning(f"Failed to post to Discord: {exc}")

    if not tag_only:
        _push_child_repo(package, dry_run)

    if dry_run:
        success("Dry run complete.")
