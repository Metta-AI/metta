import re
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional

import typer

import gitta
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from metta.common.util.discord import send_to_discord
from metta.common.util.fs import get_repo_root
from metta.setup.utils import error, header, info, success, warning
from softmax.aws.secrets_manager import get_secretsmanager_secret

VERSION_PATTERN = re.compile(r"^(\d+\.\d+\.\d+(?:\.\d+)?)$")
DEFAULT_INITIAL_VERSION = "0.0.0.1"

DISCORD_CHANNEL_WEBHOOK_URL_SECRET_NAME = "discord/channel-webhook/updates"


class Package(StrEnum):
    COGAMES = "cogames"
    METTAGRID = "mettagrid"


_RELEASE_WORKFLOW_URL_FOR_PACKAGE = {
    Package.COGAMES: "https://github.com/Metta-AI/metta/actions/workflows/release-cogames.yml",
    Package.METTAGRID: "https://github.com/Metta-AI/metta/actions/workflows/release-mettagrid.yml",
}


def _is_working_tree_clean() -> bool:
    return not gitta.run_git("status", "--porcelain").strip()


def _is_on_main_branch() -> bool:
    return gitta.get_current_branch() == "main"


def _is_staging_area_clean() -> bool:
    # git diff --cached shows the changes staged for commit.
    # If there are no staged changes, the output will be empty.
    return not gitta.run_git("diff", "--cached").strip()


def _get_metta_repo_remote() -> str:
    matching_remote_names = [
        remote_name
        for remote_name, remote_url in gitta.get_all_remotes().items()
        if METTA_GITHUB_ORGANIZATION in remote_url and METTA_GITHUB_REPO in remote_url
    ]

    if len(matching_remote_names) == 0:
        error(f"No remote found pointing to {METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}")
        raise typer.Exit(1)
    elif len(matching_remote_names) > 1:
        error(f"Multiple remotes found pointing to {METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}")
        raise typer.Exit(1)

    return matching_remote_names[0]


def _tag_prefix_for_package(package: str, /) -> str:
    return f"{package}-v"


def _get_latest_version_for_package(package: str, /) -> Optional[str]:
    tag_prefix = _tag_prefix_for_package(package)
    tags = [
        line.strip()
        for line in gitta.run_git("tag", "--list", f"{tag_prefix}*", "--sort=-v:refname").splitlines()
        if line.strip()
    ]

    if len(tags) == 0:
        return None

    latest_tag = tags[0]
    latest_version = latest_tag[len(tag_prefix) :]

    if not VERSION_PATTERN.match(latest_version):
        error(f"Latest version '{latest_version}' does not match the expected format (3-4 dot-separated integers).")
        raise typer.Exit(1)

    return latest_version


def _bump_last_part_of_version(version: str, /) -> str:
    parts = version.split(".")
    bumped = parts[:-1] + [str(int(parts[-1]) + 1)]
    return ".".join(bumped)


def _get_next_version(*, package: str, version_override: Optional[str]) -> str:
    tag_prefix = _tag_prefix_for_package(package)

    if version_override is not None:
        if not VERSION_PATTERN.match(version_override):
            error(
                f"Requested version '{version_override}' does not match the expected format "
                "(3-4 dot-separated integers)."
            )
            raise typer.Exit(1)
        if gitta.resolve_git_ref(f"{tag_prefix}{version_override}"):
            error(f"Tag '{version_override}' already exists.")
            raise typer.Exit(1)

        return version_override

    else:
        latest_version = _get_latest_version_for_package(package)
        if latest_version is None:
            next_version = DEFAULT_INITIAL_VERSION
        else:
            next_version = _bump_last_part_of_version(latest_version)

        assert VERSION_PATTERN.match(next_version)
        assert not gitta.resolve_git_ref(f"{tag_prefix}{next_version}")

        return next_version


def _pin_dependency_version(*, package: str, dependency: str, version: str, dry_run: bool) -> None:
    pyproject_path = Path(get_repo_root()) / "packages" / package / "pyproject.toml"
    assert pyproject_path.exists()

    # Pattern to match dependency in the dependencies array.
    # Matches: "dependency", or "dependency>=X.Y.Z", or "dependency==X.Y.Z"
    pattern = rf'("{dependency}(?:[><=~!]+[\d.]+)?",)'
    replacement = f'"{dependency}=={version}",'

    old_content = pyproject_path.read_text()
    new_content = re.sub(pattern, replacement, old_content)

    assert re.search(pattern, old_content)
    assert new_content != old_content

    current_branch = gitta.get_current_branch()
    new_branch_name = f"chore/update-{package}-{dependency}-to-{version}"
    new_pr_title = f"chore: update {package} {dependency} to {version}"
    new_pr_body = "This PR was automatically created by `metta/setup/tools/publish.py`."
    new_commit_msg = new_pr_title

    if dry_run:
        info(f"Would pin {dependency} to =={version} in {package}/pyproject.toml")
        info(f"Would commit changes in new branch {new_branch_name}")
        info("Would put up PR to update cogames' mettagrid dependency.")
        return

    if not _is_staging_area_clean():
        error(f"Staging area is not clean. Can't safely write and commit changes to {package}/pyproject.toml")
        raise typer.Exit(1)

    if not typer.confirm(
        f"Okay to write changes to {package}/pyproject.toml and create branch {new_branch_name}?",
        default=True,
    ):
        error("Publishing aborted.")
        raise typer.Exit(1)

    info(f"Pinning dependency {dependency} to version {version} in {package}/pyproject.toml")
    pyproject_path.write_text(new_content)

    info(f"Creating new branch {new_branch_name}")
    gitta.run_git("checkout", "-b", new_branch_name)

    info(f"Staging changes to {package}/pyproject.toml")
    gitta.run_git("add", str(pyproject_path))

    info(f"Committing changes to {package}/pyproject.toml")
    gitta.run_git("commit", "-m", new_commit_msg)

    if typer.confirm(
        f"Automatically put up PR from {new_branch_name} to {current_branch}?",
        default=True,
    ):
        info("Pushing branch to remote...")
        gitta.run_git("push", "-u", "origin", new_branch_name)

        info(f"Putting up PR from {new_branch_name} to {current_branch}...")
        pr_url = gitta.run_gh(
            "pr",
            "create",
            "--title",
            new_pr_title,
            "--body",
            new_pr_body,
            "--head",
            new_branch_name,
            "--base",
            current_branch,
        )
        success(f"PR created: {pr_url}")
    else:
        warning(f"Skipping putting up PR from {new_branch_name} to {current_branch}; do it yourself later.")


def _create_and_push_tag_to_monorepo(*, package: str, version: str, remote: str, dry_run: bool) -> None:
    tag_prefix = _tag_prefix_for_package(package)
    tag = f"{tag_prefix}{version}"

    if dry_run:
        info(f"Would create and push tag {tag} to remote {remote}")
        return

    info(f"Tagging {package} {version}...")
    gitta.run_git("tag", "-a", tag, "-m", f"Release {package} {version}")
    gitta.run_git("push", remote, tag)
    success(f"Pushed tag {tag} to remote '{remote}'")


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


def _push_git_history_to_child_repo(*, package: str, dry_run: bool) -> None:
    dry_run_prefix = "[DRY RUN] " if dry_run else ""
    info(f"{dry_run_prefix}Pushing filtered git history for {package} to child repo...")

    cmd = [f"{get_repo_root()}/devops/git/push_child_repo.py", package, "-y"]
    if dry_run:
        cmd.append("--dry-run")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        error(f"Failed to push to child repo: {exc}")
        raise typer.Exit(exc.returncode) from exc
    else:
        success(f"{dry_run_prefix}Pushed filtered git history for {package} to child repo.")


def _publish(
    package: Package,
    version_override: Optional[str],
    remote: str,
    push_git_history_to_child_repo: bool,
    create_and_push_tag_to_monorepo: bool,
    dry_run: bool,
    skip_git_checks: bool,
) -> str | None:
    if not _is_working_tree_clean():
        if skip_git_checks:
            warning("Working tree is not clean. Bypassing this check due to the --force flag.")
        else:
            error(
                "Working tree is not clean. Commit, stash, or clean changes before publishing "
                "(use --force to override)."
            )
            raise typer.Exit(1)
    if not _is_on_main_branch():
        if skip_git_checks:
            warning("Not on the main branch. Bypassing this check due to the --force flag.")
        else:
            error("Publishing is only supported from the main branch. Switch to 'main' or pass --force to override.")
            raise typer.Exit(1)

    info(f"Refreshing tags from {remote}...")
    gitta.run_git("fetch", remote, "--tags", "--force")

    tag_prefix = _tag_prefix_for_package(package)
    next_version = (
        _get_next_version(package=package, version_override=version_override)
        if create_and_push_tag_to_monorepo
        else None
    )
    next_tag = f"{tag_prefix}{next_version}" if next_version else None

    info("--------------------------------")
    info("Release summary:\n")
    print()
    print(f"  Package: {package}")
    print(f"  Branch: {gitta.get_current_branch()}")
    print(f"  Commit: {gitta.get_current_commit()}")
    if create_and_push_tag_to_monorepo:
        print()
        if latest_version := _get_latest_version_for_package(package):
            print(f"  Previous tag: {tag_prefix}{latest_version}")
        else:
            print("  Previous tag: (none found)")
        print(f"  Next tag: {next_tag}")
    print()
    info("--------------------------------")
    info("")

    if create_and_push_tag_to_monorepo:
        assert next_version is not None
        assert next_tag is not None

        # For cogames, we must publish mettagrid first and pin the cogames dependency to that exact version.
        if package == Package.COGAMES:
            if not typer.confirm("To publish cogames, we must publish mettagrid first. Okay to proceed?", default=True):
                error("Publishing aborted.")
                raise typer.Exit(1)

            print()
            print()
            header("Publishing mettagrid first...")
            mettagrid_version = _publish(
                package=Package.METTAGRID,
                version_override=None,
                remote=remote,
                push_git_history_to_child_repo=push_git_history_to_child_repo,
                create_and_push_tag_to_monorepo=create_and_push_tag_to_monorepo,
                dry_run=dry_run,
                skip_git_checks=skip_git_checks,
            )
            assert mettagrid_version is not None
            header(f"Published mettagrid with version {mettagrid_version}.")
            print()
            print()
            header(f"Resuming with publishing {package}...")

            _pin_dependency_version(
                package="cogames",
                dependency="mettagrid",
                version=mettagrid_version,
                dry_run=dry_run,
            )

        _create_and_push_tag_to_monorepo(package=package, version=next_version, remote=remote, dry_run=dry_run)

        try:
            _post_to_discord(
                package=package,
                version=next_version,
                tag_name=next_tag,
                commit=gitta.get_current_commit(),
                dry_run=dry_run,
            )
        except Exception as exc:
            warning(f"Failed to post to Discord: {exc}")

    if push_git_history_to_child_repo:
        _push_git_history_to_child_repo(package=package, dry_run=dry_run)

    if dry_run:
        success("Dry run complete")

    return next_version


def cmd_publish(
    package: Annotated[Package, typer.Argument(help="Package to publish")],
    version: Annotated[
        Optional[str],
        typer.Option("--version", "-v", help="Explicit version to tag (3-4 dot-separated integers)"),
    ] = None,
    tag_only: Annotated[
        bool,
        typer.Option("--tag-only", help="Only create/push tag (skip pushing to child repo)"),
    ] = False,
    repo_only: Annotated[
        bool,
        typer.Option("--repo-only", help="Only push to child repo (skip creating/pushing tag)"),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Preview actions without executing")] = False,
    force: Annotated[bool, typer.Option("--force", help="Bypass branch and clean checks")] = False,
):
    if repo_only and tag_only:
        error("Cannot use --repo-only and --tag-only together")
        raise typer.Exit(1)

    remote = _get_metta_repo_remote()

    _publish(
        package=package,
        version_override=version,
        remote=remote,
        create_and_push_tag_to_monorepo=not repo_only,
        push_git_history_to_child_repo=not tag_only,
        dry_run=dry_run,
        skip_git_checks=force,
    )
