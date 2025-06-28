#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "PyGithub>=2.1.1",
#   "requests>=2.31.0",
# ]
# ///
"""
Fetch Artifacts Script

Downloads zipped artifacts from previous workflow runs and saves them to disk.
Simple and focused - just gets the ZIP files for you to process as needed.
"""

import fnmatch
import os
import sys
from pathlib import Path
from typing import Any, Optional

from github import Github


class GitHubActionsOutput:
    """Helper for setting GitHub Actions outputs safely."""

    def __init__(self):
        self.github_output = os.environ.get("GITHUB_OUTPUT")

    def set_output(self, name: str, value: str) -> None:
        """Set a GitHub Actions output value."""
        if self.github_output:
            with open(self.github_output, "a", encoding="utf-8") as f:
                # Use delimiter for multiline content
                if "\n" in value:
                    delimiter = f"EOF_{hash(value) % 10000}"
                    f.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")
                else:
                    f.write(f"{name}={value}\n")
        else:
            raise RuntimeError(
                "GITHUB_OUTPUT environment variable not set. This script requires a modern GitHub Actions environment."
            )


class GitHubAPI:
    """GitHub API client using PyGithub for fetching workflow runs and artifacts."""

    def __init__(self, token: str, repo: str):
        self.token = token
        self.github = Github(token)
        self.repo = self.github.get_repo(repo)

    def get_workflow_runs(self, workflow_filename: str, exclude_run_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Get successful workflow runs for the specified workflow."""
        try:
            workflow = self.repo.get_workflow(workflow_filename)
            # Get all runs (most recent first) and filter as we go
            runs = workflow.get_runs()

            result = []
            checked_runs = 0
            max_checks = 200  # Reasonable limit to avoid checking too many runs

            for run in runs:
                checked_runs += 1

                # Only process successful runs
                if run.conclusion != "success":
                    # Continue to next run without adding to result
                    if checked_runs >= max_checks:
                        break
                    continue

                # Exclude current run if specified
                if exclude_run_id and str(run.id) == str(exclude_run_id):
                    if checked_runs >= max_checks:
                        break
                    continue

                result.append(
                    {
                        "id": run.id,
                        "created_at": run.created_at.isoformat(),
                        "url": run.html_url,
                    }
                )

                # Stop if we have enough successful runs or checked too many
                if len(result) >= 50 or checked_runs >= max_checks:
                    break

            return result
        except Exception as e:
            print(f"❌ Error fetching workflow runs: {e}")
            return []

    def get_run_artifacts(self, run_id: int) -> list[dict[str, Any]]:
        """Get artifacts for a specific workflow run."""
        try:
            run = self.repo.get_workflow_run(run_id)

            result = []
            for artifact in run.get_artifacts():
                result.append(
                    {
                        "id": artifact.id,
                        "name": artifact.name,
                        "size": artifact.size_in_bytes,
                        "created_at": artifact.created_at.isoformat(),
                        "expires_at": artifact.expires_at.isoformat() if artifact.expires_at else None,
                        "download_url": artifact.archive_download_url,
                    }
                )

            return result
        except Exception as e:
            print(f"❌ Error fetching artifacts for run {run_id}: {e}")
            return []

    def download_artifact(self, download_url: str, output_path: Path) -> bool:
        """Download an artifact ZIP file."""
        try:
            import requests

            # Use the stored token for download
            headers = {"Authorization": f"Bearer {self.token}"}

            response = requests.get(download_url, headers=headers, stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True
        except Exception as e:
            print(f"❌ Error downloading artifact: {e}")
            return False


class ArtifactFetcher:
    """Main class for fetching artifacts and saving them as ZIP files."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.github_api = GitHubAPI(config["github_token"], config["repo"])
        self.output = GitHubActionsOutput()
        self.current_run_id = os.environ.get("GITHUB_RUN_ID")

    def matches_pattern(self, artifact_name: str, pattern: str) -> bool:
        """Check if artifact name matches the pattern."""
        return fnmatch.fnmatch(artifact_name.lower(), pattern.lower())

    def find_and_collect_artifacts(self) -> tuple[list[dict[str, Any]], int]:
        """Find artifacts matching the specified pattern, collecting until we have num_artifacts."""
        print(
            f"🔍 Searching for {self.config['num_artifacts']} "
            f"artifacts matching '{self.config['artifact_name_pattern']}'"
        )
        print(f"📋 Workflow: {self.config['workflow_name']}")

        try:
            # Get all successful workflow runs
            runs = self.github_api.get_workflow_runs(self.config["workflow_name"], exclude_run_id=self.current_run_id)

            print(f"📊 Found {len(runs)} successful workflow runs to search")

            matching_artifacts = []
            runs_searched = 0
            target_count = self.config["num_artifacts"]

            for run in runs:
                runs_searched += 1
                run_id = run["id"]
                run_date = run["created_at"]

                print(
                    f"🔍 Searching run {run_id} ({run_date[:10]}) - "
                    f" {len(matching_artifacts)}/{target_count} artifacts found"
                )

                try:
                    artifacts = self.github_api.get_run_artifacts(run_id)

                    for artifact in artifacts:
                        if self.matches_pattern(artifact["name"], self.config["artifact_name_pattern"]):
                            artifact_info = {
                                "artifact_id": artifact["id"],
                                "artifact_name": artifact["name"],
                                "size_bytes": artifact["size"],
                                "created_at": artifact["created_at"],
                                "run_id": run_id,
                                "run_date": run_date,
                                "workflow_url": run["url"],
                                "expires_at": artifact["expires_at"],
                                "download_url": artifact["download_url"],
                            }
                            matching_artifacts.append(artifact_info)
                            print(f"✅ Found matching artifact: {artifact['name']} ({artifact['size']:,} bytes)")

                            # Check if we've collected enough artifacts
                            if len(matching_artifacts) >= target_count:
                                print(f"🎯 Collected {target_count} artifacts, stopping search")
                                return matching_artifacts, runs_searched

                except Exception as e:
                    print(f"⚠️ Error fetching artifacts for run {run_id}: {e}")
                    continue

            print(f"🎯 Search complete: found {len(matching_artifacts)} artifacts after searching {runs_searched} runs")
            return matching_artifacts, runs_searched

        except Exception as e:
            print(f"❌ Error fetching workflow runs: {e}")
            return [], 0

    def download_artifact(self, artifact_info: dict[str, Any]) -> dict[str, Any]:
        """Download a single artifact as a ZIP file."""
        artifact_name = artifact_info["artifact_name"]
        run_id = artifact_info["run_id"]
        download_url = artifact_info["download_url"]

        # Create output directory
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename: artifact-name_run-id.zip
        zip_filename = f"{artifact_name}_{run_id}.zip"
        zip_path = output_dir / zip_filename

        print(f"📦 Downloading {artifact_name} to {zip_path}")

        try:
            success = self.github_api.download_artifact(download_url, zip_path)

            if success and zip_path.exists():
                file_size = zip_path.stat().st_size
                print(f"✅ Downloaded successfully: {file_size:,} bytes")

                return {
                    **artifact_info,
                    "status": "downloaded",
                    "local_path": str(zip_path),
                    "local_filename": zip_filename,
                    "downloaded_size_bytes": file_size,
                }
            else:
                print("❌ Download failed - file not created")
                return {**artifact_info, "status": "failed", "error": "File not created"}

        except Exception as e:
            print(f"❌ Error downloading artifact {artifact_name}: {e}")
            return {**artifact_info, "status": "error", "error": str(e)}

    def run(self) -> dict[str, Any]:
        """Main execution method."""
        # Find and collect the requested number of artifacts
        matching_artifacts, runs_searched = self.find_and_collect_artifacts()

        if not matching_artifacts:
            self.output.set_output("success", "true")
            self.output.set_output("artifacts-found", "0")
            print("ℹ️ No matching artifacts found")
            return {"success": True, "artifacts_found": 0, "message": "No matching artifacts found"}

        # Download artifacts
        downloaded_artifacts = []
        successful_downloads = 0

        for artifact_info in matching_artifacts:
            result = self.download_artifact(artifact_info)
            downloaded_artifacts.append(result)

            if result["status"] == "downloaded":
                successful_downloads += 1

        # Set outputs
        self.output.set_output("success", "true")
        self.output.set_output("artifacts-found", str(successful_downloads))

        # Print summary
        self._print_summary(downloaded_artifacts, runs_searched)

        return {"success": True, "artifacts_found": successful_downloads, "downloaded_artifacts": downloaded_artifacts}

    def _print_summary(self, downloaded_artifacts: list[dict[str, Any]], runs_searched: int) -> None:
        """Print a summary of the download operation."""
        print("\n📊 Download Summary:")
        print(f"{'Artifact Name':<30} {'Run ID':<12} {'Date':<12} {'Size':<12} {'Status'}")
        print("-" * 80)

        total_size = 0
        successful = 0

        for artifact in downloaded_artifacts:
            name = artifact["artifact_name"][:28]
            run_id = str(artifact["run_id"])[-8:]  # Last 8 chars
            date = artifact["run_date"][:10]  # Just the date
            size = f"{artifact['size_bytes']:,}B"
            status = artifact.get("status", "unknown")

            if status == "downloaded":
                successful += 1
                total_size += artifact.get("downloaded_size_bytes", 0)

            print(f"{name:<30} {run_id:<12} {date:<12} {size:<12} {status}")

        print(f"\n✅ Successfully downloaded {successful}/{len(downloaded_artifacts)} artifacts")
        print(f"📁 Total size: {total_size:,} bytes")
        print(f"🔍 Runs searched: {runs_searched}")
        print(f"💾 Files saved to: {self.config['output_directory']}/")

        if successful > 0:
            print("\n📂 Downloaded files:")
            for artifact in downloaded_artifacts:
                if artifact.get("status") == "downloaded":
                    print(f"  • {artifact['local_filename']}")


def main():
    """Main entry point."""
    print("🚀 Starting Fetch Artifacts action")

    # Import parse_config
    script_dir = Path(__file__).parent.parent.parent / "scripts"
    sys.path.insert(0, str(script_dir))
    from utils.config import parse_config

    try:
        required_vars = [
            "GITHUB_TOKEN",
            "INPUT_WORKFLOW_NAME",
            "INPUT_ARTIFACT_NAME_PATTERN",
            "GITHUB_REPOSITORY",
        ]

        optional_vars = {
            "INPUT_NUM_ARTIFACTS": "5",
            "INPUT_OUTPUT_DIRECTORY": "downloaded-artifacts",
        }

        env_values = parse_config(required_vars, optional_vars)

        # Transform to config dict
        config = {
            "github_token": env_values["GITHUB_TOKEN"],
            "repo": env_values["GITHUB_REPOSITORY"],
            "workflow_name": env_values["INPUT_WORKFLOW_NAME"],
            "artifact_name_pattern": env_values["INPUT_ARTIFACT_NAME_PATTERN"],
            "num_artifacts": int(env_values["INPUT_NUM_ARTIFACTS"]),
            "output_directory": env_values["INPUT_OUTPUT_DIRECTORY"],
        }

        print("📋 Configuration:")
        print(f"  • Repository: {config['repo']}")
        print(f"  • Workflow: {config['workflow_name']}")
        print(f"  • Artifact pattern: {config['artifact_name_pattern']}")
        print(f"  • Number of artifacts to collect: {config['num_artifacts']}")
        print(f"  • Output directory: {config['output_directory']}")

        fetcher = ArtifactFetcher(config)
        result = fetcher.run()

        if result["success"]:
            print(f"🎉 Completed successfully: {result['artifacts_found']} artifacts downloaded")
        else:
            print(f"❌ Operation failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
