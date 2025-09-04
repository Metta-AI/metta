"""Pytest configuration and shared fixtures for gitta tests."""

import subprocess
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def isolate_git_env(monkeypatch):
    """Isolate git environment to prevent interference with user's git config."""
    # Set isolated git config for tests
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("HOME", tempfile.gettempdir())
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test Author")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test Committer")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.com")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def git_repo(temp_dir: Path) -> Path:
    """Create a basic git repository for testing."""
    # Initialize repo
    subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)

    # Set local config
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_dir, check=True, capture_output=True)

    # Create initial commit
    (temp_dir / "README.md").write_text("# Test Repository\n")
    subprocess.run(["git", "add", "."], cwd=temp_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True, capture_output=True)

    return temp_dir


@pytest.fixture
def git_repo_with_remote(git_repo: Path, temp_dir: Path) -> Path:
    """Create a git repository with a remote."""
    # Create a bare repo to act as remote
    remote_path = temp_dir / "remote.git"
    subprocess.run(["git", "init", "--bare"], cwd=remote_path, check=True, capture_output=True)

    # Add remote to the main repo
    subprocess.run(["git", "remote", "add", "origin", str(remote_path)], cwd=git_repo, check=True, capture_output=True)

    # Push to remote
    subprocess.run(["git", "push", "-u", "origin", "HEAD"], cwd=git_repo, check=True, capture_output=True)

    return git_repo


@pytest.fixture
def mock_anthropic_response():
    """Provide a mock response for Anthropic API calls."""

    def _mock_response(content: str):
        class MockMessage:
            def __init__(self, text):
                self.text = text

        class MockResponse:
            def __init__(self, content):
                self.content = [MockMessage(content)]

        return MockResponse(content)

    return _mock_response


@pytest.fixture(autouse=True)
def cleanup_env_vars(monkeypatch):
    """Clean up environment variables that might affect tests."""
    # Remove any existing API keys or tokens from environment
    for key in ["ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GH_TOKEN"]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def mock_github_api():
    """Provide mock responses for GitHub API calls."""

    class MockGitHubAPI:
        def __init__(self):
            self.responses = {}

        def set_response(self, url_pattern: str, response_data: dict, status_code: int = 200):
            """Set a mock response for a URL pattern."""
            self.responses[url_pattern] = {"data": response_data, "status_code": status_code}

        def get_response(self, url: str):
            """Get mock response for a URL."""
            for pattern, response in self.responses.items():
                if pattern in url:
                    return response
            return {"data": {}, "status_code": 404}

    return MockGitHubAPI()


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "requires_network: marks tests that require network access")
    config.addinivalue_line(
        "markers", "requires_git_filter_repo: marks tests that require git-filter-repo to be installed"
    )
