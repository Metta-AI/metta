from unittest.mock import MagicMock, patch

import httpx
import pytest

from app_backend.description_generator import GitDataProvider, TrainingRunDescriptionGenerator
from app_backend.git_client import GitCommit
from app_backend.llm_client import LLMClient


class TestTrainingRunDescriptionGenerator:
    """Tests for TrainingRunDescriptionGenerator."""

    def test_generator_initialization_no_llm_client(self):
        """Test generator initialization without LLM client."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            with patch("app_backend.description_generator.logger") as mock_logger:
                TrainingRunDescriptionGenerator(git_data_provider, None)
                mock_logger.warning.assert_called_once_with("No LLM client available - will fall back to git logs only")

    def test_generator_initialization_with_available_llm_client(self):
        """Test generator initialization with available LLM client."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)
            llm_client = LLMClient(http_client)

            with patch.object(llm_client, "is_available", return_value=True):
                generator = TrainingRunDescriptionGenerator(git_data_provider, llm_client)
                assert generator.llm_client == llm_client

    def test_generate_description_local_git_success_with_llm(self):
        """Test successful description generation using local git and LLM."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)
            llm_client = LLMClient(http_client)

            with patch.object(llm_client, "is_available", return_value=True):
                generator = TrainingRunDescriptionGenerator(git_data_provider, llm_client)

                with (
                    patch.object(
                        git_data_provider,
                        "get_commit_data",
                        return_value=(
                            [
                                GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01"),
                                GitCommit("efgh5678", "Fix bug", "Author", "2024-01-02"),
                            ],
                            "2 files changed",
                            "diff content",
                        ),
                    ),
                    patch.object(llm_client, "generate_text_with_messages", return_value="Generated LLM description"),
                ):
                    result = generator.generate_description("abcd1234567890abcd1234567890abcd12345678")

                    assert result == "Generated LLM description"
                    llm_client.generate_text_with_messages.assert_called_once()

    def test_generate_description_local_git_success_fallback_to_git_logs(self):
        """Test successful description generation falling back to git logs."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with patch.object(
                git_data_provider,
                "get_commit_data",
                return_value=(
                    [
                        GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01"),
                        GitCommit("efgh5678", "Fix bug", "Author", "2024-01-02"),
                    ],
                    "2 files changed",
                    "diff content",
                ),
            ):
                result = generator.generate_description("abcd1234567890abcd1234567890abcd12345678")

                assert "Training run based on 2 commits:" in result
                assert "abcd1234: Add new feature" in result
                assert "efgh5678: Fix bug" in result

    def test_generate_description_llm_failure_fallback_to_git_logs(self):
        """Test LLM failure fallback to git logs."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)
            # Create LLM client with key so it's available
            llm_client = LLMClient(http_client, openai_api_key="test-key")

            generator = TrainingRunDescriptionGenerator(git_data_provider, llm_client)

            # Mock HTTP client to prevent real API calls
            with (
                patch.object(
                    git_data_provider,
                    "get_commit_data",
                    return_value=(
                        [GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01")],
                        "1 file changed",
                        "diff content",
                    ),
                ),
                patch.object(http_client, "post", side_effect=Exception("LLM API error")),
                patch("app_backend.description_generator.logger") as mock_logger,
            ):
                result = generator.generate_description("abcd1234567890abcd1234567890abcd12345678")

                assert "Training run based on 1 commit:" in result
                assert "abcd1234: Add new feature" in result
                mock_logger.warning.assert_called_once()

    def test_generate_description_github_fallback_success(self):
        """Test successful description generation falling back to GitHub."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with patch.object(
                git_data_provider,
                "get_commit_data",
                return_value=(
                    [
                        GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01"),
                        GitCommit("efgh5678", "Fix bug", "Author", "2024-01-02"),
                    ],
                    "2 files changed",
                    "diff content",
                ),
            ):
                result = generator.generate_description("abcd1234567890abcd1234567890abcd12345678")

                assert "Training run based on 2 commits:" in result
                assert "abcd1234: Add new feature" in result
                assert "efgh5678: Fix bug" in result

    def test_generate_description_commit_not_found(self):
        """Test description generation with commit not found anywhere."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with patch.object(git_data_provider, "get_commit_data", side_effect=ValueError("Commit not found")):
                with pytest.raises(ValueError) as exc_info:
                    generator.generate_description("invalid-commit-hash")
                assert "Commit not found" in str(exc_info.value)

    def test_generate_description_no_commits(self):
        """Test description generation when no commits are found."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with patch.object(git_data_provider, "get_commit_data", return_value=([], "stats", "diff")):
                with pytest.raises(ValueError) as exc_info:
                    generator.generate_description("some-commit-hash")
                assert "No commits found" in str(exc_info.value)

    def test_build_llm_messages(self):
        """Test LLM messages building."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            from app_backend.description_generator import TrainingRunInput

            input_data = TrainingRunInput(
                commits=[
                    GitCommit("abcd1234", "Add feature X", "John Doe", "2024-01-01"),
                    GitCommit("efgh5678", "Improve performance", "Jane Smith", "2024-01-02"),
                ],
                diff_stats="2 files changed, 10 insertions(+), 5 deletions(-)",
                diff_content="diff --git a/file.py b/file.py\n+new code",
            )

            messages = generator._build_llm_messages(input_data)

            # Check that we have system message and examples plus current request
            assert len(messages) > 1
            assert messages[0]["role"] == "system"
            assert "Metta AI" in messages[0]["content"]
            assert "multi-agent" in messages[0]["content"]
            assert "cooperation" in messages[0]["content"]

            # Check current request message contains our data (but not commit hashes)
            current_message = messages[-1]
            assert current_message["role"] == "user"
            assert "Add feature X" in current_message["content"]
            assert "Improve performance" in current_message["content"]
            assert "John Doe" in current_message["content"]
            assert "Jane Smith" in current_message["content"]

    def test_build_git_summary(self):
        """Test git summary building."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            commits = [
                GitCommit("abcd1234", "Add feature X", "John Doe", "2024-01-01"),
                GitCommit("efgh5678", "Improve performance", "Jane Smith", "2024-01-02"),
            ]

            summary = generator._build_git_summary(commits)

            assert "Training run based on 2 commits:" in summary
            assert "abcd1234: Add feature X (by John Doe on 2024-01-01)" in summary
            assert "efgh5678: Improve performance (by Jane Smith on 2024-01-02)" in summary

    def test_build_git_summary_empty(self):
        """Test git summary with no commits."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with pytest.raises(ValueError) as exc_info:
                generator._build_git_summary([])
            assert "No commits found in this training run" in str(exc_info.value)

    def test_add_uncommitted_changes_disclaimer(self):
        """Test uncommitted changes disclaimer."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            # Test with uncommitted changes
            result = generator._add_uncommitted_changes_disclaimer(
                "Test description", "abcd1234567890abcd1234567890abcd12345678", True
            )
            assert result.startswith("WARNING: Training run abcd1234 included uncommitted changes\n\n")
            assert "Test description" in result

            # Test without uncommitted changes
            result = generator._add_uncommitted_changes_disclaimer(
                "Test description", "abcd1234567890abcd1234567890abcd12345678", False
            )
            assert result == "Test description"

    def test_generate_description_with_uncommitted_changes(self):
        """Test description generation with uncommitted changes disclaimer."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with patch.object(
                git_data_provider,
                "get_commit_data",
                return_value=(
                    [GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01")],
                    "1 file changed",
                    "diff content",
                ),
            ):
                result = generator.generate_description(
                    "abcd1234567890abcd1234567890abcd12345678", has_uncommitted_changes=True
                )

                assert "WARNING: Training run abcd1234 included uncommitted changes" in result
                assert "Training run based on 1 commit:" in result

    def test_git_error_converted_to_value_error(self):
        """Test that GitError from git client gets converted to ValueError."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with patch.object(git_data_provider, "get_commit_data", side_effect=ValueError("Git error")):
                with pytest.raises(ValueError) as exc_info:
                    generator.generate_description("invalid-commit-hash")
                assert "Git error" in str(exc_info.value)

    def test_git_not_installed_github_also_fails(self):
        """Test when git is not installed and GitHub API also fails."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, None)

            with patch.object(git_data_provider, "get_commit_data", side_effect=Exception("All clients failed")):
                with pytest.raises(Exception) as exc_info:
                    generator.generate_description("some-commit-hash")
                assert "All clients failed" in str(exc_info.value)

    def test_working_llm_client_generates_description(self):
        """Test successful description generation with working LLM client."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)
            # Create LLM client with key and mock HTTP to prevent real calls
            llm_client = LLMClient(http_client, openai_api_key="test-key")

            generator = TrainingRunDescriptionGenerator(git_data_provider, llm_client)

            # Mock successful API response
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "LLM generated description"}}]}

            with (
                patch.object(
                    git_data_provider,
                    "get_commit_data",
                    return_value=(
                        [
                            GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01"),
                            GitCommit("efgh5678", "Fix bug", "Author", "2024-01-02"),
                        ],
                        "2 files changed",
                        "diff content",
                    ),
                ),
                patch.object(http_client, "post", return_value=mock_response),
            ):
                result = generator.generate_description("abcd1234567890abcd1234567890abcd12345678")

                assert result == "LLM generated description"

    def test_non_working_llm_client_falls_back_to_git_logs(self):
        """Test fallback to git logs when LLM client is not working."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)
            # Create LLM client without keys (won't be available)
            llm_client = LLMClient(http_client)

            generator = TrainingRunDescriptionGenerator(git_data_provider, llm_client)

            with patch.object(
                git_data_provider,
                "get_commit_data",
                return_value=(
                    [GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01")],
                    "1 file changed",
                    "diff content",
                ),
            ):
                result = generator.generate_description("abcd1234567890abcd1234567890abcd12345678")

                assert "Training run based on 1 commit:" in result
                assert "abcd1234: Add new feature" in result

    def test_llm_client_available_but_generation_fails(self):
        """Test LLM client is available but generation fails, fallback to git logs."""
        with httpx.Client() as http_client:
            git_data_provider = GitDataProvider(http_client)
            # Create LLM client with key and mock HTTP failure
            llm_client = LLMClient(http_client, openai_api_key="test-key")

            generator = TrainingRunDescriptionGenerator(git_data_provider, llm_client)

            with (
                patch.object(
                    git_data_provider,
                    "get_commit_data",
                    return_value=(
                        [GitCommit("abcd1234", "Add new feature", "Author", "2024-01-01")],
                        "1 file changed",
                        "diff content",
                    ),
                ),
                patch.object(http_client, "post", side_effect=Exception("API rate limit exceeded")),
            ):
                result = generator.generate_description("abcd1234567890abcd1234567890abcd12345678")

                assert "Training run based on 1 commit:" in result
                assert "abcd1234: Add new feature" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
