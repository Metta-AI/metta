"""Tests for metta.common.util.fs module."""

import hashlib
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from metta.common.util.fs import atomic_write, cd_repo_root, get_file_hash, get_repo_root, wait_for_file


class TestGetRepoRoot:
    """Test cases for the get_repo_root function."""

    def test_get_repo_root_from_git_directory(self):
        """Test finding repo root when current directory is the git root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                root = get_repo_root()
                assert root == Path(temp_dir).resolve()

    def test_get_repo_root_from_subdirectory(self):
        """Test finding repo root from a subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create git directory
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            # Create subdirectory
            sub_dir = Path(temp_dir) / "subdir" / "deep" / "nested"
            sub_dir.mkdir(parents=True)

            with patch("pathlib.Path.cwd", return_value=sub_dir):
                root = get_repo_root()
                assert root == Path(temp_dir).resolve()

    def test_get_repo_root_from_parent_directory(self):
        """Test finding repo root when it's in a parent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            root_dir = Path(temp_dir) / "project"
            root_dir.mkdir()
            git_dir = root_dir / ".git"
            git_dir.mkdir()

            # Create deep subdirectory
            work_dir = root_dir / "src" / "main" / "python"
            work_dir.mkdir(parents=True)

            with patch("pathlib.Path.cwd", return_value=work_dir):
                root = get_repo_root()
                assert root == root_dir.resolve()

    def test_get_repo_root_not_found(self):
        """Test SystemExit when no git directory is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # No .git directory
            work_dir = Path(temp_dir) / "not_a_repo"
            work_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=work_dir):
                with pytest.raises(SystemExit, match="Repository root not found"):
                    get_repo_root()

    def test_get_repo_root_multiple_git_dirs(self):
        """Test that it finds the closest git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create outer git repo
            outer_root = Path(temp_dir) / "outer"
            outer_root.mkdir()
            (outer_root / ".git").mkdir()

            # Create inner git repo
            inner_root = outer_root / "projects" / "inner"
            inner_root.mkdir(parents=True)
            (inner_root / ".git").mkdir()

            # Work from inside inner repo
            work_dir = inner_root / "src"
            work_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=work_dir):
                root = get_repo_root()
                assert root == inner_root.resolve()

    def test_get_repo_root_symlinks(self):
        """Test repo root detection with symlinks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create actual repo
            real_repo = Path(temp_dir) / "real_repo"
            real_repo.mkdir()
            (real_repo / ".git").mkdir()

            # Create symlink
            link_repo = Path(temp_dir) / "linked_repo"
            link_repo.symlink_to(real_repo)

            with patch("pathlib.Path.cwd", return_value=link_repo):
                root = get_repo_root()
                # Should resolve to the real path
                assert root == real_repo.resolve()


class TestCdRepoRoot:
    """Test cases for the cd_repo_root function."""

    def test_cd_repo_root_success(self):
        """Test successful change to repo root."""
        with tempfile.TemporaryDirectory() as temp_dir:
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                with patch("os.chdir") as mock_chdir:
                    cd_repo_root()
                    mock_chdir.assert_called_once_with(Path(temp_dir).resolve())

    def test_cd_repo_root_from_subdirectory(self):
        """Test changing to repo root from subdirectory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup repo structure
            git_dir = Path(temp_dir) / ".git"
            git_dir.mkdir()

            sub_dir = Path(temp_dir) / "subdir"
            sub_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=sub_dir):
                with patch("os.chdir") as mock_chdir:
                    cd_repo_root()
                    mock_chdir.assert_called_once_with(Path(temp_dir).resolve())

    def test_cd_repo_root_not_found(self):
        """Test SystemExit when repo root cannot be found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir) / "not_a_repo"
            work_dir.mkdir()

            with patch("pathlib.Path.cwd", return_value=work_dir):
                with pytest.raises(SystemExit, match="Repository root not found"):
                    cd_repo_root()


class TestAtomicWrite:
    """Test cases for the atomic_write function."""

    def test_atomic_write_success(self):
        """Test successful atomic write operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "test_file.txt"
            content = "Hello, atomic write!"

            def write_func(path):
                with open(path, "w") as f:
                    f.write(content)

            atomic_write(write_func, target_path)

            # File should exist and contain correct content
            assert target_path.exists()
            assert target_path.read_text() == content

    def test_atomic_write_creates_directory(self):
        """Test that atomic_write creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "nested" / "dirs" / "file.txt"

            def write_func(path):
                with open(path, "w") as f:
                    f.write("test content")

            atomic_write(write_func, target_path)

            assert target_path.exists()
            assert target_path.parent.exists()

    def test_atomic_write_custom_suffix(self):
        """Test atomic_write with custom temporary file suffix."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "test.dat"

            def write_func(path):
                # Verify temp file has custom suffix
                assert str(path).endswith(".backup")
                with open(path, "w") as f:
                    f.write("data")

            atomic_write(write_func, target_path, suffix=".backup")

            assert target_path.exists()
            assert target_path.read_text() == "data"

    def test_atomic_write_error_cleanup(self):
        """Test that temporary file is cleaned up on error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "test.txt"

            def failing_write_func(path):
                with open(path, "w") as f:
                    f.write("partial")
                raise RuntimeError("Write failed")

            with pytest.raises(RuntimeError, match="Write failed"):
                atomic_write(failing_write_func, target_path)

            # Target file should not exist
            assert not target_path.exists()

            # No temporary files should remain
            temp_files = list(Path(temp_dir).glob("*.tmp"))
            assert len(temp_files) == 0

    def test_atomic_write_binary_data(self):
        """Test atomic write with binary data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "binary_file.bin"
            binary_data = b'\x00\x01\x02\x03\xFF\xFE'

            def write_binary(path):
                with open(path, "wb") as f:
                    f.write(binary_data)

            atomic_write(write_binary, target_path)

            assert target_path.exists()
            assert target_path.read_bytes() == binary_data

    def test_atomic_write_overwrites_existing(self):
        """Test that atomic_write overwrites existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "existing.txt"

            # Create existing file
            target_path.write_text("old content")

            def write_new(path):
                with open(path, "w") as f:
                    f.write("new content")

            atomic_write(write_new, target_path)

            assert target_path.read_text() == "new content"

    def test_atomic_write_preserves_existing_on_error(self):
        """Test that existing file is preserved when write fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "existing.txt"
            original_content = "original content"

            # Create existing file
            target_path.write_text(original_content)

            def failing_write(path):
                with open(path, "w") as f:
                    f.write("partial new content")
                raise ValueError("Write error")

            with pytest.raises(ValueError, match="Write error"):
                atomic_write(failing_write, target_path)

            # Original file should be unchanged
            assert target_path.read_text() == original_content

    def test_atomic_write_string_path(self):
        """Test atomic_write with string path instead of Path object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = str(Path(temp_dir) / "string_path.txt")

            def write_func(path):
                with open(path, "w") as f:
                    f.write("string path test")

            atomic_write(write_func, target_path)

            assert Path(target_path).exists()
            assert Path(target_path).read_text() == "string path test"


class TestWaitForFile:
    """Test cases for the wait_for_file function."""

    def test_wait_for_file_already_exists(self):
        """Test waiting for a file that already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "existing.txt"
            test_file.write_text("content")

            result = wait_for_file(test_file, timeout=1.0)
            assert result is True

    def test_wait_for_file_created_during_wait(self):
        """Test waiting for a file that gets created during the wait."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "delayed.txt"

            # Create file after short delay
            def create_delayed_file():
                time.sleep(0.1)
                test_file.write_text("delayed content")

            import threading
            thread = threading.Thread(target=create_delayed_file)
            thread.start()

            try:
                result = wait_for_file(test_file, timeout=2.0, check_interval=0.05)
                assert result is True
                assert test_file.exists()
            finally:
                thread.join()

    def test_wait_for_file_timeout(self):
        """Test timeout when file is never created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "nonexistent.txt"

            start_time = time.time()
            result = wait_for_file(test_file, timeout=0.2, check_interval=0.05)
            elapsed = time.time() - start_time

            assert result is False
            assert elapsed >= 0.2
            assert elapsed < 0.5  # Should not wait much longer than timeout

    def test_wait_for_file_stability_check(self):
        """Test that file stability is properly checked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "growing.txt"

            # Create file that grows over time
            def write_growing_file():
                test_file.write_text("start")
                time.sleep(0.1)
                test_file.write_text("start + more")
                time.sleep(0.1)
                test_file.write_text("start + more + final")

            import threading
            thread = threading.Thread(target=write_growing_file)
            thread.start()

            try:
                result = wait_for_file(
                    test_file,
                    timeout=2.0,
                    check_interval=0.05,
                    stability_duration=0.2
                )
                assert result is True
                assert "final" in test_file.read_text()
            finally:
                thread.join()

    def test_wait_for_file_min_size_requirement(self):
        """Test minimum file size requirement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "small.txt"

            # Create very small file
            test_file.write_text("x")

            # Should fail with high min_size requirement
            result = wait_for_file(test_file, timeout=0.5, min_size=100)
            assert result is False

            # Should succeed with reasonable min_size (use shorter stability duration for faster test)
            result = wait_for_file(test_file, timeout=0.5, min_size=1, stability_duration=0.1)
            assert result is True

    def test_wait_for_file_progress_callback(self):
        """Test progress callback functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "progress.txt"

            callback_calls = []

            def progress_callback(elapsed, status):
                callback_calls.append((elapsed, status))

            # Create file after short delay
            def create_file():
                time.sleep(0.1)
                test_file.write_text("content")

            import threading
            thread = threading.Thread(target=create_file)
            thread.start()

            try:
                result = wait_for_file(
                    test_file,
                    timeout=1.0,
                    check_interval=0.05,
                    progress_callback=progress_callback
                )

                assert result is True
                assert len(callback_calls) > 0

                # Check that different statuses were called
                statuses = [call[1] for call in callback_calls]
                assert "waiting" in statuses
                assert "found" in statuses or "stable" in statuses

                # Check that elapsed times are reasonable
                elapsed_times = [call[0] for call in callback_calls]
                assert all(t >= 0 for t in elapsed_times)
                assert elapsed_times == sorted(elapsed_times)  # Should be increasing
            finally:
                thread.join()

    def test_wait_for_file_oserror_handling(self):
        """Test handling of OSError during file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("initial")

            # Mock stat to raise OSError occasionally
            original_stat = Path.stat
            call_count = 0

            def mock_stat(self):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Fail on second call
                    raise OSError("File temporarily unavailable")
                return original_stat(self)

            with patch.object(Path, 'stat', mock_stat):
                result = wait_for_file(
                    test_file,
                    timeout=1.0,
                    check_interval=0.1,
                    stability_duration=0.2
                )

                # Should still succeed despite OSError
                assert result is True

    def test_wait_for_file_string_path(self):
        """Test wait_for_file with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = str(Path(temp_dir) / "string_path.txt")
            Path(test_file).write_text("content")

            result = wait_for_file(test_file, timeout=0.5, stability_duration=0.1)
            assert result is True

    def test_wait_for_file_edge_case_parameters(self):
        """Test edge cases with unusual parameter values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "edge_case.txt"
            test_file.write_text("test")

            # Very short timeout with short stability check
            result = wait_for_file(test_file, timeout=0.2, stability_duration=0.05)
            assert result is True  # File exists immediately

            # Very long check interval with short stability
            result = wait_for_file(test_file, timeout=0.5, check_interval=0.2, stability_duration=0.1)
            assert result is True  # Should still work for existing file

            # Zero stability duration
            result = wait_for_file(test_file, timeout=0.5, stability_duration=0.0)
            assert result is True


class TestGetFileHash:
    """Test cases for the get_file_hash function."""

    def test_get_file_hash_basic(self):
        """Test basic file hash computation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            content = "Hello, hash world!"
            test_file.write_text(content)

            file_hash = get_file_hash(test_file)

            # Verify it's a valid SHA256 hash
            assert isinstance(file_hash, str)
            assert len(file_hash) == 64  # SHA256 hex digest length
            assert all(c in "0123456789abcdef" for c in file_hash)

            # Verify reproducibility
            second_hash = get_file_hash(test_file)
            assert file_hash == second_hash

    def test_get_file_hash_different_content(self):
        """Test that different content produces different hashes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "file1.txt"
            file2 = Path(temp_dir) / "file2.txt"

            file1.write_text("content A")
            file2.write_text("content B")

            hash1 = get_file_hash(file1)
            hash2 = get_file_hash(file2)

            assert hash1 != hash2

    def test_get_file_hash_identical_content(self):
        """Test that identical content produces identical hashes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "file1.txt"
            file2 = Path(temp_dir) / "file2.txt"

            content = "identical content"
            file1.write_text(content)
            file2.write_text(content)

            hash1 = get_file_hash(file1)
            hash2 = get_file_hash(file2)

            assert hash1 == hash2

    def test_get_file_hash_binary_data(self):
        """Test file hash with binary data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "binary.bin"
            binary_data = bytes(range(256))  # All possible byte values
            test_file.write_bytes(binary_data)

            file_hash = get_file_hash(test_file)

            assert isinstance(file_hash, str)
            assert len(file_hash) == 64

    def test_get_file_hash_large_file(self):
        """Test file hash with large file (tests chunked reading)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "large.txt"

            # Create file larger than chunk size (8192 bytes)
            large_content = "A" * 10000
            test_file.write_text(large_content)

            file_hash = get_file_hash(test_file)

            assert isinstance(file_hash, str)
            assert len(file_hash) == 64

            # Verify it matches expected SHA256
            expected_hash = hashlib.sha256(large_content.encode()).hexdigest()
            assert file_hash == expected_hash

    def test_get_file_hash_empty_file(self):
        """Test file hash with empty file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "empty.txt"
            test_file.write_text("")

            file_hash = get_file_hash(test_file)

            # Hash of empty file should be known SHA256 value
            expected_hash = hashlib.sha256(b"").hexdigest()
            assert file_hash == expected_hash

    def test_get_file_hash_custom_hash_function(self):
        """Test file hash with custom hash function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")

            # Use MD5 instead of SHA256
            md5_hash = get_file_hash(test_file, hash_func=hashlib.md5)

            assert isinstance(md5_hash, str)
            assert len(md5_hash) == 32  # MD5 hex digest length

            # Compare with expected MD5
            expected_md5 = hashlib.md5(b"test content").hexdigest()
            assert md5_hash == expected_md5

    def test_get_file_hash_string_path(self):
        """Test get_file_hash with string path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = str(Path(temp_dir) / "test.txt")
            Path(test_file).write_text("string path test")

            file_hash = get_file_hash(test_file)

            assert isinstance(file_hash, str)
            assert len(file_hash) == 64

    def test_get_file_hash_nonexistent_file(self):
        """Test that get_file_hash raises appropriate error for nonexistent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = Path(temp_dir) / "does_not_exist.txt"

            with pytest.raises(FileNotFoundError):
                get_file_hash(nonexistent_file)

    def test_get_file_hash_known_values(self):
        """Test file hash against known values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "known.txt"

            # Known content and expected SHA256
            content = "The quick brown fox jumps over the lazy dog"
            expected_sha256 = "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"

            test_file.write_text(content)
            file_hash = get_file_hash(test_file)

            assert file_hash == expected_sha256
