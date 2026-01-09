"""Test that generated protobuf files are up to date with .proto sources."""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from metta.common.util.fs import get_repo_root

GENERATED_PATTERNS = ["*_pb2.py", "*_pb2.pyi"]


def _hash_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _collect_fresh_hashes(root: Path) -> dict[str, str]:
    """Collect hashes for generated proto files in a temp directory."""
    hashes = {}
    for pattern in GENERATED_PATTERNS:
        for path in root.rglob(pattern):
            rel_path = str(path.relative_to(root))
            hashes[rel_path] = _hash_file(path)
    return hashes


def _collect_committed_hashes(root: Path) -> dict[str, str]:
    """Collect hashes for tracked generated proto files using git."""
    hashes = {}
    for pattern in GENERATED_PATTERNS:
        result = subprocess.run(
            ["git", "ls-files", pattern],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        for rel_path in result.stdout.strip().split("\n"):
            if rel_path:
                hashes[rel_path] = _hash_file(root / rel_path)
    return hashes


def test_proto_files_up_to_date():
    """Verify generated proto files match their .proto sources."""
    repo_root = get_repo_root()
    gen_script = repo_root / "scripts" / "generate_protos.py"

    assert gen_script.exists(), f"Proto generation script not found: {gen_script}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate to temp directory
        result = subprocess.run(
            [sys.executable, str(gen_script), "--output", str(tmpdir)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Proto generation failed:\n{result.stderr}\n{result.stdout}"

        fresh = _collect_fresh_hashes(tmpdir)
        committed = _collect_committed_hashes(repo_root)

        if fresh == committed:
            return

        # Compute delta
        errors = []
        for path, fresh_hash in sorted(fresh.items()):
            committed_hash = committed.pop(path, None)
            if committed_hash is None:
                errors.append(f"  missing: {path}")
            elif committed_hash != fresh_hash:
                errors.append(f"  outdated: {path}")

        for path in sorted(committed.keys()):
            errors.append(f"  orphaned: {path}")

        pytest.fail("Generated proto files need updating. Run: python scripts/generate_protos.py\n" + "\n".join(errors))
