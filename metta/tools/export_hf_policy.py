from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from hf_metta_policy import MettaPolicyConfig

DEFAULT_CODE_ROOTS = ("agent/src", "metta")
IGNORE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo")


def export_hf_policy(
    checkpoint_path: Path,
    export_dir: Path,
    *,
    checkpoint_filename: str = "policy.pt",
    code_roots: Sequence[str] = DEFAULT_CODE_ROOTS,
) -> None:
    """Create a Hugging Face style artifact containing checkpoint, config, and code snapshot."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    export_dir = export_dir.expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    snapshot_dir = export_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint_path, export_dir / checkpoint_filename)

    package_src = repo_root / "hf_metta_policy"
    shutil.copytree(package_src, export_dir / "hf_metta_policy", dirs_exist_ok=True, ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))

    normalized_roots = _normalize_roots(code_roots)
    for root in normalized_roots:
        src = (repo_root / root).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Code root does not exist: {src}")
        dest = snapshot_dir / root
        _copy_tree(src, dest)

    source_commit = _get_commit_hash(repo_root)

    config = MettaPolicyConfig(
        checkpoint_filename=checkpoint_filename,
        source_commit=source_commit,
        code_roots=[str(root) for root in normalized_roots],
    )
    config.save_pretrained(export_dir)

    readme_path = export_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(
            "# Metta policy export\n\n"
            "This directory was generated for offline Hugging Face interoperability.\n",
            encoding="utf-8",
        )


def _copy_tree(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dest,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
    )


def _normalize_roots(code_roots: Sequence[str]) -> list[Path]:
    normalized: list[Path] = []
    for root in code_roots:
        normalized.append(Path(root.strip("/")))
    return normalized


def _get_commit_hash(repo_root: Path) -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
            .decode("utf-8")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Metta policy checkpoint in Hugging Face format.")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pt checkpoint produced by CheckpointManager.")
    parser.add_argument("--export-dir", required=True, help="Directory where the HF artifact should be written.")
    parser.add_argument(
        "--checkpoint-filename",
        default="policy.pt",
        help="Filename to use for the copied checkpoint inside the export directory.",
    )
    parser.add_argument(
        "--code-root",
        action="append",
        dest="code_roots",
        default=[],
        help="Relative path to include in the code snapshot (can be provided multiple times).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    code_roots: list[str] = args.code_roots or list(DEFAULT_CODE_ROOTS)
    export_hf_policy(
        Path(args.checkpoint),
        Path(args.export_dir),
        checkpoint_filename=args.checkpoint_filename,
        code_roots=code_roots,
    )


if __name__ == "__main__":
    main()
