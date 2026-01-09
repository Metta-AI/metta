#!/usr/bin/env python3
"""Generate Python protobuf bindings from .proto files.

Proto files in ./proto/metta/ are compiled to ./metta/, preserving directory structure.
For example:
    proto/metta/protobuf/sim/single_episode.proto
      -> metta/protobuf/sim/single_episode_pb2.py
      -> metta/protobuf/sim/single_episode_pb2.pyi
"""

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from metta.common.util.fs import get_repo_root

REPO_ROOT = get_repo_root()

PROTO_ROOT = REPO_ROOT / "proto"


def get_proto_mappings(root_dir: Path = REPO_ROOT) -> list[dict]:
    """Get proto mappings with output paths relative to root_dir."""
    return [
        {
            "subdir": Path("metta"),
            "output_root": root_dir,
        },
        # Future: {"subdir": Path("mettagrid"), "output_root": root_dir / "packages/mettagrid/python/src"},
    ]


def find_proto_files(subdir: Path) -> list[Path]:
    """Find all .proto files under proto/{subdir}/."""
    return sorted((PROTO_ROOT / subdir).rglob("*.proto"))


def generate_protos(subdir: Path, output_root: Path) -> bool:
    """Generate Python bindings for protos in proto/{subdir}/ to output_root/{subdir}/."""
    proto_files = find_proto_files(subdir)
    if not proto_files:
        print(f"No .proto files found in {PROTO_ROOT / subdir}")
        return True

    output_root.mkdir(parents=True, exist_ok=True)

    # Use PROTO_ROOT as proto_path so generated module names include the full package path
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={PROTO_ROOT}",
        f"--python_out={output_root}",
        f"--pyi_out={output_root}",
        *[str(f.relative_to(PROTO_ROOT)) for f in proto_files],
    ]

    print(f"Generating protos: {PROTO_ROOT / subdir} -> {output_root}")
    for f in proto_files:
        print(f"  {f.relative_to(PROTO_ROOT)}")

    result = subprocess.run(cmd, cwd=PROTO_ROOT, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"protoc failed:\n{result.stderr}", file=sys.stderr)
        return False

    if result.stderr:
        print(result.stderr)

    return True


def ensure_init_files(subdir: Path, output_root: Path) -> None:
    """Create __init__.py files in generated directories.

    Stops at output_root/subdir to avoid creating __init__.py in namespace packages.
    """
    subdir_root = output_root / subdir
    for proto_file in find_proto_files(subdir):
        rel_path = proto_file.relative_to(PROTO_ROOT)
        # Create __init__.py in each parent directory, but not in subdir itself
        # (which may be a namespace package like metta/)
        for parent in (output_root / rel_path).parents:
            if parent == subdir_root or parent == output_root:
                break
            init_file = parent / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"  Created {init_file.relative_to(output_root)}")


def main(
    output: Annotated[Optional[Path], typer.Option(help="Output directory (default: repo root)")] = None,
) -> None:
    """Generate Python protobuf bindings from .proto files."""
    root_dir = output or REPO_ROOT
    success = True

    for mapping in get_proto_mappings(root_dir):
        subdir = mapping["subdir"]
        output_root = mapping["output_root"]

        if not (PROTO_ROOT / subdir).exists():
            print(f"Proto path does not exist: {PROTO_ROOT / subdir}")
            continue

        if not generate_protos(subdir, output_root):
            success = False
            continue

        ensure_init_files(subdir, output_root)

    if not success:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
