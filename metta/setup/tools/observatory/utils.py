import subprocess
from pathlib import Path

from metta.common.util.fs import get_repo_root


def build_img(tag: str, dockerfile_path: Path, build_args: list[str] | None = None):
    cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path)]
    if build_args:
        cmd.extend(build_args)
    cmd.append(str(get_repo_root()))
    subprocess.run(cmd, check=True)
