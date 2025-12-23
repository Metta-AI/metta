import subprocess
from pathlib import Path

from metta.common.util.fs import get_repo_root
from metta.setup.utils import info

IMAGE_NAME = "metta-policy-evaluator-local:latest"
KIND_CLUSTER_NAME = "metta-local"


def build_img(tag: str, dockerfile_path: Path, build_args: list[str] | None = None):
    cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path)]
    if build_args:
        cmd.extend(build_args)
    cmd.append(str(get_repo_root()))
    subprocess.run(cmd, check=True)


def build_and_load_image(force_build: bool = True):
    image_exists = subprocess.run(["docker", "image", "inspect", IMAGE_NAME], capture_output=True).returncode == 0

    if force_build or not image_exists:
        info(f"Building {IMAGE_NAME}...")
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                IMAGE_NAME,
                "-f",
                "devops/docker/Dockerfile.policy_evaluator",
                "--platform",
                "linux/amd64",
                ".",
            ],
            check=True,
            cwd=get_repo_root(),
        )

    info(f"Loading {IMAGE_NAME} into Kind...")
    subprocess.run(
        ["kind", "load", "docker-image", IMAGE_NAME, "--name", KIND_CLUSTER_NAME],
        check=True,
    )
