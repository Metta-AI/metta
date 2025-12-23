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


def _get_kind_image_id(image_name: str) -> str | None:
    result = subprocess.run(
        [
            "docker",
            "exec",
            f"{KIND_CLUSTER_NAME}-control-plane",
            "crictl",
            "images",
            "-q",
            f"docker.io/library/{image_name}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split("\n")[0]
    return None


def _cleanup_kind_images():
    info("Cleaning up old images in Kind...")
    subprocess.run(
        ["docker", "exec", f"{KIND_CLUSTER_NAME}-control-plane", "crictl", "rmi", "--prune"],
        capture_output=True,
    )


def _get_local_image_id(image_name: str) -> str | None:
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def build_image(force_build: bool = True):
    image_exists = subprocess.run(["docker", "image", "inspect", IMAGE_NAME], capture_output=True).returncode == 0
    old_image_id = _get_local_image_id(IMAGE_NAME) if image_exists else None

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
        if old_image_id:
            new_image_id = _get_local_image_id(IMAGE_NAME)
            if new_image_id != old_image_id:
                info(f"Removing old image {old_image_id[:12]}...")
                subprocess.run(["docker", "rmi", old_image_id], capture_output=True)


def load_image_into_kind(force_load: bool = False):
    old_image_id = _get_kind_image_id(IMAGE_NAME)

    if old_image_id and not force_load:
        info(f"{IMAGE_NAME} already loaded in Kind, skipping")
        return

    info(f"Loading {IMAGE_NAME} into Kind...")
    subprocess.run(
        ["kind", "load", "docker-image", IMAGE_NAME, "--name", KIND_CLUSTER_NAME],
        check=True,
    )

    if old_image_id:
        new_image_id = _get_kind_image_id(IMAGE_NAME)
        if new_image_id != old_image_id:
            info(f"Removing old image {old_image_id[:12]} from Kind...")
            subprocess.run(
                ["docker", "exec", f"{KIND_CLUSTER_NAME}-control-plane", "crictl", "rmi", old_image_id],
                capture_output=True,
            )

    _cleanup_kind_images()
