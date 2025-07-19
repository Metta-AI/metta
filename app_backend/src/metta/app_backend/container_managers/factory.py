import os

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.docker import DockerContainerManager
from metta.app_backend.container_managers.k8s import K8sPodManager


def create_container_manager(
    container_runtime: str | None = None, namespace: str | None = None, kubeconfig: str | None = None
) -> AbstractContainerManager:
    runtime = container_runtime or os.environ.get("CONTAINER_RUNTIME", "docker").lower()

    if runtime == "docker":
        return DockerContainerManager()
    elif runtime in ["k8s"]:
        return K8sPodManager(namespace=namespace, kubeconfig=kubeconfig)
    else:
        raise ValueError(f"Unsupported container runtime: {runtime}. Use 'docker' or 'k8s'.")
