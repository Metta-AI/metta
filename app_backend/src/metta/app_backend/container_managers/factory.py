import os

from metta.app_backend.container_managers.base import AbstractContainerManager
from metta.app_backend.container_managers.docker import DockerContainerManager
from metta.app_backend.container_managers.eks import EksPodManager


def create_container_manager(
    container_runtime: str | None = None, namespace: str | None = None, kubeconfig: str | None = None
) -> AbstractContainerManager:
    runtime = container_runtime or os.environ.get("CONTAINER_RUNTIME", "docker").lower()

    if runtime == "docker":
        return DockerContainerManager()
    elif runtime == "eks" or runtime == "kubernetes":
        namespace = namespace or os.environ.get("K8S_NAMESPACE", "observatory")
        kubeconfig = kubeconfig or os.environ.get("KUBECONFIG")
        return EksPodManager(namespace=namespace, kubeconfig=kubeconfig)
    else:
        raise ValueError(f"Unsupported container runtime: {runtime}. Use 'docker' or 'eks'.")
