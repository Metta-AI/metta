import os

import metta.app_backend.container_managers.base
import metta.app_backend.container_managers.docker
import metta.app_backend.container_managers.k8s


def create_container_manager(
    container_runtime: str | None = None, namespace: str | None = None, kubeconfig: str | None = None
) -> metta.app_backend.container_managers.base.AbstractContainerManager:
    runtime = container_runtime or os.environ.get("CONTAINER_RUNTIME", "k8s").lower()

    if runtime == "docker":
        return metta.app_backend.container_managers.docker.DockerContainerManager()
    elif runtime == "k8s":
        return metta.app_backend.container_managers.k8s.K8sPodManager(namespace=namespace, kubeconfig=kubeconfig)
    else:
        raise ValueError(f"Unsupported container runtime: {runtime}. Use 'docker' or 'k8s'.")
