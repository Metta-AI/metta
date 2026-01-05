from kubernetes import config as kubernetes_config

from metta.app_backend.job_runner.config import JobDispatchConfig


def load_k8s_config(cfg: JobDispatchConfig) -> None:
    if cfg.LOCAL_DEV:
        if not cfg.LOCAL_DEV_K8S_CONTEXT:
            raise ValueError("LOCAL_DEV=true requires LOCAL_DEV_K8S_CONTEXT to be set")
        kubernetes_config.load_kube_config(context=cfg.LOCAL_DEV_K8S_CONTEXT)
    else:
        # Prod: require in-cluster config, no silent fallback
        kubernetes_config.load_incluster_config()
