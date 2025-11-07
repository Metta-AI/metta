"""Kubernetes resource efficiency and health collector for Datadog monitoring.

Tracks:
- Resource waste (CPU/memory overallocation)
- Pod health (crashes, restarts, failures)
- Underutilization (idle/underperforming pods)
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from kubernetes import client, config

from devops.datadog.utils.base import BaseCollector


class KubernetesCollector(BaseCollector):
    """Collector for Kubernetes cluster efficiency and health metrics.

    Monitors resource waste, pod health issues, and underutilization to help
    optimize cluster costs and reliability.
    """

    def __init__(self):
        super().__init__(name="kubernetes")

        try:
            config.load_incluster_config()
            self.logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            config.load_kube_config()
            self.logger.info("Loaded kubeconfig from local environment")

        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

    def collect_metrics(self) -> dict[str, Any]:
        metrics = {}

        metrics.update(self._collect_resource_efficiency())
        metrics.update(self._collect_pod_health())
        metrics.update(self._collect_underutilization())

        return metrics

    def _collect_resource_efficiency(self) -> dict[str, Any]:
        metrics = {}

        try:
            pods = self.core_v1.list_pod_for_all_namespaces()

            total_cpu_requested = 0.0
            total_cpu_used = 0.0
            total_memory_requested = 0.0
            total_memory_used = 0.0
            overallocated_count = 0

            try:
                custom_api = client.CustomObjectsApi()
                pod_metrics = custom_api.list_cluster_custom_object(
                    group="metrics.k8s.io", version="v1beta1", plural="podmetrics"
                )
            except Exception as e:
                self.logger.warning(f"Could not fetch pod metrics: {e}")
                pod_metrics = None

            usage_map = {}
            if pod_metrics:
                for pod_metric in pod_metrics.get("items", []):
                    namespace = pod_metric["metadata"]["namespace"]
                    name = pod_metric["metadata"]["name"]
                    key = f"{namespace}/{name}"

                    cpu_usage = 0.0
                    memory_usage = 0.0

                    for container in pod_metric.get("containers", []):
                        cpu_str = container["usage"].get("cpu", "0")
                        if cpu_str.endswith("n"):
                            cpu_usage += float(cpu_str[:-1]) / 1_000_000_000
                        elif cpu_str.endswith("m"):
                            cpu_usage += float(cpu_str[:-1]) / 1000
                        else:
                            cpu_usage += float(cpu_str)

                        mem_str = container["usage"].get("memory", "0")
                        if mem_str.endswith("Ki"):
                            memory_usage += float(mem_str[:-2]) * 1024
                        elif mem_str.endswith("Mi"):
                            memory_usage += float(mem_str[:-2]) * 1024 * 1024
                        elif mem_str.endswith("Gi"):
                            memory_usage += float(mem_str[:-2]) * 1024 * 1024 * 1024
                        else:
                            memory_usage += float(mem_str)

                    usage_map[key] = {"cpu": cpu_usage, "memory": memory_usage}

            for pod in pods.items:
                if pod.status.phase not in ["Running", "Pending"]:
                    continue

                namespace = pod.metadata.namespace
                name = pod.metadata.name
                key = f"{namespace}/{name}"

                cpu_requested = 0.0
                memory_requested = 0.0

                for container in pod.spec.containers:
                    if container.resources and container.resources.requests:
                        cpu_req = container.resources.requests.get("cpu", "0")
                        if cpu_req.endswith("m"):
                            cpu_requested += float(cpu_req[:-1]) / 1000
                        else:
                            cpu_requested += float(cpu_req)

                        mem_req = container.resources.requests.get("memory", "0")
                        if mem_req.endswith("Ki"):
                            memory_requested += float(mem_req[:-2]) * 1024
                        elif mem_req.endswith("Mi"):
                            memory_requested += float(mem_req[:-2]) * 1024 * 1024
                        elif mem_req.endswith("Gi"):
                            memory_requested += float(mem_req[:-2]) * 1024 * 1024 * 1024
                        else:
                            memory_requested += float(mem_req)

                total_cpu_requested += cpu_requested
                total_memory_requested += memory_requested

                if key in usage_map:
                    cpu_used = usage_map[key]["cpu"]
                    memory_used = usage_map[key]["memory"]

                    total_cpu_used += cpu_used
                    total_memory_used += memory_used

                    if cpu_requested > 0 and cpu_used / cpu_requested < 0.2:
                        overallocated_count += 1
                    elif memory_requested > 0 and memory_used / memory_requested < 0.2:
                        overallocated_count += 1

            metrics["k8s.resources.waste"] = []
            metrics["k8s.resources.efficiency"] = []

            if total_cpu_requested > 0:
                cpu_waste = max(0.0, total_cpu_requested - total_cpu_used)
                metrics["k8s.resources.waste"].append((cpu_waste, ["resource:cpu", "unit:cores"]))
                cpu_efficiency = (total_cpu_used / total_cpu_requested) * 100
                metrics["k8s.resources.efficiency"].append((cpu_efficiency, ["resource:cpu"]))

            if total_memory_requested > 0:
                memory_waste_bytes = max(0.0, total_memory_requested - total_memory_used)
                memory_waste_gb = memory_waste_bytes / (1024**3)
                metrics["k8s.resources.waste"].append((memory_waste_gb, ["resource:memory", "unit:gb"]))
                memory_efficiency = (total_memory_used / total_memory_requested) * 100
                metrics["k8s.resources.efficiency"].append((memory_efficiency, ["resource:memory"]))

            metrics["k8s.pods"] = [(overallocated_count, ["status:overallocated"])]

        except Exception as e:
            self.logger.error(f"Failed to collect resource efficiency metrics: {e}", exc_info=True)

        return metrics

    def _collect_pod_health(self) -> dict[str, Any]:
        metrics = {}

        try:
            pods = self.core_v1.list_pod_for_all_namespaces()

            crash_looping = 0
            failed = 0
            pending = 0
            oomkilled_24h = 0
            high_restarts = 0
            image_pull_errors = 0

            for pod in pods.items:
                if pod.status.phase == "Failed":
                    failed += 1
                elif pod.status.phase == "Pending":
                    pending += 1

                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if container_status.state and container_status.state.waiting:
                            if container_status.state.waiting.reason == "CrashLoopBackOff":
                                crash_looping += 1
                            if "ImagePull" in (container_status.state.waiting.reason or ""):
                                image_pull_errors += 1

                        if container_status.restart_count > 5:
                            high_restarts += 1

                        if container_status.last_state and container_status.last_state.terminated:
                            if container_status.last_state.terminated.reason == "OOMKilled":
                                finished_at = container_status.last_state.terminated.finished_at
                                if finished_at:
                                    if isinstance(finished_at, str):
                                        finished_at = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                                    if finished_at > datetime.now(timezone.utc) - timedelta(hours=24):
                                        oomkilled_24h += 1

            if "k8s.pods" not in metrics:
                metrics["k8s.pods"] = []

            metrics["k8s.pods"].extend(
                [
                    (crash_looping, ["issue:crash_looping"]),
                    (failed, ["phase:failed"]),
                    (pending, ["phase:pending"]),
                    (oomkilled_24h, ["issue:oomkilled", "timeframe:24h"]),
                    (high_restarts, ["issue:high_restarts"]),
                    (image_pull_errors, ["issue:image_pull_error"]),
                ]
            )

        except Exception as e:
            self.logger.error(f"Failed to collect pod health metrics: {e}", exc_info=True)

        return metrics

    def _collect_underutilization(self) -> dict[str, Any]:
        metrics = {}

        try:
            try:
                custom_api = client.CustomObjectsApi()
                pod_metrics = custom_api.list_cluster_custom_object(
                    group="metrics.k8s.io", version="v1beta1", plural="podmetrics"
                )
            except Exception as e:
                self.logger.warning(f"Could not fetch pod metrics for underutilization: {e}")
                return metrics

            idle_count = 0
            low_cpu = 0
            low_memory = 0

            for pod_metric in pod_metrics.get("items", []):
                cpu_usage = 0.0
                memory_usage = 0.0

                for container in pod_metric.get("containers", []):
                    cpu_str = container["usage"].get("cpu", "0")
                    if cpu_str.endswith("n"):
                        cpu_usage += float(cpu_str[:-1]) / 1_000_000_000
                    elif cpu_str.endswith("m"):
                        cpu_usage += float(cpu_str[:-1]) / 1000
                    else:
                        cpu_usage += float(cpu_str)

                    mem_str = container["usage"].get("memory", "0")
                    if mem_str.endswith("Mi"):
                        memory_usage += float(mem_str[:-2])
                    elif mem_str.endswith("Gi"):
                        memory_usage += float(mem_str[:-2]) * 1024

                if cpu_usage < 0.001 and memory_usage < 10:
                    idle_count += 1
                if cpu_usage < 0.01:
                    low_cpu += 1
                if memory_usage < 50:
                    low_memory += 1

            if "k8s.pods" not in metrics:
                metrics["k8s.pods"] = []

            metrics["k8s.pods"].extend(
                [
                    (idle_count, ["utilization:idle"]),
                    (low_cpu, ["utilization:low_cpu"]),
                    (low_memory, ["utilization:low_memory"]),
                ]
            )

            deployments = self.apps_v1.list_deployment_for_all_namespaces()
            zero_replicas = sum(1 for deployment in deployments.items if deployment.spec.replicas == 0)
            metrics["k8s.deployments"] = [(zero_replicas, ["status:zero_replicas"])]

        except Exception as e:
            self.logger.error(f"Failed to collect underutilization metrics: {e}", exc_info=True)

        return metrics
