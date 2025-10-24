"""Kubernetes resource efficiency and health collector for Datadog monitoring.

Tracks:
- Resource waste (CPU/memory overallocation)
- Pod health (crashes, restarts, failures)
- Underutilization (idle/underperforming pods)
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from kubernetes import client, config

from devops.datadog.common.base import BaseCollector


class KubernetesCollector(BaseCollector):
    """Collector for Kubernetes cluster efficiency and health metrics.

    Monitors resource waste, pod health issues, and underutilization to help
    optimize cluster costs and reliability.
    """

    def __init__(self):
        """Initialize Kubernetes collector with cluster connection."""
        super().__init__(name="kubernetes")

        # Load Kubernetes config (in-cluster or kubeconfig)
        try:
            config.load_incluster_config()
            self.logger.info("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            config.load_kube_config()
            self.logger.info("Loaded kubeconfig from local environment")

        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()

    def collect_metrics(self) -> dict[str, Any]:
        """Collect all Kubernetes efficiency and health metrics."""
        metrics = {}

        # Collect resource efficiency metrics
        metrics.update(self._collect_resource_efficiency())

        # Collect pod health metrics
        metrics.update(self._collect_pod_health())

        # Collect underutilization metrics
        metrics.update(self._collect_underutilization())

        return metrics

    def _collect_resource_efficiency(self) -> dict[str, Any]:
        """Calculate resource waste and efficiency scores."""
        metrics = {
            "k8s.resources.cpu_waste_cores": 0.0,
            "k8s.resources.memory_waste_gb": 0.0,
            "k8s.resources.cpu_efficiency_pct": None,
            "k8s.resources.memory_efficiency_pct": None,
            "k8s.resources.overallocated_pods": 0,
        }

        try:
            # Get all pods with resource requests
            pods = self.core_v1.list_pod_for_all_namespaces()

            total_cpu_requested = 0.0
            total_cpu_used = 0.0
            total_memory_requested = 0.0
            total_memory_used = 0.0
            overallocated_count = 0

            # Get pod metrics from metrics-server
            try:
                custom_api = client.CustomObjectsApi()
                pod_metrics = custom_api.list_cluster_custom_object(
                    group="metrics.k8s.io", version="v1beta1", plural="podmetrics"
                )
            except Exception as e:
                self.logger.warning(f"Could not fetch pod metrics: {e}")
                pod_metrics = None

            # Build usage map from metrics
            usage_map = {}
            if pod_metrics:
                for pod_metric in pod_metrics.get("items", []):
                    namespace = pod_metric["metadata"]["namespace"]
                    name = pod_metric["metadata"]["name"]
                    key = f"{namespace}/{name}"

                    cpu_usage = 0.0
                    memory_usage = 0.0

                    for container in pod_metric.get("containers", []):
                        # Parse CPU (e.g., "34m" -> 0.034 cores)
                        cpu_str = container["usage"].get("cpu", "0")
                        if cpu_str.endswith("n"):
                            cpu_usage += float(cpu_str[:-1]) / 1_000_000_000
                        elif cpu_str.endswith("m"):
                            cpu_usage += float(cpu_str[:-1]) / 1000
                        else:
                            cpu_usage += float(cpu_str)

                        # Parse memory (e.g., "5307Mi" -> bytes)
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

            # Calculate resource requests and compare with usage
            for pod in pods.items:
                if pod.status.phase not in ["Running", "Pending"]:
                    continue

                namespace = pod.metadata.namespace
                name = pod.metadata.name
                key = f"{namespace}/{name}"

                # Sum resource requests across containers
                cpu_requested = 0.0
                memory_requested = 0.0

                for container in pod.spec.containers:
                    if container.resources and container.resources.requests:
                        # Parse CPU request
                        cpu_req = container.resources.requests.get("cpu", "0")
                        if cpu_req.endswith("m"):
                            cpu_requested += float(cpu_req[:-1]) / 1000
                        else:
                            cpu_requested += float(cpu_req)

                        # Parse memory request
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

                # Get actual usage
                if key in usage_map:
                    cpu_used = usage_map[key]["cpu"]
                    memory_used = usage_map[key]["memory"]

                    total_cpu_used += cpu_used
                    total_memory_used += memory_used

                    # Detect overallocation (using < 20% of requested resources)
                    if cpu_requested > 0 and cpu_used / cpu_requested < 0.2:
                        overallocated_count += 1
                    elif memory_requested > 0 and memory_used / memory_requested < 0.2:
                        overallocated_count += 1

            # Calculate waste and efficiency
            if total_cpu_requested > 0:
                cpu_waste = total_cpu_requested - total_cpu_used
                metrics["k8s.resources.cpu_waste_cores"] = max(0.0, cpu_waste)
                metrics["k8s.resources.cpu_efficiency_pct"] = (total_cpu_used / total_cpu_requested) * 100

            if total_memory_requested > 0:
                memory_waste_bytes = total_memory_requested - total_memory_used
                metrics["k8s.resources.memory_waste_gb"] = max(0.0, memory_waste_bytes / (1024**3))
                metrics["k8s.resources.memory_efficiency_pct"] = (total_memory_used / total_memory_requested) * 100

            metrics["k8s.resources.overallocated_pods"] = overallocated_count

        except Exception as e:
            self.logger.error(f"Failed to collect resource efficiency metrics: {e}", exc_info=True)

        return metrics

    def _collect_pod_health(self) -> dict[str, Any]:
        """Track pod failures, restarts, and health issues."""
        metrics = {
            "k8s.pods.crash_looping": 0,
            "k8s.pods.failed": 0,
            "k8s.pods.pending": 0,
            "k8s.pods.oomkilled_24h": 0,
            "k8s.pods.high_restarts": 0,
            "k8s.pods.image_pull_errors": 0,
        }

        try:
            pods = self.core_v1.list_pod_for_all_namespaces()

            for pod in pods.items:
                # Count pod states
                if pod.status.phase == "Failed":
                    metrics["k8s.pods.failed"] += 1
                elif pod.status.phase == "Pending":
                    metrics["k8s.pods.pending"] += 1

                # Check container statuses
                if pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        # Crash looping
                        if container_status.state and container_status.state.waiting:
                            if container_status.state.waiting.reason == "CrashLoopBackOff":
                                metrics["k8s.pods.crash_looping"] += 1

                        # Image pull errors
                        if container_status.state and container_status.state.waiting:
                            if "ImagePull" in (container_status.state.waiting.reason or ""):
                                metrics["k8s.pods.image_pull_errors"] += 1

                        # High restart count (>5 restarts)
                        if container_status.restart_count > 5:
                            metrics["k8s.pods.high_restarts"] += 1

                        # OOMKilled in last 24 hours
                        if container_status.last_state and container_status.last_state.terminated:
                            if container_status.last_state.terminated.reason == "OOMKilled":
                                finished_at = container_status.last_state.terminated.finished_at
                                if finished_at:
                                    if isinstance(finished_at, str):
                                        finished_at = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
                                    if finished_at > datetime.now(timezone.utc) - timedelta(hours=24):
                                        metrics["k8s.pods.oomkilled_24h"] += 1

        except Exception as e:
            self.logger.error(f"Failed to collect pod health metrics: {e}", exc_info=True)

        return metrics

    def _collect_underutilization(self) -> dict[str, Any]:
        """Detect idle and underperforming pods."""
        metrics = {
            "k8s.pods.idle_count": 0,
            "k8s.pods.low_cpu_usage": 0,
            "k8s.pods.low_memory_usage": 0,
            "k8s.deployments.zero_replicas": 0,
        }

        try:
            # Get pod metrics
            try:
                custom_api = client.CustomObjectsApi()
                pod_metrics = custom_api.list_cluster_custom_object(
                    group="metrics.k8s.io", version="v1beta1", plural="podmetrics"
                )
            except Exception as e:
                self.logger.warning(f"Could not fetch pod metrics for underutilization: {e}")
                return metrics

            # Check for very low resource usage
            for pod_metric in pod_metrics.get("items", []):
                cpu_usage = 0.0
                memory_usage = 0.0

                for container in pod_metric.get("containers", []):
                    # Parse CPU
                    cpu_str = container["usage"].get("cpu", "0")
                    if cpu_str.endswith("n"):
                        cpu_usage += float(cpu_str[:-1]) / 1_000_000_000
                    elif cpu_str.endswith("m"):
                        cpu_usage += float(cpu_str[:-1]) / 1000
                    else:
                        cpu_usage += float(cpu_str)

                    # Parse memory
                    mem_str = container["usage"].get("memory", "0")
                    if mem_str.endswith("Mi"):
                        memory_usage += float(mem_str[:-2])
                    elif mem_str.endswith("Gi"):
                        memory_usage += float(mem_str[:-2]) * 1024

                # Detect idle pods (< 1m CPU and < 10Mi memory)
                if cpu_usage < 0.001 and memory_usage < 10:
                    metrics["k8s.pods.idle_count"] += 1

                # Detect low CPU usage (< 10m)
                if cpu_usage < 0.01:
                    metrics["k8s.pods.low_cpu_usage"] += 1

                # Detect low memory usage (< 50Mi)
                if memory_usage < 50:
                    metrics["k8s.pods.low_memory_usage"] += 1

            # Check for deployments with 0 replicas
            deployments = self.apps_v1.list_deployment_for_all_namespaces()
            for deployment in deployments.items:
                if deployment.spec.replicas == 0:
                    metrics["k8s.deployments.zero_replicas"] += 1

        except Exception as e:
            self.logger.error(f"Failed to collect underutilization metrics: {e}", exc_info=True)

        return metrics
