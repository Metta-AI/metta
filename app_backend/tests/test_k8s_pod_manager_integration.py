"""
Integration tests for K8sPodManager using kind (Kubernetes in Docker).

This test suite:
1. Spins up a local Kubernetes cluster using kind
2. Deploys a test backend service
3. Tests the K8sPodManager implementation
4. Cleans up all resources
"""

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from typing import Any, Dict, List

import pytest
import yaml

from metta.app_backend.container_managers.k8s import K8sPodManager


class TestK8sPodManagerIntegration:
    """Integration tests for K8sPodManager using kind."""

    @classmethod
    def setup_class(cls):
        """Set up logging."""
        cls.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        cls.logger.addHandler(handler)
        cls.logger.setLevel(logging.INFO)

    @pytest.fixture(scope="class")
    def kind_cluster_name(self):
        """Generate unique cluster name."""
        return f"metta-test-{uuid.uuid4().hex[:8]}"

    @pytest.fixture(scope="class")
    def kind_config(self):
        """Kind cluster configuration."""
        return {
            "kind": "Cluster",
            "apiVersion": "kind.x-k8s.io/v1alpha4",
            "nodes": [{"role": "control-plane"}, {"role": "worker"}],
        }

    @pytest.fixture(scope="class")
    def kind_cluster(self, kind_cluster_name, kind_config):
        """Create and manage kind cluster."""
        subprocess.run(["kind", "version"], capture_output=True, check=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(kind_config, f)
            config_file = f.name

        try:
            # Create cluster
            self.logger.info(f"Creating kind cluster: {kind_cluster_name}")
            subprocess.run(
                ["kind", "create", "cluster", "--name", kind_cluster_name, "--config", config_file], check=True
            )

            # Get kubeconfig
            kubeconfig_result = subprocess.run(
                ["kind", "get", "kubeconfig", "--name", kind_cluster_name], capture_output=True, text=True, check=True
            )

            # Write kubeconfig to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".kubeconfig", delete=False) as f:
                f.write(kubeconfig_result.stdout)
                kubeconfig_path = f.name

            yield kubeconfig_path

        finally:
            # Delete cluster
            self.logger.info(f"Deleting kind cluster: {kind_cluster_name}")
            subprocess.run(["kind", "delete", "cluster", "--name", kind_cluster_name], check=False)

            # Clean up temp files
            if "config_file" in locals():
                os.unlink(config_file)
            if "kubeconfig_path" in locals():
                os.unlink(kubeconfig_path)

    @pytest.fixture(scope="class")
    def test_namespace(self, kind_cluster):
        """Create test namespace."""
        namespace = f"test-{uuid.uuid4().hex[:8]}"

        # Create namespace
        subprocess.run(["kubectl", "--kubeconfig", kind_cluster, "create", "namespace", namespace], check=True)

        yield namespace

        # Delete namespace (will cascade delete all resources)
        subprocess.run(["kubectl", "--kubeconfig", kind_cluster, "delete", "namespace", namespace], check=False)

    @pytest.fixture(scope="class")
    def mock_worker_image_in_kind(self, kind_cluster, kind_cluster_name):
        """Load mock worker image into kind cluster."""
        # First check if the image exists locally
        try:
            subprocess.run(["docker", "image", "inspect", "test-worker:latest"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # Build the test worker image if it doesn't exist
            self.logger.info("Building test worker image")
            subprocess.run(
                ["docker", "build", "-f", "app_backend/tests/Dockerfile.test_worker", "-t", "test-worker:latest", "."],
                check=True,
            )

        # Load image into kind
        self.logger.info("Loading test worker image into kind cluster")
        subprocess.run(["kind", "load", "docker-image", "test-worker:latest", "--name", kind_cluster_name], check=True)

        return "test-worker:latest"

    @pytest.fixture(scope="class")
    def mock_backend_deployment(self, kind_cluster, test_namespace):
        """Deploy a simple mock backend for testing."""
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "mock-backend", "namespace": test_namespace},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": "mock-backend"}},
                "template": {
                    "metadata": {"labels": {"app": "mock-backend"}},
                    "spec": {
                        "containers": [{"name": "nginx", "image": "nginx:alpine", "ports": [{"containerPort": 80}]}]
                    },
                },
            },
        }

        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "mock-backend", "namespace": test_namespace},
            "spec": {"selector": {"app": "mock-backend"}, "ports": [{"port": 80, "targetPort": 80}]},
        }

        # Apply manifests
        for manifest in [deployment_manifest, service_manifest]:
            cmd = ["kubectl", "--kubeconfig", kind_cluster, "create", "-f", "-"]
            subprocess.run(cmd, input=json.dumps(manifest), text=True, check=True)

        # Wait for deployment to be ready
        subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                kind_cluster,
                "-n",
                test_namespace,
                "wait",
                "--for=condition=available",
                "deployment/mock-backend",
                "--timeout=60s",
            ],
            check=True,
        )

        return f"http://mock-backend.{test_namespace}.svc.cluster.local"

    @pytest.fixture
    def k8s_pod_manager(self, kind_cluster, test_namespace):
        """Create K8sPodManager instance for testing."""
        # Set WANDB_API_KEY for testing (required by K8sPodManager)
        os.environ["WANDB_API_KEY"] = "test-api-key"
        from metta.app_backend.container_managers.k8s import K8sPodManager

        return K8sPodManager(namespace=test_namespace, kubeconfig=kind_cluster)

    # Helper methods

    def get_pods(self, kubeconfig: str, namespace: str, label_selector: str = "") -> List[Dict[str, Any]]:
        """Get pods in namespace."""
        cmd = ["kubectl", "--kubeconfig", kubeconfig, "-n", namespace, "get", "pods", "-o", "json"]
        if label_selector:
            cmd.extend(["-l", label_selector])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout).get("items", [])

    def wait_for_pod_phase(self, kubeconfig: str, namespace: str, pod_name: str, phase: str, timeout: int = 30):
        """Wait for pod to reach specific phase."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            pods = self.get_pods(kubeconfig, namespace)
            for pod in pods:
                if pod["metadata"]["name"] == pod_name:
                    if pod["status"]["phase"] == phase:
                        return
            time.sleep(1)
        raise TimeoutError(f"Pod {pod_name} did not reach phase {phase} within {timeout}s")

    # Tests

    @pytest.mark.asyncio
    async def test_start_worker_pod(
        self, k8s_pod_manager, kind_cluster, test_namespace, mock_backend_deployment, mock_worker_image_in_kind
    ):
        """Test starting a worker pod."""
        git_hash = "test-hash-1"

        # Start worker
        worker_info = k8s_pod_manager.start_worker_container(
            git_hash=git_hash, backend_url=mock_backend_deployment, docker_image=mock_worker_image_in_kind
        )

        assert worker_info.git_hash == git_hash
        assert worker_info.container_name.startswith("eval-worker-")
        assert worker_info.container_id  # Should have UID
        assert worker_info.alive

        # Verify pod exists
        pods = self.get_pods(kind_cluster, test_namespace, "app=eval-worker")
        assert len(pods) == 1
        assert pods[0]["metadata"]["name"] == worker_info.container_name
        assert pods[0]["metadata"]["labels"]["git-hash"] == git_hash

    @pytest.mark.asyncio
    async def test_discover_alive_workers(
        self, k8s_pod_manager, kind_cluster, test_namespace, mock_backend_deployment, mock_worker_image_in_kind
    ):
        """Test discovering alive workers."""
        # Start multiple workers
        git_hashes = ["discover-1", "discover-2", "discover-3"]
        started_workers = []

        for git_hash in git_hashes:
            worker = k8s_pod_manager.start_worker_container(
                git_hash=git_hash, backend_url=mock_backend_deployment, docker_image=mock_worker_image_in_kind
            )
            started_workers.append(worker)

        # Wait for pods to be running
        for worker in started_workers:
            self.wait_for_pod_phase(kind_cluster, test_namespace, worker.container_name, "Running")

        # Discover workers
        discovered = await k8s_pod_manager.discover_alive_workers()

        # Filter to only the workers we started in this test
        discovered_in_test = [w for w in discovered if w.git_hash in git_hashes]

        assert len(discovered_in_test) == 3
        discovered_hashes = {w.git_hash for w in discovered_in_test}
        assert discovered_hashes == set(git_hashes)

    @pytest.mark.asyncio
    async def test_cleanup_pod(
        self, k8s_pod_manager, kind_cluster, test_namespace, mock_backend_deployment, mock_worker_image_in_kind
    ):
        """Test cleaning up a pod."""
        # Start a worker
        worker = k8s_pod_manager.start_worker_container(
            git_hash="cleanup-test", backend_url=mock_backend_deployment, docker_image=mock_worker_image_in_kind
        )

        # Verify it exists
        pods_before = self.get_pods(kind_cluster, test_namespace, "app=eval-worker")
        pod_names_before = {p["metadata"]["name"] for p in pods_before}
        assert worker.container_name in pod_names_before

        # Clean up by container ID (UID)
        k8s_pod_manager.cleanup_container(worker.container_id)

        # Wait a bit for deletion
        time.sleep(2)

        # Verify it's gone
        pods_after = self.get_pods(kind_cluster, test_namespace, "app=eval-worker")
        pod_names_after = {p["metadata"]["name"] for p in pods_after}
        assert worker.container_name not in pod_names_after

    @pytest.mark.asyncio
    async def test_cleanup_by_name(
        self, k8s_pod_manager, kind_cluster, test_namespace, mock_backend_deployment, mock_worker_image_in_kind
    ):
        """Test cleaning up a pod by name."""
        # Start a worker
        worker = k8s_pod_manager.start_worker_container(
            git_hash="cleanup-name-test", backend_url=mock_backend_deployment, docker_image=mock_worker_image_in_kind
        )

        # Clean up by name
        k8s_pod_manager.cleanup_container(worker.container_name)

        # Wait and verify
        time.sleep(2)
        pods = self.get_pods(kind_cluster, test_namespace, "app=eval-worker")
        pod_names = {p["metadata"]["name"] for p in pods}
        assert worker.container_name not in pod_names

    @pytest.mark.asyncio
    async def test_namespace_isolation(
        self, kind_cluster, test_namespace, mock_backend_deployment, mock_worker_image_in_kind
    ):
        """Test that manager only sees pods in its namespace."""
        # Create another namespace
        other_namespace = f"other-{uuid.uuid4().hex[:8]}"
        subprocess.run(["kubectl", "--kubeconfig", kind_cluster, "create", "namespace", other_namespace], check=True)

        try:
            # Create managers for each namespace
            manager1 = K8sPodManager(namespace=test_namespace, kubeconfig=kind_cluster)
            manager2 = K8sPodManager(namespace=other_namespace, kubeconfig=kind_cluster)

            # Start workers in each namespace
            manager1.start_worker_container(
                git_hash="namespace-1", backend_url=mock_backend_deployment, docker_image=mock_worker_image_in_kind
            )

            manager2.start_worker_container(
                git_hash="namespace-2", backend_url=mock_backend_deployment, docker_image=mock_worker_image_in_kind
            )

            # Each manager should only see its own pods
            discovered1 = await manager1.discover_alive_workers()
            discovered2 = await manager2.discover_alive_workers()

            assert len(discovered1) == 1
            assert discovered1[0].git_hash == "namespace-1"

            assert len(discovered2) == 1
            assert discovered2[0].git_hash == "namespace-2"

        finally:
            # Clean up other namespace
            subprocess.run(
                ["kubectl", "--kubeconfig", kind_cluster, "delete", "namespace", other_namespace], check=False
            )

    def test_error_handling(self, k8s_pod_manager, kind_cluster, test_namespace):
        """Test error handling for invalid operations."""
        # Try to start worker with invalid image - it will create the pod but fail to start
        worker = k8s_pod_manager.start_worker_container(
            git_hash="error-test", backend_url="http://backend", docker_image="invalid-image:does-not-exist"
        )

        # Wait a bit and check that the pod is in error state
        time.sleep(5)
        pods = self.get_pods(kind_cluster, test_namespace, "git-hash=error-test")
        assert len(pods) == 1
        pod = pods[0]

        # Check that the pod is in error state (ErrImageNeverPull or similar)
        container_statuses = pod.get("status", {}).get("containerStatuses", [])
        if container_statuses:
            waiting = container_statuses[0].get("state", {}).get("waiting", {})
            assert waiting.get("reason") in [
                "ErrImageNeverPull",
                "InvalidImageName",
                "ImagePullBackOff",
                "ErrImagePull",
            ]

        # Cleanup the error pod
        k8s_pod_manager.cleanup_container(worker.container_id)

        # Cleanup non-existent pod should not raise
        k8s_pod_manager.cleanup_container("non-existent-pod-id")
