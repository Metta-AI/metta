import logging
import time

import pytest
import requests
from testcontainers.core.container import DockerContainer
from testcontainers.postgres import PostgresContainer

from metta.common.util.fs import get_repo_root


class TestDockerIntegration:
    """Integration tests for the app_backend Docker container."""

    @classmethod
    def setup_class(cls):
        """Set up logging for the test class."""
        cls.logger = logging.getLogger(__name__)
        if not cls.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            cls.logger.addHandler(handler)
            cls.logger.setLevel(logging.INFO)

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Create a PostgreSQL container for the backend to connect to."""
        container = None
        try:
            self.logger.info("Starting PostgreSQL container")
            container = PostgresContainer(
                image="postgres:17",
                username="test_user",
                password="test_password",
                dbname="test_db",
                driver=None,
            )
            container.start()
            self.logger.info("Successfully started PostgreSQL container")
            yield container
        except Exception as e:
            self.logger.error(f"Failed to start PostgreSQL container: {e}")
            pytest.skip(f"Failed to start PostgreSQL container: {e}")
        finally:
            if container:
                try:
                    self.logger.info("Stopping PostgreSQL container")
                    container.stop()
                    self.logger.info("Successfully stopped PostgreSQL container")
                except Exception as e:
                    self.logger.error(f"Failed to stop PostgreSQL container: {e}")

    @pytest.fixture(scope="class")
    def app_backend_container(self, postgres_container: PostgresContainer, docker_client):
        """Build and start the app_backend Docker container."""
        try:
            import docker

            project_root = get_repo_root()
            client = docker_client

            # Build the Docker image first
            self.logger.info("Building Docker image for app_backend")
            image, build_logs = client.images.build(
                path=str(project_root), dockerfile="app_backend/Dockerfile", tag="test-app-backend:latest", rm=True
            )
            self.logger.info("Successfully built Docker image")

            # Create a shared network for container communication
            # Try to remove existing network first, then create new one
            try:
                existing_network = client.networks.get("test-app-backend-network")
                self.logger.info("Found existing network, removing it")
                existing_network.remove()
                self.logger.info("Successfully removed existing network")
            except docker.errors.NotFound:  # type: ignore[reportAttributeAccessIssue]
                self.logger.info("No existing network found, proceeding with creation")
            except Exception as e:
                self.logger.warning(f"Failed to remove existing network: {e}")

            network = client.networks.create("test-app-backend-network")
            self.logger.info("Created Docker network")

            # Connect postgres container to the network
            self.logger.info("Connecting PostgreSQL container to network")
            network.connect(postgres_container._container)  # type: ignore[reportArgumentType]

            # Use the postgres container name as hostname for internal communication
            db_host = postgres_container._container.name  # type: ignore[reportOptionalMemberAccess]
            db_uri = f"postgresql://test_user:test_password@{db_host}:5432/test_db"
            self.logger.info(f"Database URI: {db_uri}")

            # Start the app container
            self.logger.info("Starting app backend container")
            container = (
                DockerContainer(image="test-app-backend:latest")
                .with_exposed_ports(8000)
                .with_env("STATS_DB_URI", db_uri)
                .with_env(
                    "DEBUG_USER_EMAIL",
                    "",  # Disable debug mode
                )
                .with_kwargs(network=network.name)
            )

            container.start()
            self.logger.info("App backend container started")

            # Wait for the service to be ready
            host = container.get_container_host_ip()
            port = container.get_exposed_port(8000)
            self.logger.info(f"Waiting for service to be ready at {host}:{port}")

            # Wait up to 200 seconds for the service to respond
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    response = requests.get(f"http://{host}:{port}/whoami", timeout=2)
                    if response.status_code == 200:
                        self.logger.info(f"Service ready after {attempt + 1} attempts")
                        break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    if attempt == max_attempts - 1:
                        # Print container logs for debugging
                        logs = container.get_logs()
                        self.logger.error(f"Service failed to become ready after {max_attempts} attempts")
                        self.logger.error(f"Container logs: {logs}")
                        self.logger.error(
                            f"Container status: {container.get_container_host_ip()}:{container.get_exposed_port(8000)}"
                        )
                        raise
                    if attempt % 5 == 0:  # Log every 5 attempts
                        self.logger.info(f"Attempt {attempt + 1}/{max_attempts} - service not ready yet")
                    time.sleep(1)

            yield container, host, port

            # Cleanup phase - log all cleanup operations
            self.logger.info("Starting cleanup process")

            # Stop the app container
            try:
                self.logger.info("Stopping app container")
                container.stop()
                self.logger.info("Successfully stopped app container")
            except Exception as e:
                self.logger.error(f"Failed to stop app container: {e}")

            # Clean up the network
            try:
                self.logger.info("Disconnecting containers from Docker network")
                # Disconnect the postgres container from the network
                try:
                    network.disconnect(postgres_container._container)  # type: ignore[reportArgumentType]
                    self.logger.info("Disconnected PostgreSQL container from network")
                except Exception as disconnect_error:
                    self.logger.warning(f"Failed to disconnect PostgreSQL container: {disconnect_error}")

                self.logger.info("Removing Docker network")
                network.remove()
                self.logger.info("Successfully removed Docker network")
            except Exception as e:
                self.logger.error(f"Failed to remove Docker network: {e}")

            # Clean up the built image
            try:
                self.logger.info("Removing built Docker image")
                client.images.remove("test-app-backend:latest", force=True)
                self.logger.info("Successfully removed Docker image")
            except Exception as e:
                self.logger.error(f"Failed to remove Docker image: {e}")

            self.logger.info("Cleanup process completed")
        except Exception as e:
            pytest.fail(f"Failed to start app_backend container: {e}")

    @pytest.mark.slow
    def test_whoami_endpoint_no_auth(self, app_backend_container):
        """Test /whoami endpoint without authentication returns 'unknown'."""
        container, host, port = app_backend_container

        response = requests.get(f"http://{host}:{port}/whoami")

        assert response.status_code == 200
        data = response.json()
        assert data["user_email"] == "unknown"

    @pytest.mark.slow
    def test_whoami_endpoint_with_email_auth(self, app_backend_container):
        """Test /whoami endpoint with email authentication."""
        container, host, port = app_backend_container

        headers = {"X-Auth-Request-Email": "test@example.com"}
        response = requests.get(f"http://{host}:{port}/whoami", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["user_email"] == "test@example.com"

    @pytest.mark.slow
    def test_container_health_check(self, app_backend_container):
        """Test that the container is healthy and responding."""
        container, host, port = app_backend_container

        # Test basic connectivity
        response = requests.get(f"http://{host}:{port}/whoami")
        assert response.status_code == 200

        # Verify response format
        data = response.json()
        assert "user_email" in data
        assert isinstance(data["user_email"], str)

    @pytest.mark.slow
    def test_protected_endpoint_without_auth(self, app_backend_container):
        """Test that protected endpoints require authentication."""
        container, host, port = app_backend_container

        # Try to create a token without authentication
        response = requests.post(f"http://{host}:{port}/tokens", json={"name": "test_token"})

        assert response.status_code == 401

    @pytest.mark.slow
    def test_protected_endpoint_with_auth(self, app_backend_container):
        """Test that protected endpoints work with authentication."""
        container, host, port = app_backend_container

        headers = {"X-Auth-Request-Email": "test@example.com"}

        # Try to create a token with authentication
        response = requests.post(f"http://{host}:{port}/tokens", json={"name": "test_token"}, headers=headers)

        # Should succeed (200) or fail due to missing database tables (500)
        # but not fail due to authentication (401)
        assert response.status_code != 401


if __name__ == "__main__":
    # Simple test runner for debugging
    pytest.main([__file__, "-v", "-s"])
