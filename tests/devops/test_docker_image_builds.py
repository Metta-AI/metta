import pytest

from metta.common.util.fs import get_repo_root


class TestDockerImagesBuilds:
    # Example flaky run: https://github.com/Metta-AI/metta/actions/runs/16970332995/job/48105323797
    @pytest.mark.skip(reason="Flaky test")
    def test_observatory_frontend_image(self, docker_client):
        project_root = get_repo_root()
        client = docker_client
        image, build_logs = client.images.build(
            path=str(project_root),
            dockerfile="observatory/Dockerfile",
            tag="test-observatory-frontend:latest",
            rm=True,
        )
        assert image is not None
        assert build_logs is not None
