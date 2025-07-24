import docker

from metta.common.util.fs import get_repo_root


class TestDockerImagesBuilds:
    def test_observatory_frontend_image(self):
        project_root = get_repo_root()
        client = docker.from_env()
        image, build_logs = client.images.build(
            path=str(project_root / "observatory"),
            dockerfile="Dockerfile",
            tag="test-observatory-frontend:latest",
            rm=True,
        )
        assert image is not None
        assert build_logs is not None
