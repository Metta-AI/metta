docker build -f devops/docker/Dockerfile.base -t docker.io/mettaai/metta-base:latest .
docker build -f devops/docker/Dockerfile -t docker.io/mettaai/metta:latest .
