branch=$(git rev-parse --abbrev-ref HEAD)
docker build -f devops/docker/Dockerfile.base --build-arg CACHE_DATE=$(date +%Y%m%d_%H%M%S) --build-arg BRANCH=$branch -t mettaai/metta-base:latest .
docker build -f devops/docker/Dockerfile --build-arg CACHE_DATE=$(date +%Y%m%d_%H%M%S) --build-arg BRANCH=$branch -t mettaai/metta:latest .
