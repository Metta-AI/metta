#!/bin/bash
set -e

echo "🚀 Setting up Metta Docker Sandbox..."

# Check prerequisites
for tool in docker docker-compose; do
    if ! command -v "$tool" &> /dev/null; then
        echo "❌ Please install $tool"
        exit 1
    fi
done

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f metta/sweep/docker/docker-compose.yml down 2>/dev/null || true
docker rm -f metta-sweep-master metta-sweep-worker 2>/dev/null || true

# Build and start the Docker environment
echo "📦 Building and starting Docker environment..."
docker-compose -f metta/sweep/docker/docker-compose.yml up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Check if both containers are running
echo "🔍 Checking container status..."
MASTER_RUNNING=$(docker ps --filter "name=metta-sweep-master" --format "{{.Names}}" | grep -c "metta-sweep-master" || echo "0")
WORKER_RUNNING=$(docker ps --filter "name=metta-sweep-worker" --format "{{.Names}}" | grep -c "metta-sweep-worker" || echo "0")

if [ "$MASTER_RUNNING" = "1" ] && [ "$WORKER_RUNNING" = "1" ]; then
    echo "✅ Both containers are running!"
    echo "   📍 Master: metta-sweep-master"
    echo "   📍 Worker: metta-sweep-worker"
    echo ""
    echo "🎯 Single-node test:"
    echo "   ./devops/docker_launch.py sweep run=test trainer.total_timesteps=50 +hardware=macbook"
    echo ""
    echo "🌐 Multi-node test:"
    echo "   ./devops/docker_launch.py sweep run=test trainer.total_timesteps=50 +hardware=macbook --nodes=2"
    echo ""
    echo "🛑 To stop:"
    echo "   docker-compose -f metta/sweep/docker/docker-compose.yml down"
else
    echo "❌ Container setup failed!"
    echo "   Master running: $MASTER_RUNNING/1"
    echo "   Worker running: $WORKER_RUNNING/1"
    echo ""
    echo "📋 Check logs with:"
    echo "   docker-compose -f metta/sweep/docker/docker-compose.yml logs"
    exit 1
fi
