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

# Build and start the Docker environment
echo "📦 Building and starting Docker environment..."
docker-compose -f metta/sweep/docker/docker-compose.yml up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if containers are running
echo "🔍 Checking container status..."
if docker ps | grep -q metta-sweep-master; then
    echo "✅ Docker sandbox is ready!"
    echo ""
    echo "🎯 Quick test:"
    echo "   ./devops/docker_launch.py sweep run=test trainer.total_timesteps=50 trainer.curriculum=/env/mettagrid/curriculum/arena/random +hardware=macbook"
    echo ""
    echo "🛑 To stop:"
    echo "   docker-compose -f metta/sweep/docker/docker-compose.yml down"
else
    echo "❌ Container setup failed"
    exit 1
fi
