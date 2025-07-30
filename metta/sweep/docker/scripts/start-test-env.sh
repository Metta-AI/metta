#!/bin/bash
set -e

echo "🚀 Starting Metta Sweep Test Environment..."
echo "   Hostname: $(hostname)"
echo "   Test Mode: ${TEST_MODE:-standalone}"
echo "   Rank: ${RANK:-N/A}"
echo "   World Size: ${WORLD_SIZE:-N/A}"

# Setup directories and permissions
echo "📁 Setting up directories..."
mkdir -p /home/metta/train_dir /tmp/ray /home/metta/test-results
sudo chown -R metta:metta /home/metta/train_dir /tmp/ray /home/metta/test-results 2>/dev/null || {
    echo "⚠️  Cannot change ownership (running without sudo)"
}

# Change to metta directory
cd /home/metta/metta

# Ensure virtual environment exists and is properly configured
echo "🐍 Setting up virtual environment..."
if [ ! -f ".venv/bin/activate" ] || [ ! -f ".venv/pyvenv.cfg" ]; then
    echo "   Creating virtual environment..."
    rm -rf .venv
    uv venv
    echo "   Installing dependencies..."
    uv sync
    echo "   ✅ Virtual environment created successfully"
else
    echo "   ✅ Virtual environment already exists"
fi

# Activate virtual environment and setup
echo "🐍 Activating environment..."
source .venv/bin/activate
source devops/setup.env

# Start dummy API server in background if we're the master node or standalone
if [ "${TEST_MODE:-standalone}" = "master" ] || [ "${RANK:-0}" = "0" ] || [ "${TEST_MODE:-standalone}" = "standalone" ]; then
    echo "🤖 Starting dummy API server..."
            python /home/metta/metta/test-scripts/dummy-api-server.py &

    # Wait a moment for the server to start
    sleep 2

    # Test the dummy API
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Dummy API server is running"
    else
        echo "⚠️  Dummy API server may not be responding"
    fi
fi

# Configure environment based on test mode
case "${TEST_MODE:-standalone}" in
    "master")
        echo "🎯 Configuring as master node (rank ${RANK:-0})"

        # Keep container running and ready for commands
        echo "✅ Master node ready for sweep commands!"
        echo ""
        echo "🔥 Quick test:"
        echo "   docker exec -it metta-sweep-master bash -c \"cd /home/metta/metta && source .venv/bin/activate && devops/sweep.sh run=test trainer.total_timesteps=50 +hardware=macbook\""
        echo ""
        tail -f /dev/null
        ;;

    "worker")
        echo "👷 Configuring as worker node (rank ${RANK})"

        # Wait for master to be ready
        echo "⏳ Waiting for master node..."
        until ping -c 1 sweep-master > /dev/null 2>&1; do
            echo "   Waiting for master node..."
            sleep 2
        done
        echo "✅ Master node is reachable"

        # Keep container running and wait for distributed commands
        echo "✅ Worker node ready!"
        tail -f /dev/null
        ;;

    *)
        echo "🖥️  Starting in standalone/interactive mode"
        echo "✅ Environment ready!"
        echo ""
        echo "🔥 Try running:"
        echo "   devops/sweep.sh run=standalone_test trainer.total_timesteps=100 +hardware=macbook"
        echo ""

        # Keep container running for manual testing
        bash
        ;;
esac
