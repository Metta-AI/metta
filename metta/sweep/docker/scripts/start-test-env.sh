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

# Setup environment (venv already active via Dockerfile ENV)
echo "🐍 Setting up environment..."
source devops/setup.env

# Note: Dummy API removed - using real APIs or disabling API calls for testing

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
        echo "✅ Worker node ready for distributed training!"
        echo ""
        echo "🔥 Ready to receive commands via docker exec"
        echo ""

        # Keep container running and ready for commands
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
