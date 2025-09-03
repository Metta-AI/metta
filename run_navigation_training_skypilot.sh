#!/bin/bash

# Navigation Training Script using SkyPilot
# Launches navigation training with both learning progress and random curricula
# on cloud infrastructure with 4 nodes and 2 GPUs per node

set -e  # Exit on any error

echo "🚀 Starting Navigation Training Comparison via SkyPilot"
echo "======================================================"

# Configuration
NODES=1
GPUS_PER_NODE=1

echo "📊 Training Configuration:"
echo "   Cloud infrastructure: $NODES nodes × $GPUS_PER_NODE GPUs"
echo "   Using default trainer configuration"
echo ""

# Function to launch training via SkyPilot
launch_training() {
    local curriculum_type=$1
    local use_lp=$2
    local run_name="bullm_navigation_${curriculum_type}_$(date +%m%d_%H%M%S)"

    echo "🎯 Launching $curriculum_type curriculum training via SkyPilot..."
    echo "   Run name: $run_name"
    echo "   Learning progress: $use_lp"
    echo "   Resources: $NODES nodes × $GPUS_PER_NODE GPUs"
    echo ""

    # Build the launch command
    local launch_cmd=(
        "uv" "run" "--active" "devops/skypilot/launch.py"
        "experiments.recipes.navigation.train"
        "--run" "$run_name"
        "--nodes" "$NODES"
        "--gpus" "$GPUS_PER_NODE"
        "--max-runtime-hours" "24"
        "--confirm"
        "--skip-git-check"
        "--args"
        "run=$run_name"
        "use_learning_progress=$use_lp"
    )

    echo "🚀 Executing: ${launch_cmd[*]}"
    echo ""

    # Execute the launch command
    "${launch_cmd[@]}"

    echo "✅ $curriculum_type training launched successfully"
    echo "   Run ID: $run_name"
    echo "   Check SkyPilot dashboard for status"
    echo ""
}

# Check if SkyPilot is available
if ! command -v sky &> /dev/null; then
    echo "❌ Error: SkyPilot is not installed or not in PATH"
    echo "   Please install SkyPilot first:"
    echo "   curl -sSL https://install.sky.run | bash"
    echo "   source ~/.bashrc  # or restart your terminal"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "devops/skypilot/launch.py" ]; then
    echo "❌ Error: devops/skypilot/launch.py not found"
    echo "   Please run this script from the metta repository root"
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed or not in PATH"
    echo "   Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   source ~/.bashrc  # or restart your terminal"
    exit 1
fi

echo "🔍 Pre-flight checks passed"
echo ""

# Ask for confirmation before launching expensive cloud jobs
echo "⚠️  WARNING: This will launch cloud jobs with significant costs"
echo "   - $NODES nodes × $GPUS_PER_NODE GPUs = $((NODES * GPUS_PER_NODE)) total GPUs"
echo "   - Estimated cost: $((NODES * 10))-$$((NODES * 20)) per hour (varies by region)"
echo "   - Total estimated cost for 24 hours: $((NODES * 240))-$$((NODES * 480))"
echo ""
read -p "Do you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Launch cancelled"
    exit 0
fi

echo ""

# Launch learning progress curriculum training
echo "🔄 Starting Learning Progress Curriculum Training..."
launch_training "learning_progress" "true"

# Wait a moment between launches
sleep 10

# Launch random curriculum training
echo "🔄 Starting Random Curriculum Training..."
launch_training "random" "false"

echo "🎉 All training jobs launched successfully!"
echo ""
echo "📊 Job Summary:"
echo "   - Learning Progress: Launched via SkyPilot"
echo "   - Random Curriculum: Launched via SkyPilot"
echo ""
echo "📈 Next Steps:"
echo "   1. Check SkyPilot dashboard: sky queue"
echo "   2. Monitor job status: sky logs <job_id>"
echo "   3. View WandB experiments in the metta project"
echo "   4. Compare training progress between curricula"
echo ""
echo "💡 Tips:"
echo "   - Use 'sky cancel <job_id>' to stop a job early"
echo "   - Use 'sky logs <job_id>' to view real-time logs"
echo "   - Check 'sky status' for overall cluster status"
