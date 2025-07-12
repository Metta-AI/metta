#!/bin/bash
# Metta GPU Sandbox Container Runner
# Provides secure GPU-enabled containers for testing and isolation

set -euo pipefail

# Default values
CONTAINER_NAME="metta-gpu-sandbox-$$"
IMAGE="mettaai/metta:latest"
GPU_DEVICE="0"  # Default to first GPU
NETWORK_MODE="none"  # Default to isolated
MEMORY_LIMIT="32g"
SHM_SIZE="16g"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
REMOTE_MODE=false
DOCKER_HOST="${DOCKER_HOST:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run a secure Metta GPU sandbox container with isolation and resource limits.

OPTIONS:
    -n, --name NAME           Container name (default: metta-gpu-sandbox-<PID>)
    -i, --image IMAGE         Docker image (default: mettaai/metta:latest)
    -g, --gpu DEVICE          GPU device(s) to use (default: 0, use 'all' for all GPUs)
    -w, --network MODE        Network mode: none, bridge, host (default: none)
    -m, --memory LIMIT        Memory limit (default: 32g)
    -s, --shm-size SIZE       Shared memory size (default: 16g)
    -d, --workspace DIR       Workspace directory (default: current directory)
    -r, --remote HOST         Remote Docker host (e.g., ssh://user@host)
    -h, --help                Show this help message

SECURITY FEATURES:
    - Read-only root filesystem with specific writable mounts
    - Dropped capabilities (except SYS_PTRACE for debugging)
    - No new privileges escalation
    - Network isolation by default
    - Resource limits enforced

EXAMPLES:
    # Run with default settings (GPU 0, network isolated)
    $0

    # Run with all GPUs and network access
    $0 --gpu all --network bridge

    # Run on remote Docker host
    $0 --remote ssh://user@gpu-server --gpu 2

    # Run with specific memory limits
    $0 --memory 64g --shm-size 32g

EOF
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not found. Please install Docker first.${NC}"
        exit 1
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        echo -e "${RED}Error: Docker daemon is not running.${NC}"
        exit 1
    fi
}

# Check GPU availability
check_gpu() {
    local gpu_test_cmd="docker"
    if [[ -n "$DOCKER_HOST" ]]; then
        gpu_test_cmd="docker -H $DOCKER_HOST"
    fi

    echo -e "${BLUE}Checking GPU availability...${NC}"

    if ! $gpu_test_cmd run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: GPU not available or NVIDIA Container Toolkit not installed.${NC}"
        echo -e "Please ensure:"
        echo -e "  1. NVIDIA drivers are installed"
        echo -e "  2. NVIDIA Container Toolkit is installed"
        echo -e "  3. Docker daemon is configured for GPU support"
        exit 1
    fi

    echo -e "${GREEN}✓ GPU check passed${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        -w|--network)
            NETWORK_MODE="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY_LIMIT="$2"
            shift 2
            ;;
        -s|--shm-size)
            SHM_SIZE="$2"
            shift 2
            ;;
        -d|--workspace)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        -r|--remote)
            REMOTE_MODE=true
            DOCKER_HOST="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate workspace directory
if [[ ! -d "$WORKSPACE_DIR" ]]; then
    echo -e "${RED}Error: Workspace directory does not exist: $WORKSPACE_DIR${NC}"
    exit 1
fi

# Check prerequisites
check_docker
check_gpu

# Build docker run command
DOCKER_CMD="docker"
if [[ -n "$DOCKER_HOST" ]]; then
    DOCKER_CMD="docker -H $DOCKER_HOST"
    echo -e "${BLUE}Using remote Docker host: $DOCKER_HOST${NC}"
fi

DOCKER_CMD+=" run -it --rm"
DOCKER_CMD+=" --name $CONTAINER_NAME"

# GPU configuration
if [[ "$GPU_DEVICE" == "all" ]]; then
    DOCKER_CMD+=" --gpus all"
    echo -e "${GREEN}✓ All GPUs enabled${NC}"
else
    DOCKER_CMD+=" --gpus device=$GPU_DEVICE"
    echo -e "${GREEN}✓ GPU $GPU_DEVICE enabled${NC}"
fi

# Security settings
DOCKER_CMD+=" --cap-drop ALL"
DOCKER_CMD+=" --cap-add SYS_PTRACE"  # For debugging
DOCKER_CMD+=" --security-opt no-new-privileges"
DOCKER_CMD+=" --security-opt seccomp=unconfined"  # Required for PyTorch

# Resource limits
DOCKER_CMD+=" --memory $MEMORY_LIMIT"
DOCKER_CMD+=" --memory-swap $MEMORY_LIMIT"
DOCKER_CMD+=" --shm-size $SHM_SIZE"
DOCKER_CMD+=" --ulimit nofile=64000"
DOCKER_CMD+=" --ulimit nproc=640000"

# Network configuration
case "$NETWORK_MODE" in
    none)
        DOCKER_CMD+=" --network none"
        echo -e "${GREEN}✓ Network isolated${NC}"
        ;;
    bridge)
        DOCKER_CMD+=" --network bridge"
        echo -e "${YELLOW}! Network enabled (bridge mode)${NC}"
        ;;
    host)
        DOCKER_CMD+=" --network host"
        echo -e "${YELLOW}! Network enabled (host mode)${NC}"
        ;;
    *)
        echo -e "${RED}Error: Invalid network mode: $NETWORK_MODE${NC}"
        exit 1
        ;;
esac

# Read-only root filesystem
DOCKER_CMD+=" --read-only"

# Temporary filesystems
DOCKER_CMD+=" --tmpfs /tmp:rw,noexec,nosuid,size=4g"
DOCKER_CMD+=" --tmpfs /run:rw,noexec,nosuid,size=1g"
DOCKER_CMD+=" --tmpfs /root/.cache:rw,noexec,nosuid,size=4g"
DOCKER_CMD+=" --tmpfs /var/tmp:rw,noexec,nosuid,size=2g"

# Create sandbox directories
SANDBOX_BASE="$HOME/.metta-gpu-sandbox/$CONTAINER_NAME"
mkdir -p "$SANDBOX_BASE"/{data,cache,wandb,output,logs}

# Volume mounts
DOCKER_CMD+=" -v $WORKSPACE_DIR:/workspace/metta:ro"  # Read-only workspace
DOCKER_CMD+=" -v $SANDBOX_BASE/data:/workspace/metta/train_dir:rw"
DOCKER_CMD+=" -v $SANDBOX_BASE/cache:/var/cache/metta:rw"
DOCKER_CMD+=" -v $SANDBOX_BASE/wandb:/workspace/metta/wandb:rw"
DOCKER_CMD+=" -v $SANDBOX_BASE/output:/output:rw"
DOCKER_CMD+=" -v $SANDBOX_BASE/logs:/var/log/metta:rw"

# Environment variables
DOCKER_CMD+=" -e METTA_HOST=gpu-sandbox"
DOCKER_CMD+=" -e METTA_USER=sandbox"
DOCKER_CMD+=" -e UV_LINK_MODE=copy"
DOCKER_CMD+=" -e PYTHONDONTWRITEBYTECODE=1"
DOCKER_CMD+=" -e PYTHONUNBUFFERED=1"
DOCKER_CMD+=" -e WANDB_MODE=offline"
DOCKER_CMD+=" -e WANDB_DISABLED=true"
DOCKER_CMD+=" -e CUDA_VISIBLE_DEVICES=$GPU_DEVICE"

# GPU-specific environment
DOCKER_CMD+=" -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
DOCKER_CMD+=" -e NCCL_DEBUG=WARN"

# Add the image and command
DOCKER_CMD+=" $IMAGE"
DOCKER_CMD+=" bash -c 'cd /workspace/metta && source .venv/bin/activate && exec bash'"

# Print configuration summary
echo -e "\n${GREEN}GPU Sandbox Configuration:${NC}"
echo "  Container: $CONTAINER_NAME"
echo "  Image: $IMAGE"
echo "  GPU Device(s): $GPU_DEVICE"
echo "  Memory: $MEMORY_LIMIT (SHM: $SHM_SIZE)"
echo "  Network: $NETWORK_MODE"
echo "  Workspace: $WORKSPACE_DIR (read-only)"
echo "  Data directory: $SANDBOX_BASE"
if [[ "$REMOTE_MODE" == true ]]; then
    echo "  Remote host: $DOCKER_HOST"
fi
echo ""

# Write sandbox info file
cat > "$SANDBOX_BASE/sandbox_info.txt" << EOF
Sandbox Name: $CONTAINER_NAME
Created: $(date)
Image: $IMAGE
GPU Device(s): $GPU_DEVICE
Memory Limit: $MEMORY_LIMIT
Shared Memory: $SHM_SIZE
Network Mode: $NETWORK_MODE
Workspace: $WORKSPACE_DIR
Remote Mode: $REMOTE_MODE
Docker Host: $DOCKER_HOST
EOF

# Run the container
echo -e "${GREEN}Starting GPU sandbox container...${NC}"
echo -e "${YELLOW}Press Ctrl+D or type 'exit' to stop the container${NC}\n"

# Execute the Docker command
eval $DOCKER_CMD
EXIT_CODE=$?

# Post-run cleanup and summary
echo -e "\n${GREEN}GPU sandbox container stopped.${NC}"
echo -e "Exit code: $EXIT_CODE"
echo -e "Session data preserved in: ${BLUE}$SANDBOX_BASE${NC}"
echo -e "To view logs: ${YELLOW}ls -la $SANDBOX_BASE/logs/${NC}"
echo -e "To clean up: ${YELLOW}rm -rf $SANDBOX_BASE${NC}"

# Generate summary report
if [[ -d "$SANDBOX_BASE/output" ]]; then
    OUTPUT_FILES=$(find "$SANDBOX_BASE/output" -type f | wc -l)
    if [[ $OUTPUT_FILES -gt 0 ]]; then
        echo -e "\nOutput files generated: ${GREEN}$OUTPUT_FILES${NC}"
        echo -e "View outputs: ${YELLOW}ls -la $SANDBOX_BASE/output/${NC}"
    fi
fi

exit $EXIT_CODE
