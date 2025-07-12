#!/bin/bash
# Metta Sandbox Container Runner
# Provides secure GPU-enabled containers for testing and isolation

set -euo pipefail

# Default values
CONTAINER_NAME="metta-sandbox-$$"
IMAGE="mettaai/metta:latest"
GPU_ENABLED=true
NETWORK_ENABLED=false
MEMORY_LIMIT="32g"
SHM_SIZE="16g"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run a secure Metta sandbox container with GPU support.

OPTIONS:
    -n, --name NAME           Container name (default: metta-sandbox-<PID>)
    -i, --image IMAGE         Docker image (default: mettaai/metta:latest)
    -g, --no-gpu              Disable GPU support
    -w, --network             Enable network access (disabled by default)
    -m, --memory LIMIT        Memory limit (default: 32g)
    -s, --shm-size SIZE       Shared memory size (default: 16g)
    -d, --workspace DIR       Workspace directory (default: current directory)
    -h, --help                Show this help message

EXAMPLES:
    # Run with default settings (GPU enabled, network disabled)
    $0

    # Run without GPU
    $0 --no-gpu

    # Run with network access
    $0 --network

    # Run with custom memory limits
    $0 --memory 64g --shm-size 32g

EOF
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
        -g|--no-gpu)
            GPU_ENABLED=false
            shift
            ;;
        -w|--network)
            NETWORK_ENABLED=true
            shift
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

# Build docker run command
DOCKER_CMD="docker run -it --rm"
DOCKER_CMD+=" --name $CONTAINER_NAME"

# GPU support
if [[ "$GPU_ENABLED" == true ]]; then
    DOCKER_CMD+=" --gpus all"
    echo -e "${GREEN}✓ GPU support enabled${NC}"
else
    echo -e "${YELLOW}! GPU support disabled${NC}"
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

# Network settings
if [[ "$NETWORK_ENABLED" == true ]]; then
    echo -e "${YELLOW}! Network access enabled${NC}"
else
    DOCKER_CMD+=" --network none"
    echo -e "${GREEN}✓ Network isolated${NC}"
fi

# Read-only root filesystem
DOCKER_CMD+=" --read-only"

# Temporary filesystems
DOCKER_CMD+=" --tmpfs /tmp:rw,noexec,nosuid,size=4g"
DOCKER_CMD+=" --tmpfs /run:rw,noexec,nosuid,size=1g"
DOCKER_CMD+=" --tmpfs /root/.cache:rw,noexec,nosuid,size=2g"

# Create sandbox directories if they don't exist
SANDBOX_BASE="$HOME/.metta-sandbox/$CONTAINER_NAME"
mkdir -p "$SANDBOX_BASE"/{data,cache,wandb,output}

# Volume mounts
DOCKER_CMD+=" -v $WORKSPACE_DIR:/workspace/metta:ro"  # Read-only workspace
DOCKER_CMD+=" -v $SANDBOX_BASE/data:/workspace/metta/train_dir:rw"
DOCKER_CMD+=" -v $SANDBOX_BASE/cache:/var/cache/metta:rw"
DOCKER_CMD+=" -v $SANDBOX_BASE/wandb:/workspace/metta/wandb:rw"
DOCKER_CMD+=" -v $SANDBOX_BASE/output:/output:rw"

# Environment variables
DOCKER_CMD+=" -e METTA_HOST=sandbox"
DOCKER_CMD+=" -e METTA_USER=sandbox"
DOCKER_CMD+=" -e UV_LINK_MODE=copy"
DOCKER_CMD+=" -e PYTHONDONTWRITEBYTECODE=1"
DOCKER_CMD+=" -e PYTHONUNBUFFERED=1"
DOCKER_CMD+=" -e WANDB_MODE=offline"
DOCKER_CMD+=" -e WANDB_DISABLED=true"

if [[ "$GPU_ENABLED" == true ]]; then
    DOCKER_CMD+=" -e CUDA_VISIBLE_DEVICES=all"
else
    DOCKER_CMD+=" -e CUDA_VISIBLE_DEVICES=''"
fi

# Add the image and command
DOCKER_CMD+=" $IMAGE"
DOCKER_CMD+=" bash -c 'cd /workspace/metta && exec bash'"

# Print configuration summary
echo -e "\n${GREEN}Sandbox Configuration:${NC}"
echo "  Container: $CONTAINER_NAME"
echo "  Image: $IMAGE"
echo "  Memory: $MEMORY_LIMIT (SHM: $SHM_SIZE)"
echo "  Workspace: $WORKSPACE_DIR (read-only)"
echo "  Data directory: $SANDBOX_BASE"
echo ""

# Run the container
echo -e "${GREEN}Starting sandbox container...${NC}\n"
eval $DOCKER_CMD

# Cleanup message
echo -e "\n${GREEN}Sandbox container stopped.${NC}"
echo -e "Temporary data preserved in: $SANDBOX_BASE"
echo -e "To clean up: ${YELLOW}rm -rf $SANDBOX_BASE${NC}"
