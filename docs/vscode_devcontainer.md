# VSCode Dev Container Setup for Metta

This guide explains how to use VSCode Dev Containers for Metta development, providing isolated, reproducible development environments with optional GPU support.

## Overview

The Metta project provides four dev container configurations:

1. **GPU Development (`gpu-dev`)** - Full GPU-enabled environment for training and testing
2. **CPU Development (`cpu-dev`)** - Lightweight environment for code development without GPU
3. **Remote GPU (`remote-gpu`)** - Optimized for cloud GPU instances and remote development
4. **Sandbox GPU (`sandbox-gpu`)** - Secure GPU environment for untrusted code execution

## Prerequisites

### Local Development
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine
- [VSCode](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- (For GPU) NVIDIA GPU with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Remote Development
- SSH access to a GPU-enabled machine
- Docker and NVIDIA Container Toolkit installed on the remote machine

## Quick Start

### Option 1: Using Default Configuration

1. Open the Metta project in VSCode
2. Press `Cmd/Ctrl+Shift+P` and select "Dev Containers: Reopen in Container"
3. VSCode will build and start the container

### Option 2: Selecting a Specific Configuration

1. Open Command Palette (`Cmd/Ctrl+Shift+P`)
2. Select "Dev Containers: Open Folder in Container..."
3. Choose "From 'devcontainer.json' in subfolder"
4. Select one of:
   - `.devcontainer/gpu-dev` - For local GPU development
   - `.devcontainer/cpu-dev` - For CPU-only development
   - `.devcontainer/remote-gpu` - For remote GPU instances
   - `.devcontainer/sandbox-gpu` - For secure sandboxed GPU execution

## Configuration Details

### GPU Development Container

**Location**: `.devcontainer/gpu-dev/devcontainer.json`

**Features**:
- Full CUDA support with GPU passthrough
- X11 forwarding for GUI applications
- Large shared memory (80GB) for training
- Pre-configured Python and C++ development tools
- Automatic dependency installation with `uv`

**Use Cases**:
- Training neural networks
- Running GPU-accelerated simulations
- Developing CUDA kernels
- Testing GPU-specific features

### CPU Development Container

**Location**: `.devcontainer/cpu-dev/devcontainer.json`

**Features**:
- No GPU requirements
- Reduced resource usage (8GB shared memory)
- Skips GPU-related tests automatically
- Same development tools as GPU version

**Use Cases**:
- Code development and refactoring
- Running CPU-only tests
- Documentation and configuration work
- Development on machines without NVIDIA GPUs

### Remote GPU Container

**Location**: `.devcontainer/remote-gpu/devcontainer.json`

**Features**:
- Optimized for cloud GPU instances (AWS, GCP, Azure)
- Large shared memory allocation (120GB)
- SSH key mounting for GitHub access
- Cloud provider environment variables
- NCCL optimizations for multi-GPU setups

**Use Cases**:
- Training on cloud GPU instances
- Distributed training experiments
- Remote development via SSH
- CI/CD pipeline testing

### Sandbox GPU Container

**Location**: `.devcontainer/sandbox-gpu/devcontainer.json`

**Features**:
- Secure environment for untrusted code execution
- Limited network access (bridge mode for package installation only)
- Isolated data directories (~/metta-sandbox/*)
- Reduced memory limits (32GB RAM, 16GB shared memory)
- Minimal VSCode extensions for security
- PyTorch GPU optimizations for sandbox environments
- CUDA kernel compilation cache

**Use Cases**:
- Testing experimental or untrusted code
- Running code from external contributors
- Isolated GPU experiments
- Security-sensitive development

**Security Features**:
- Dropped Linux capabilities (except SYS_PTRACE and IPC_LOCK)
- No new privileges flag
- Isolated data directories
- Offline mode for Weights & Biases by default
- Limited port forwarding (API and TensorBoard only)

**Setup**:
Before using the sandbox container, create the necessary directories:
```bash
mkdir -p ~/metta-sandbox/{train_dir,checkpoints,.cache,wandb,.cuda_cache}
```

## Environment Variables

The following environment variables are automatically configured:

- `METTA_HOST` - Hostname of the container
- `METTA_USER` - Your username
- `WANDB_API_KEY` - Weights & Biases API key (from host)
- `CUDA_VISIBLE_DEVICES` - GPU device selection
- `UV_LINK_MODE=copy` - UV package manager optimization

### Sandbox-Specific Variables
- `WANDB_MODE=offline` - Disables cloud sync by default
- `WANDB_DISABLED=true` - Fully disables W&B by default
- `PYTORCH_CUDA_ALLOC_CONF` - Memory allocation optimization

## Volume Mounts

### Persistent Data
- `~/metta-data/train_dir` → `/workspace/metta/train_dir`
- `~/metta-data/checkpoints` → `/workspace/metta/checkpoints`

### Sandbox Data (Isolated)
- `~/metta-sandbox/train_dir` → `/workspace/metta/train_dir`
- `~/metta-sandbox/checkpoints` → `/workspace/metta/checkpoints`
- `~/metta-sandbox/.cuda_cache` → `/workspace/metta/.cuda_cache`

### Cache Directories
- `~/.cache/uv` → `/root/.cache/uv`
- `~/.cache/pip` → `/root/.cache/pip`

### Display (GPU containers only)
- `/tmp/.X11-unix` - X11 socket
- `/mnt/wslg` - WSL2 GUI support

## Port Forwarding

The following ports are automatically forwarded:

- `8000` - Backend API server
- `3000` - Gridworks frontend
- `6006` - TensorBoard
- `8888` - Jupyter notebooks
- `8265` - Ray dashboard (remote-gpu only)

## VSCode Extensions

The dev containers automatically install:

### Python Development
- Python
- Pylance
- Python Debugger
- Ruff formatter
- Black formatter

### C++ Development
- C/C++ tools
- CMake Tools
- CMake language support

### Other Tools
- YAML support
- Docker support
- GitLens
- GitHub Pull Requests
- Error Lens
- Spell Checker

## Tips and Tricks

### 1. GPU Memory Issues

If you encounter GPU memory errors:
```bash
# Check GPU usage
nvidia-smi

# Set specific GPU
export CUDA_VISIBLE_DEVICES=0

# Clear GPU memory
nvidia-smi --gpu-reset
```

### 2. Switching Between Configurations

To switch configurations without rebuilding:
1. Open Command Palette
2. Select "Dev Containers: Rebuild Container"
3. Choose a different configuration

### 3. Persistent Settings

Add personal settings to:
- `.devcontainer/settings.json` - Shared settings
- `~/.config/Code/User/settings.json` - Personal settings

### 4. Running Tests

```bash
# GPU tests
pytest tests/

# CPU-only tests
pytest tests/ -m "not gpu"

# Specific test file
pytest tests/test_simulation.py -v
```

### 5. Building C++ Components

```bash
cd mettagrid
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 6. Using with Remote SSH

1. Install "Remote - SSH" extension
2. Connect to remote machine
3. Open Metta folder
4. Reopen in container using remote-gpu configuration

### 7. Using the Sandbox Container

The sandbox container provides a secure environment:

```bash
# Initialize sandbox directories (one-time setup)
mkdir -p ~/metta-sandbox/{train_dir,checkpoints,.cache,wandb,.cuda_cache}

# Open in sandbox
# Select .devcontainer/sandbox-gpu when prompted

# Enable network access for W&B (if needed)
export WANDB_MODE=online
export WANDB_DISABLED=false
```

## Troubleshooting

### Container Won't Start

1. Check Docker daemon is running
2. Verify NVIDIA drivers (for GPU containers):
   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

### Extensions Not Loading

1. Rebuild container: "Dev Containers: Rebuild Container"
2. Check extension compatibility with container architecture

### Permission Issues

The containers use root user by default. For the sandbox container, this is intentional for GPU access.

### Network Issues

If you can't access services:
1. Check port forwarding in VSCode
2. Ensure `network_mode: host` is not conflicting
3. Try explicit port mapping instead

### Sandbox-Specific Issues

1. **CUDA Compilation Errors**: Ensure `~/.metta-sandbox/.cuda_cache` exists
2. **Out of Memory**: Sandbox has lower memory limits (32GB)
3. **Network Access**: Sandbox uses bridge network, not host network

## Advanced Configuration

### Custom Docker Image

Modify `.devcontainer/devcontainer.json`:
```json
"build": {
    "dockerfile": "custom.Dockerfile",
    "args": {
        "CUSTOM_ARG": "value"
    }
}
```

### Additional Mounts

Add to `mounts` array:
```json
"mounts": [
    "source=/local/path,target=/container/path,type=bind"
]
```

### Environment-Specific Settings

Create `.devcontainer/.env`:
```bash
WANDB_PROJECT=my-project
CUSTOM_VAR=value
```

## Security Considerations

1. **SSH Keys**: The remote-gpu configuration mounts SSH keys read-only
2. **Docker Socket**: GPU configurations mount Docker socket for nested containers
3. **Capabilities**: `SYS_PTRACE` and `seccomp=unconfined` are enabled for debugging
4. **Sandbox Isolation**: The sandbox-gpu configuration uses separate data directories and restricted capabilities

## Updating Dev Containers

When the base image updates:
1. Pull latest changes: `git pull`
2. Rebuild container: "Dev Containers: Rebuild Container"
3. Run `uv sync --locked` to update dependencies

## Contributing

To modify dev container configurations:
1. Edit files in `.devcontainer/`
2. Test changes locally
3. Document any new features
4. Submit PR with description of changes
