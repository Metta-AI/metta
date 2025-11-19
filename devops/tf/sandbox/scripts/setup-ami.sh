#!/bin/bash
# AMI Setup Script for Researcher Sandbox
#
# This script installs all necessary software on a base Ubuntu 22.04 instance
# to create the researcher sandbox AMI.
#
# Installed software:
# - Docker + NVIDIA Container Toolkit
# - NVIDIA GPU drivers (535)
# - uv (Python 3.12 environment manager)
# - cogames CLI and mettagrid package
#
# Usage:
#   1. Launch Ubuntu 22.04 GPU instance (g5.12xlarge)
#   2. SSH in: ssh ubuntu@<instance-ip>
#   3. Run this script: curl -fsSL <url-to-this-script> | bash
#   4. Create AMI from the instance

set -euo pipefail

echo "=========================================="
echo "Researcher Sandbox AMI Setup"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Ensure running as root (cloud-init runs as root)
if [ "$EUID" -ne 0 ]; then
  echo "Error: This script must be run as root"
  exit 1
fi

# Function to log with timestamps
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Function to handle errors
error_exit() {
  log "ERROR: $1"
  exit 1
}

# Update system and install build tools
log "[1/9] Updating system packages and installing build tools..."
sudo apt-get update || error_exit "apt-get update failed"
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y || error_exit "apt-get upgrade failed"
# Install build tools required for:
# - build-essential: C/C++ compiler for pufferlib-core and pybind11
# - python3-dev: Python development headers for building extensions
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python3-dev || error_exit "build tools installation failed"
log "System packages updated and build tools installed successfully"

# Install Docker
log "[2/9] Installing Docker..."
if ! command -v docker &> /dev/null; then
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh || error_exit "Failed to download Docker install script"
  sudo sh /tmp/get-docker.sh || error_exit "Docker installation failed"
  sudo usermod -aG docker ubuntu || error_exit "Failed to add ubuntu to docker group"
  rm /tmp/get-docker.sh
  log "Docker installed successfully"
else
  log "Docker already installed"
fi

# Install NVIDIA drivers
echo "[3/9] Installing NVIDIA GPU drivers..."
if ! command -v nvidia-smi &> /dev/null; then
  sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
  echo "NVIDIA drivers installed (will be active after reboot)"
else
  echo "NVIDIA drivers already installed"
  nvidia-smi
fi

# Install NVIDIA Container Toolkit
echo "[4/9] Installing NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  echo "NVIDIA Container Toolkit installed"
else
  echo "NVIDIA Container Toolkit already installed"
fi

# Install uv and git
log "[5/9] Installing uv and git..."
sudo apt-get install -y git
if ! su - ubuntu -c "command -v uv" &> /dev/null; then
  su - ubuntu -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  log "uv installed successfully"
else
  log "uv already installed"
fi

# Install Bazel (required for building mettagrid C++ code)
log "[6/9] Installing Bazel..."
if ! command -v bazel &> /dev/null; then
  # Detect architecture
  ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
    BAZEL_ARCH="amd64"
  elif [ "$ARCH" = "aarch64" ]; then
    BAZEL_ARCH="arm64"
  else
    error_exit "Unsupported architecture: $ARCH"
  fi

  # Install Bazelisk which automatically manages Bazel versions
  sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-${BAZEL_ARCH}
  sudo chmod +x /usr/local/bin/bazel
  log "Bazel installed successfully"
else
  log "Bazel already installed"
fi

# Install Nim (required for building mettascope in mettagrid)
log "[7/9] Installing Nim compiler..."
if ! command -v nim &> /dev/null; then
  # Install Nim from binary release (choosenim has reliability issues)
  NIM_VERSION="2.2.2"
  ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
    NIM_ARCH="x64"
  elif [ "$ARCH" = "aarch64" ]; then
    NIM_ARCH="arm64"
  else
    error_exit "Unsupported architecture for Nim: $ARCH"
  fi

  cd /tmp
  wget https://nim-lang.org/download/nim-${NIM_VERSION}-linux_${NIM_ARCH}.tar.xz
  tar xf nim-${NIM_VERSION}-linux_${NIM_ARCH}.tar.xz
  sudo mv nim-${NIM_VERSION} /opt/nim
  sudo ln -s /opt/nim/bin/nim /usr/local/bin/nim
  sudo ln -s /opt/nim/bin/nimble /usr/local/bin/nimble
  rm nim-${NIM_VERSION}-linux_${NIM_ARCH}.tar.xz
  cd - > /dev/null

  log "Nim installed successfully"
else
  log "Nim already installed"
fi

# Install cogames and mettagrid packages
log "[8/9] Installing cogames and mettagrid..."

# Create Python environment with uv (run as ubuntu user)
if [ ! -d "/home/ubuntu/sandbox-env" ]; then
  log "Creating Python 3.12 environment..."
  su - ubuntu -c 'cd ~ && /home/ubuntu/.local/bin/uv init -p 3.12 sandbox-env'

  # Install cogames CLI as a tool
  log "Installing cogames CLI..."
  su - ubuntu -c 'cd ~/sandbox-env && /home/ubuntu/.local/bin/uv tool install cogames'

  # Install mettagrid in the venv
  log "Installing mettagrid package..."
  su - ubuntu -c 'cd ~/sandbox-env && /home/ubuntu/.local/bin/uv add mettagrid'

  # Add venv activation to bashrc
  su - ubuntu -c 'cat >> ~/.bashrc << '"'"'BASHRC_EOF'"'"'

# Activate sandbox Python environment
source ~/sandbox-env/.venv/bin/activate
BASHRC_EOF'

  log "Python environment created and configured"
else
  log "sandbox-env already exists"
fi

# Create welcome message
log "[9/9] Creating welcome message..."
cat > /home/ubuntu/README.txt << 'EOF'
========================================
Softmax Research Sandbox
========================================

Welcome! This sandbox is provided as part of the ALife 2024 compute credits program.

Available Resources:
--------------------
- 4x NVIDIA A10G GPUs (96GB VRAM total)
  Check status: nvidia-smi

- Pre-installed software:
  - Docker + NVIDIA Container Toolkit
  - Python 3.12 environment (via uv)
  - cogames CLI and mettagrid package
  - Auto-activated Python environment on login

Getting Started:
----------------
1. Check GPU availability:
   nvidia-smi

2. Train your first agent:
   cogames train --help

3. Submit your trained agent:
   cogames submit <path_to_model>

4. Monitor your compute usage:
   cogames sandbox status

Tips:
-----
- Your instance will be stopped when inactive to save credits
- Data persists on EBS volume between stops/starts
- Check remaining credits: cogames sandbox status
- Stop your instance when done: cogames sandbox stop

Documentation:
--------------
- cogames docs: https://cogames.io/docs
- Questions? Email: research@softmax.com

Your credit limit and usage:
-----------------------------
Run: cogames sandbox status

Have fun training!
EOF

# Set proper permissions
sudo chown -R ubuntu:ubuntu /home/ubuntu/README.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Reboot instance for NVIDIA drivers: sudo reboot"
echo "2. After reboot, verify GPU: nvidia-smi"
echo "3. Test Docker + GPU: docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"
echo "4. If everything works, create AMI from this instance"
echo ""
