#!/bin/bash
# AMI Setup Script for Researcher Sandbox
#
# This script installs all necessary software on a base Ubuntu 22.04 instance
# to create the researcher sandbox AMI.
#
# Installed software:
# - Docker + NVIDIA Container Toolkit
# - NVIDIA GPU drivers (535)
# - Python 3.12 + uv
# - Puffer and cogames repositories (from GitHub)
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
echo ""

# Ensure running as ubuntu user
if [ "$USER" != "ubuntu" ]; then
    echo "Error: This script must be run as 'ubuntu' user"
    exit 1
fi

# Update system
echo "[1/7] Updating system packages..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Install Docker
echo "[2/7] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sudo sh /tmp/get-docker.sh
    sudo usermod -aG docker ubuntu
    rm /tmp/get-docker.sh
    echo "Docker installed successfully"
else
    echo "Docker already installed"
fi

# Install NVIDIA drivers
echo "[3/7] Installing NVIDIA GPU drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
    echo "NVIDIA drivers installed (will be active after reboot)"
else
    echo "NVIDIA drivers already installed"
    nvidia-smi
fi

# Install NVIDIA Container Toolkit
echo "[4/7] Installing NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "NVIDIA Container Toolkit installed"
else
    echo "NVIDIA Container Toolkit already installed"
fi

# Install Python and uv
echo "[5/7] Installing Python 3.12 and uv..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3-pip git
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv installed successfully"
else
    echo "uv already installed"
fi

# Install cogames and mettagrid packages
echo "[6/7] Installing cogames and mettagrid..."
cd /home/ubuntu

# Install cogames CLI and mettagrid environment
if ! ~/.cargo/bin/uv tool list | grep -q cogames; then
    echo "Installing cogames CLI..."
    ~/.cargo/bin/uv tool install cogames
else
    echo "cogames already installed"
fi

# Install mettagrid Python package for training
if ! python3.12 -c "import mettagrid" 2>/dev/null; then
    echo "Installing mettagrid package..."
    ~/.cargo/bin/uv pip install mettagrid
else
    echo "mettagrid already installed"
fi

# Create welcome message
echo "[7/7] Creating welcome message..."
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
  - Python 3.12 + uv package manager
  - cogames CLI and mettagrid package

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
