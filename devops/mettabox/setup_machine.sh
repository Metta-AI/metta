#!/bin/bash

# REMEMBER TO UPDATE DOCKER PULL FLAG WHEN UPDATING THIS SCRIPT

# Install essentials
apt-get install -y \
  linux-headers-$(uname -r) \
  build-essential \
  openssh-server \
  vim \
  dkms \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg

# Install Tailscale
if ! command -v tailscale &> /dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

echo "cd /home/metta/metta/devops/mettabox && bash docker.sh test" >> /home/metta/.bashrc

# Docker
if ! command -v docker &> /dev/null; then
  # Add Docker's official GPG key:
  sudo apt-get update
  sudo apt-get install ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  # Add the repository to Apt sources:
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
fi

# Update the package list to reflect new repositories
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# NVIDIA drivers - more specific installation
ubuntu-version=$(lsb_release -rs)
if nvidia-smi &> /dev/null; then
  echo "NVIDIA drivers already installed, skipping driver installation"
else
  # Get the recommended driver without installing
  recommended_driver=$(ubuntu-drivers devices | grep "recommended" | awk '{print $3}')
  if [ -n "$recommended_driver" ]; then
    apt-get install -y "$recommended_driver"
  else
    echo "No recommended NVIDIA driver found"
  fi
fi

# Nvidia container (have to use Debian 11 bullseye for now)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update && apt-get install -y nvidia-container-toolkit
systemctl restart docker

sudo usermod -aG docker metta
mkdir -p /home/metta/data_dir

# Completion message with instructions
echo -e "Installation complete.\nTo complete installation:\n\
1) Grab the Tailscale auth key from Spacelift: https://metta-ai.app.spacelift.io/stack/efs/outputs\n\
2) Initialize Tailscale:\n\
   - sudo tailscale up --auth-key=<auth-key> --advertise-tags=tag:mettabox\n\
3) Reboot the machine."
