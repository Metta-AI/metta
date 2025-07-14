#!/usr/bin/env bash

# Install dependencies
#!/usr/bin/env bash
export DEBIAN_FRONTEND=noninteractive

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "☠️  This installer must be run as root. Try: sudo $0" >&2
  exit 1
fi

# REMEMBER TO UPDATE DOCKER PULL FLAG WHEN UPDATING THIS SCRIPT
# -------------------------------------------------------------

################################################################################
# 0.  Baseline packages
################################################################################
apt-get update -y
apt-get install -y \
  linux-headers-$(uname -r) \
  build-essential \
  openssh-server \
  vim \
  dkms \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  rsyslog # for remote journald if you enable it later

################################################################################
# 1.  Tailscale
################################################################################
if ! command -v tailscale &> /dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

# Make tailscaled restart automatically if it crashes or loses connectivity
mkdir -p /etc/systemd/system/tailscaled.service.d
cat << 'EOF' | tee /etc/systemd/system/tailscaled.service.d/10-restart.conf
[Service]
Restart=always
RestartSec=5
EOF
systemctl daemon-reload
systemctl restart tailscaled

################################################################################
# 2.  Docker (engine + NVIDIA container runtime)
################################################################################
if ! command -v docker &> /dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  echo "deb [arch=$(dpkg --print-architecture) \
        signed-by=/etc/apt/keyrings/docker.asc] \
        https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null
fi

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

# NVIDIA container runtime (Debian bullseye repo still required)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
systemctl restart docker
