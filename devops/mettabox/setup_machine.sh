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
#systemctl restart tailscaled

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

################################################################################
# 3.  NVIDIA driver (pin to a known-good version) + GPU persistence
################################################################################
DRIVER_PACKAGE="nvidia-driver-570"
if nvidia-smi &> /dev/null; then
  echo "NVIDIA driver already installed, skipping."
else
  apt-get install -y "$DRIVER_PACKAGE"
fi
# Hold the package so unattended-upgrades won’t jump to 550+
apt-mark hold "$DRIVER_PACKAGE"

# Turn on persistence mode so the GPU never enters low-power states
nvidia-smi -pm 1 || true # ignore error on first boot (no driver yet)

# Mask systemd helpers that can race with GPUs on resume
systemctl mask nvidia-hibernate nvidia-suspend nvidia-resume

################################################################################
# 4.  Disable every form of sleep / suspend (systemd + logind)
################################################################################
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Ignore power button / lid close ACPI events
sed -i 's/^#\?HandlePowerKey=.*/HandlePowerKey=ignore/' /etc/systemd/logind.conf
sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
sed -i 's/^#\?HandleLidSwitchDocked=.*/HandleLidSwitchDocked=ignore/' /etc/systemd/logind.conf
systemctl restart systemd-logind

################################################################################
# 5.  Kernel crash dumps (optional but recommended)
################################################################################
echo "kdump-tools kdump-tools/use_kdump boolean true" \
  | debconf-set-selections
apt-get install -y linux-crashdump # installs kdump-tools
sed -i 's/^USE_KDUMP=.*/USE_KDUMP=1/' /etc/default/kdump-tools
systemctl enable kdump-tools

################################################################################
# 6.  Local tweaks
################################################################################
usermod -aG docker metta
mkdir -p /home/metta/data_dir
echo "cd /home/metta/metta/devops/mettabox && bash docker.sh test" >> /home/metta/.bashrc

################################################################################
# 7.  Reboot hint
################################################################################
echo -e "\n\033[32mInstallation complete.\033[0m\n\
Next steps:\n\
  1) Get your Tailscale auth-key (Spacelift → efs → outputs).\n\
  2) Activate Tailscale:  sudo tailscale up --auth-key=<key> --advertise-tags=tag:mettabox\n\
  3) Reboot:              sudo reboot\n"
