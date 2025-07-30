#!/bin/bash

echo "=== GPU DIAGNOSTICS AND RECOVERY ==="
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"

# Function to run a quick GPU health check
quick_gpu_check() {
    docker run --rm --gpus all metta:latest python -c "
import torch
import sys
if torch.cuda.is_available():
    print(f'✅ GPU Check Passed - PyTorch {torch.__version__}, CUDA {torch.version.cuda}, Device: {torch.cuda.get_device_name(0)}')
    sys.exit(0)
else:
    print('❌ GPU Check Failed - CUDA not available')
    sys.exit(1)
" 2>/dev/null
}

# FAST PATH: Try the end-to-end test first
echo -e "\n--- Quick GPU Health Check ---"
if quick_gpu_check; then
    echo "✅ GPU is fully functional! Skipping detailed diagnostics."
    exit 0
fi

# If we get here, something is wrong, so run full diagnostics
echo "⚠️  Quick check failed. Running detailed diagnostics..."

# Collect system information
echo -e "\n--- System Information ---"
echo "Instance type: $(ec2-metadata --instance-type 2>/dev/null || echo 'Unknown')"
echo "AMI ID: $(ec2-metadata --ami-id 2>/dev/null || echo 'Unknown')"
echo "Region: $(ec2-metadata --availability-zone 2>/dev/null || echo 'Unknown')"
echo "Kernel: $(uname -r)"

# Check if GPU is visible to the system
echo -e "\n--- PCI Devices ---"
GPU_PCI=$(lspci | grep -i nvidia)
if [ -n "$GPU_PCI" ]; then
    echo "$GPU_PCI"
    GPU_VISIBLE=true
else
    echo "⚠️  No NVIDIA devices found in lspci"
    GPU_VISIBLE=false
    # No point continuing if GPU isn't visible at hardware level
    echo -e "\n❌ GPU not visible at hardware level. This requires a different instance or AMI."
    echo "Recommendation: Try a different region or verify the instance type supports GPUs."
    exit 1
fi

# Check if basic nvidia-smi works (faster than docker test)
echo -e "\n--- Host NVIDIA-SMI Test ---"
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ Host nvidia-smi works"

    # If host nvidia-smi works, just test docker
    echo -e "\n--- Docker GPU Test ---"
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "✅ Docker GPU access works"

        # Docker works, so the issue must be with the metta container
        echo -e "\n--- Retrying Python CUDA Check ---"
        if quick_gpu_check; then
            echo "✅ GPU is now functional after retry!"
            exit 0
        else
            echo "⚠️  metta container CUDA issue - checking PyTorch installation..."
            docker run --rm --gpus all metta:latest python -c "
import subprocess
import sys
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'PyTorch CUDA built: {torch.cuda.is_built()}')
    print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
    # Check if this is a CPU-only PyTorch build
    if not torch.cuda.is_built():
        print('❌ PyTorch is CPU-only build!')
except Exception as e:
    print(f'Error importing torch: {e}')
"
            echo "This appears to be a PyTorch/CUDA compatibility issue in the metta container."
            exit 1
        fi
    else
        echo "⚠️  Docker GPU test failed - checking Docker configuration..."
    fi
else
    echo "⚠️  Host nvidia-smi not working - will attempt driver recovery"
fi

# If we get here, we need to do deeper diagnostics and recovery
echo -e "\n--- Detailed Diagnostics ---"

# Check NVIDIA kernel modules
NVIDIA_MODULES=$(lsmod | grep nvidia)
if [ -n "$NVIDIA_MODULES" ]; then
    echo "Kernel modules loaded:"
    echo "$NVIDIA_MODULES"
    MODULES_LOADED=true
else
    echo "⚠️  No NVIDIA kernel modules loaded"
    MODULES_LOADED=false
fi

# Check NVIDIA device files
if ls /dev/nvidia* 2>/dev/null > /dev/null; then
    echo "Device files present: $(ls /dev/nvidia* 2>/dev/null | wc -l) files"
    DEVICE_FILES=true
else
    echo "⚠️  No /dev/nvidia* files found"
    DEVICE_FILES=false
fi

# Check Docker GPU runtime configuration
DOCKER_GPU_RUNTIME=false
if [ -f /etc/docker/daemon.json ]; then
    if grep -qE "nvidia|gpu" /etc/docker/daemon.json 2>/dev/null; then
        echo "Docker GPU runtime is configured"
        DOCKER_GPU_RUNTIME=true
    else
        echo "⚠️  No GPU runtime in Docker config"
    fi
fi

# Check for NVIDIA driver packages
NVIDIA_PACKAGES=$(dpkg -l 2>/dev/null | grep -c nvidia || rpm -qa 2>/dev/null | grep -c nvidia)
if [ "$NVIDIA_PACKAGES" -gt 0 ]; then
    echo "NVIDIA packages installed: $NVIDIA_PACKAGES packages found"
else
    echo "⚠️  No NVIDIA packages found"
fi

# Check dmesg for errors
DMESG_ERRORS=$(sudo dmesg | grep -i "nvidia.*error" | tail -5)
if [ -n "$DMESG_ERRORS" ]; then
    echo -e "\nRecent NVIDIA errors in dmesg:"
    echo "$DMESG_ERRORS"
fi

echo -e "\n=== ATTEMPTING GPU RECOVERY ==="

# Detect OS
if command -v apt-get &> /dev/null; then
    OS="ubuntu"
elif command -v yum &> /dev/null; then
    OS="amazon-linux"
else
    echo "⚠️  Unsupported OS for automatic recovery"
    exit 1
fi

# Try simple fixes first
if [ "$MODULES_LOADED" = false ]; then
    echo "Loading NVIDIA kernel modules..."
    sudo modprobe nvidia 2>&1 || echo "Failed to load nvidia module"
    sudo modprobe nvidia_uvm 2>&1 || echo "Failed to load nvidia_uvm module"
    sudo modprobe nvidia_drm 2>&1 || echo "Failed to load nvidia_drm module"

    # Quick test after module loading
    if nvidia-smi &> /dev/null; then
        echo "✅ nvidia-smi working after module load"
        if quick_gpu_check; then
            echo "✅ GPU recovered after loading modules!"
            exit 0
        fi
    fi
fi

# If still not working, try driver installation
if ! nvidia-smi &> /dev/null; then
    echo "Installing/repairing NVIDIA drivers..."

    if [ "$OS" = "ubuntu" ]; then
        sudo apt-get update
        sudo apt-get install -y linux-headers-$(uname -r)

        # Try driver installation
        for driver_version in 535 525 470; do
            echo "Trying nvidia-driver-${driver_version}..."
            if sudo apt-get install -y nvidia-driver-${driver_version}; then
                echo "✅ Installed nvidia-driver-${driver_version}"
                break
            fi
        done
    elif [ "$OS" = "amazon-linux" ]; then
        sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)
        sudo yum install -y nvidia-driver-latest-dkms
    fi

    # Reload modules
    sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
    sudo modprobe nvidia nvidia_uvm nvidia_drm
fi

# Fix Docker runtime if needed
if [ "$DOCKER_GPU_RUNTIME" = false ] && command -v nvidia-ctk &> /dev/null; then
    echo "Configuring Docker GPU runtime..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    sleep 3
fi

# Create device files if missing
if [ "$DEVICE_FILES" = false ]; then
    echo "Creating NVIDIA device files..."
    sudo nvidia-modprobe -u -c=0 || true
fi

# Restart services
sudo systemctl restart nvidia-persistenced 2>/dev/null || true

# Final test
echo -e "\n--- Final GPU Test ---"
sleep 3

if quick_gpu_check; then
    echo "✅ GPU recovery successful!"
    exit 0
else
    echo "❌ GPU recovery failed. Manual intervention required."
    echo "Please check the diagnostics above or try a different AMI/region."
    exit 1
fi
