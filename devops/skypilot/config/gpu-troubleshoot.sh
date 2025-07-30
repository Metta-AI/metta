#!/bin/bash

echo "=== GPU DIAGNOSTICS ==="
echo "Timestamp: $(date)"
echo "Container Hostname: $(hostname)"

# Function to run a quick GPU health check
quick_gpu_check() {
    python3 -c "
import torch
import sys
if torch.cuda.is_available():
    print(f'✅ GPU Check Passed - PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    sys.exit(0)
else:
    print('❌ GPU Check Failed - CUDA not available')
    sys.exit(1)
" 2>&1
}

# FAST PATH: Try the end-to-end test first
echo -e "\n--- Quick GPU Health Check ---"
if quick_gpu_check; then
    echo "✅ GPU is fully functional!"
    exit 0
fi

# If we get here, something is wrong, so run diagnostics
echo "⚠️  Quick check failed. Running diagnostics..."

# Check environment variables
echo -e "\n--- GPU Environment Variables ---"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "NVIDIA_VISIBLE_DEVICES: ${NVIDIA_VISIBLE_DEVICES:-<not set>}"
echo "NVIDIA_DRIVER_CAPABILITIES: ${NVIDIA_DRIVER_CAPABILITIES:-<not set>}"

# Check if nvidia-smi is available (should be if nvidia runtime is configured)
echo -e "\n--- NVIDIA-SMI Test ---"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "✅ nvidia-smi command works"
        # Show GPU info
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    else
        echo "❌ nvidia-smi command exists but failed to run"
        echo "Exit code: $?"
    fi
else
    echo "❌ nvidia-smi command not found"
    echo "This suggests the container wasn't started with GPU support (--gpus flag)"
fi

# Check CUDA libraries
echo -e "\n--- CUDA Libraries Check ---"
if [ -d "/usr/local/cuda" ]; then
    echo "✅ CUDA directory found at /usr/local/cuda"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        echo "CUDA Version: $(cat /usr/local/cuda/version.txt)"
    elif [ -f "/usr/local/cuda/version.json" ]; then
        echo "CUDA Version: $(cat /usr/local/cuda/version.json | grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)"
    fi
else
    echo "⚠️  /usr/local/cuda not found"
fi

# Check for NVIDIA libraries in standard locations
echo -e "\n--- NVIDIA Library Check ---"
FOUND_LIBS=0
for lib in libcuda.so libnvidia-ml.so libcudart.so; do
    if find /usr/lib* /usr/local/lib* -name "$lib*" 2>/dev/null | head -1 | grep -q .; then
        echo "✅ Found $lib"
        ((FOUND_LIBS++))
    else
        echo "❌ Missing $lib"
    fi
done

if [ $FOUND_LIBS -eq 0 ]; then
    echo "⚠️  No NVIDIA libraries found - container may not have GPU support"
fi

# Check device files (these should be mounted by nvidia-container-runtime)
echo -e "\n--- Device Files Check ---"
if ls /dev/nvidia* 2>/dev/null | head -5 | grep -q .; then
    DEVICE_COUNT=$(ls /dev/nvidia* 2>/dev/null | wc -l)
    echo "✅ Found $DEVICE_COUNT NVIDIA device files"
else
    echo "❌ No /dev/nvidia* files found"
    echo "This indicates the container wasn't started with proper GPU support"
fi

# Python-specific diagnostics
echo -e "\n--- Python Environment Diagnostics ---"
python3 -c "
import sys
import os
import subprocess

print(f'Python: {sys.version.split()[0]}')
print(f'Python executable: {sys.executable}')

# Check if torch is installed
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'PyTorch CUDA built: {torch.cuda.is_built()}')
    print(f'PyTorch CUDA available: {torch.cuda.is_available()}')

    if torch.cuda.is_built():
        print(f'PyTorch CUDA version: {torch.version.cuda}')
        # Check if this matches system CUDA
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                import re
                match = re.search(r'release (\d+\.\d+)', result.stdout)
                if match:
                    system_cuda = match.group(1)
                    torch_cuda = '.'.join(torch.version.cuda.split('.')[:2])
                    if system_cuda == torch_cuda:
                        print(f'✅ PyTorch CUDA ({torch_cuda}) matches system CUDA ({system_cuda})')
                    else:
                        print(f'⚠️  PyTorch CUDA ({torch_cuda}) differs from system CUDA ({system_cuda})')
        except:
            pass
    else:
        print('❌ PyTorch is CPU-only build!')
        print('You need to install GPU-enabled PyTorch:')
        print('  pip install torch --index-url https://download.pytorch.org/whl/cu118')

except ImportError:
    print('❌ PyTorch not installed')
except Exception as e:
    print(f'Error checking PyTorch: {e}')

# Check LD_LIBRARY_PATH
print(f'\\nLD_LIBRARY_PATH: {os.environ.get(\"LD_LIBRARY_PATH\", \"<not set>\")}')
"

# Attempt recovery for common issues
echo -e "\n=== ATTEMPTING RECOVERY ==="

# 1. Check if this is a PyTorch issue
if python3 -c "import torch; exit(0 if torch.cuda.is_built() else 1)" 2>/dev/null; then
    echo "PyTorch has CUDA support built-in"
else
    echo "⚠️  PyTorch is CPU-only. To fix inside container:"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
    echo "  or for CUDA 12.1:"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
fi

# 2. Check LD_LIBRARY_PATH
if [ -z "$LD_LIBRARY_PATH" ] || [[ ! "$LD_LIBRARY_PATH" =~ cuda ]]; then
    echo -e "\n⚠️  LD_LIBRARY_PATH may need CUDA paths. Try:"
    echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
fi

# 3. Final recommendations
echo -e "\n--- Recommendations ---"
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "❌ Container doesn't have GPU access. Ensure:"
    echo "  1. The host has working GPU drivers (test nvidia-smi on host)"
    echo "  2. Docker/container runtime has nvidia-container-toolkit installed"
    echo "  3. Container was started with GPU support:"
    echo "     - Docker: docker run --gpus all ..."
    echo "     - Kubernetes: proper GPU resource requests"
    echo "     - SkyPilot: accelerators configured in task YAML"
elif ! quick_gpu_check; then
    echo "⚠️  GPU is accessible but PyTorch can't use it. Check:"
    echo "  1. PyTorch has GPU support (not CPU-only build)"
    echo "  2. PyTorch CUDA version matches system CUDA version"
    echo "  3. All required libraries are accessible"
fi

# Final test with more details
echo -e "\n--- Final Test ---"
python3 -c "
import torch
import sys

print('Checking CUDA...')
if not torch.cuda.is_available():
    # Try to get more specific error
    try:
        torch.cuda.init()
    except Exception as e:
        print(f'CUDA initialization error: {e}')

    # Check if this is a driver/library issue
    try:
        import ctypes
        ctypes.CDLL('libcuda.so.1')
        print('libcuda.so.1 is loadable')
    except Exception as e:
        print(f'Cannot load libcuda.so.1: {e}')
else:
    print('✅ CUDA is available and working!')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

exit 0
