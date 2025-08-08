#!/usr/bin/env python
"""
NCCL Diagnostics and Testing Module

This module provides comprehensive GPU/CUDA/NCCL diagnostics and testing capabilities
for distributed PyTorch training environments.
"""

import logging
import os
import subprocess
import sys
from typing import Any

import torch
import torch.distributed as dist

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], check: bool = False) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr
    except Exception as e:
        return -1, "", str(e)


def get_gpu_diagnostics() -> dict[str, Any]:
    """Collect comprehensive GPU diagnostics."""
    diagnostics = {
        "nvidia_smi": None,
        "cuda_version": None,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "nccl_version": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "gpu_count": 0,
        "pytorch_cuda_available": torch.cuda.is_available(),
        "pytorch_cuda_version": (getattr(getattr(torch, "version", None), "cuda", None)),
        "errors": [],
    }

    # Get nvidia-smi output
    logger.info("Running nvidia-smi...")
    code, stdout, stderr = run_command(["nvidia-smi"])
    if code == 0:
        diagnostics["nvidia_smi"] = stdout
        # Try to get GPU count
        try:
            diagnostics["gpu_count"] = torch.cuda.device_count()
        except Exception as e:
            diagnostics["errors"].append(f"Failed to get GPU count: {e}")
    else:
        diagnostics["errors"].append(f"nvidia-smi failed: {stderr}")

    # Get CUDA version from nvcc
    logger.info("Checking CUDA version...")
    code, stdout, stderr = run_command(["nvcc", "--version"])
    if code == 0:
        # Extract version from output
        for line in stdout.split("\n"):
            if "release" in line:
                diagnostics["cuda_version"] = line.strip()
                break
    else:
        diagnostics["cuda_version"] = "nvcc not found"

    # Get NCCL version
    logger.info("Checking NCCL version...")
    nccl_paths = ["/usr/local/cuda/lib64/libnccl.so", "/usr/lib/x86_64-linux-gnu/libnccl.so", "/usr/lib64/libnccl.so"]

    for nccl_path in nccl_paths:
        if os.path.exists(nccl_path):
            code, stdout, stderr = run_command(["strings", nccl_path])
            if code == 0:
                for line in stdout.split("\n"):
                    if line.startswith("NCCL"):
                        diagnostics["nccl_version"] = line.strip()
                        break
                if diagnostics["nccl_version"]:
                    break

    if not diagnostics["nccl_version"]:
        diagnostics["nccl_version"] = "NCCL not found"

    return diagnostics


def print_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Pretty print diagnostics information."""
    print("=== GPU Diagnostics ===")

    if diagnostics["nvidia_smi"]:
        print(diagnostics["nvidia_smi"])
    else:
        print("nvidia-smi: Not available")

    print(f"PyTorch Version: {diagnostics['torch_version']}")
    print(f"\nCUDA Version: {diagnostics['cuda_version']}")
    print(f"NCCL Version: {diagnostics['nccl_version']}")
    print(f"CUDA_VISIBLE_DEVICES: {diagnostics['cuda_visible_devices']}")
    print(f"PyTorch CUDA Available: {diagnostics['pytorch_cuda_available']}")
    print(f"PyTorch CUDA Version: {diagnostics['pytorch_cuda_version']}")
    print(f"GPU Count: {diagnostics['gpu_count']}")

    if diagnostics["errors"]:
        print("\nErrors encountered:")
        for error in diagnostics["errors"]:
            print(f"  - {error}")

    print("=====================\n")


def _detect_iface_to(master_addr: str) -> str | None:
    """Return the iface name that routes to master_addr, or None if not found."""
    if not master_addr:
        return None
    try:
        # Example output: "… dev ens5 src 172.31.33.153 uid 0"
        out = subprocess.check_output(
            [
                "bash",
                "-lc",
                f"ip route get {master_addr} | awk '{{for(i=1;i<=NF;i++) if($i==\"dev\"){{print $(i+1); exit}}}}'",
            ],
            text=True,
        ).strip()
        return out or None
    except Exception as e:
        logger.warning(f"Could not detect iface to {master_addr}: {e}")
        return None


def _iface_is_up(iface: str) -> bool:
    """Return True if the network interface exists and is UP."""
    try:
        out = subprocess.check_output(
            ["bash", "-lc", f"ip -o link show dev {iface}"],
            text=True,
        )
        return "state UP" in out
    except Exception:
        return False


def setup_nccl_debug_env(master_addr: str | None = os.environ.get("MASTER_ADDR")) -> None:
    """Set sane NCCL defaults for test runs, with optional verbose mode via METTA_NCCL_DEBUG=1."""
    debug_mode = os.environ.get("METTA_NCCL_DEBUG", "0") == "1"

    defaults = {
        "NCCL_DEBUG": "INFO" if debug_mode else "VERSION",
        "NCCL_DEBUG_SUBSYS": "ALL" if debug_mode else "",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_SHM_DISABLE": "1",  # keep isolation by default
        "NCCL_P2P_DISABLE": "1",  # keep isolation by default
        "NCCL_IB_DISABLE": "1",  # no IB/RDMA on these boxes
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_PORT_RANGE": os.environ.get("NCCL_PORT_RANGE", "43000-43063"),
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_MAX_NCHANNELS": "2",
    }
    if debug_mode:
        defaults["CUDA_LAUNCH_BLOCKING"] = "1"

    for k, v in defaults.items():
        if v:  # skip empty strings
            os.environ.setdefault(k, v)
        logger.info(f"{k}={os.environ[k]}")

    iface = _detect_iface_to(master_addr) if master_addr else None
    if iface and _iface_is_up(iface):
        os.environ["NCCL_SOCKET_IFNAME"] = iface
    else:
        # safer: explicitly prefer your real NIC
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "enp39s0")

    logger.info(f"NCCL_SOCKET_IFNAME={os.environ['NCCL_SOCKET_IFNAME']}")
    logger.info(f"MASTER_ADDR={master_addr or os.environ.get('MASTER_ADDR', '<unset>')}")


def test_nccl_communication() -> bool:
    """Test NCCL communication in distributed setting."""
    logger.info("Testing NCCL communication...")

    try:
        # Check if we're in a distributed environment
        if "RANK" not in os.environ:
            logger.warning("RANK not set, skipping distributed NCCL test")
            return True

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"Setting device to local_rank = {local_rank}")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Initialize process group
        logger.info("Initializing process group...")
        dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Rank {rank}/{world_size}: Process group initialized")
        logger.info(f"Rank {rank}: Using device {device}")

        # Test 1: All-reduce
        logger.info(f"Rank {rank}: Testing all-reduce...")
        tensor = torch.ones(1).to(device) * (rank + 1)
        dist.all_reduce(tensor)
        expected = world_size * (world_size + 1) // 2
        if abs(tensor.item() - expected) > 1e-6:
            raise ValueError(f"All-reduce failed: expected {expected}, got {tensor.item()}")
        logger.info(f"Rank {rank}: All-reduce test passed")

        # Test 2: Broadcast
        logger.info(f"Rank {rank}: Testing broadcast...")
        tensor = torch.zeros(1).to(device)
        if rank == 0:
            tensor.fill_(42)
        dist.broadcast(tensor, 0)
        if abs(tensor.item() - 42) > 1e-6:
            raise ValueError(f"Broadcast failed: expected 42, got {tensor.item()}")
        logger.info(f"Rank {rank}: Broadcast test passed")

        # Test 3: Barrier
        logger.info(f"Rank {rank}: Testing barrier...")
        dist.barrier()
        logger.info(f"Rank {rank}: Barrier test passed")

        logger.info(f"Rank {rank}: NCCL tests completed successfully")
        return True

    except Exception as e:
        logger.error(f"NCCL test failed: {e}", exc_info=True)
        return False
    finally:
        # Ensure we don’t leak communicators on failures too
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def test_single_gpu() -> bool:
    """Test single GPU functionality."""
    logger.info("Testing single GPU functionality...")

    try:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return False

        device = torch.device("cuda:0")

        # Test 1: Basic tensor operations
        logger.info("Testing basic tensor operations...")
        tensor = torch.ones(100, 100).to(device)
        result = torch.matmul(tensor, tensor)
        if result[0, 0].item() != 100:
            raise ValueError("Matrix multiplication failed")

        # Test 2: Memory allocation
        logger.info("Testing memory allocation...")
        large_tensor = torch.zeros(1000, 1000, 100).to(device)
        del large_tensor
        torch.cuda.empty_cache()

        logger.info("Single GPU tests passed")
        return True

    except Exception as e:
        logger.error(f"Single GPU test failed: {e}", exc_info=True)
        return False


def main():
    """Main function to run all diagnostics and tests."""
    # Collect diagnostics
    logger.info("Collecting GPU diagnostics...")
    diagnostics = get_gpu_diagnostics()

    # Print diagnostics
    print_diagnostics(diagnostics)

    # Setup debug environment
    setup_nccl_debug_env()

    # Run tests
    all_passed = True

    # Single GPU test
    if diagnostics["pytorch_cuda_available"]:
        if not test_single_gpu():
            all_passed = False
            logger.error("Single GPU test failed")
    else:
        logger.warning("Skipping GPU tests - CUDA not available")
        all_passed = False

    # NCCL communication test
    if "RANK" in os.environ:
        if not test_nccl_communication():
            all_passed = False
            logger.error("NCCL communication test failed")
    else:
        logger.info("Not in distributed environment, skipping NCCL communication test")

    # Summary
    if all_passed:
        logger.info("All tests passed!")
        print("\n✓ All tests passed!")
    else:
        logger.error("Some tests failed!")
        print("\n✗ Some tests failed!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
