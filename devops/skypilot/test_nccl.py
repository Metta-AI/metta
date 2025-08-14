#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NCCL Diagnostics and Testing Module

This module provides comprehensive GPU/CUDA/NCCL diagnostics and testing capabilities
for distributed PyTorch training environments.
"""

import datetime
import logging
import os
import subprocess
import sys
import time
from typing import Any, Optional

import torch
import torch.distributed as dist

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def measure_p2p_bandwidth(
    device: torch.device,
    src_rank: int = 0,
    dst_rank: int = 1,
    message_size_mb: int = 64,
    num_iterations: int = 10,
    num_warmup: int = 5,
) -> Optional[dict[str, float]]:
    """
    Measure point-to-point bandwidth between two specific ranks.
    Fixed version with proper synchronization.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Skip if only one rank
    if world_size < 2:
        return None

    # All ranks must participate in barriers
    if rank not in [src_rank, dst_rank]:
        # Non-participating ranks just wait at barriers
        dist.barrier()  # Initial sync
        dist.barrier()  # After warmup
        dist.barrier()  # After measurement
        return None

    # Create tensor
    message_size = (message_size_mb * 1024 * 1024) // 4  # Convert to float32 elements
    tensor = torch.randn(message_size, dtype=torch.float32, device=device)
    bytes_transferred = tensor.numel() * tensor.element_size()

    # Initial synchronization
    dist.barrier()

    # Warmup
    for _ in range(num_warmup):
        if rank == src_rank:
            dist.send(tensor, dst=dst_rank)
        elif rank == dst_rank:
            dist.recv(tensor, src=src_rank)

    # Synchronize after warmup
    torch.cuda.synchronize()
    dist.barrier()

    # Measure bandwidth
    start = time.perf_counter()

    for _ in range(num_iterations):
        if rank == src_rank:
            dist.send(tensor, dst=dst_rank)
        elif rank == dst_rank:
            dist.recv(tensor, src=src_rank)

    torch.cuda.synchronize()

    # Final synchronization
    dist.barrier()

    end = time.perf_counter()

    if rank == dst_rank:
        elapsed_seconds = (end - start) / num_iterations
        bandwidth_gbps = (bytes_transferred / elapsed_seconds) / 1e9

        return {
            "src_rank": src_rank,
            "dst_rank": dst_rank,
            "message_size_mb": message_size_mb,
            "bandwidth_gbps": bandwidth_gbps,
            "time_ms": elapsed_seconds * 1000,
        }

    return None


def test_nccl_benchmarks() -> bool:
    """Run NCCL bandwidth benchmarks with proper error handling."""
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Log start
        if rank == 0:
            logger.info("Starting NCCL benchmarks...")

        # Ensure all ranks are ready
        dist.barrier()

        # Collect results
        results = {}

        # 1. P2P Bandwidth Test (if we have at least 2 ranks)
        if world_size >= 2:
            if rank == 0:
                logger.info("Running P2P bandwidth test...")

            p2p_result = measure_p2p_bandwidth(device)

            # Gather results from rank 1 to rank 0
            if rank == 1 and p2p_result:
                dist.send(torch.tensor([p2p_result["bandwidth_gbps"]], dtype=torch.float32, device=device), dst=0)
            elif rank == 0:
                if world_size >= 2:
                    bandwidth_tensor = torch.zeros(1, device=device)
                    dist.recv(bandwidth_tensor, src=1)
                    results["p2p_bandwidth"] = {
                        "src_rank": 0,
                        "dst_rank": 1,
                        "message_size_mb": 64,
                        "bandwidth_gbps": float(bandwidth_tensor.item()),
                        "time_ms": 0,  # Will be calculated
                    }

        # Ensure all ranks complete P2P test
        dist.barrier()

        # 2. Allreduce Bandwidth Test
        if rank == 0:
            logger.info("Running allreduce bandwidth test...")

        allreduce_results = measure_allreduce_bandwidth(device)
        if rank == 0 and allreduce_results:
            results["allreduce_bandwidth"] = allreduce_results

        # Final synchronization
        dist.barrier()

        # Store results for later use
        if rank == 0:
            print_benchmark_results(results)

        return True

    except Exception as e:
        logger.error(f"Rank {rank}: Benchmark test failed: {e}", exc_info=True)
        # Try to sync remaining ranks
        try:
            dist.barrier()
        except Exception:
            pass
        return False


def measure_allreduce_bandwidth(
    device: torch.device,
    sizes_mb: list[int] | None = None,
    num_iterations: int = 10,
    num_warmup: int = 5,
) -> list[dict[str, float]]:
    """
    Measure allreduce bandwidth with better error handling.
    """
    if sizes_mb is None:
        sizes_mb = [1, 4, 16, 64]

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    results = []

    for size_mb in sizes_mb:
        size_elements = (size_mb * 1024 * 1024) // 4  # Convert MB to float32 elements

        try:
            # Check available memory first
            free_memory = torch.cuda.mem_get_info(device.index)[0]
            required_memory = size_elements * 4 * 2  # float32 * 2 for safety

            if free_memory < required_memory:
                if rank == 0:
                    logger.warning(f"Skipping size {size_mb}MB - insufficient memory")
                continue

            # Allocate tensor
            tensor = torch.randn(size_elements, dtype=torch.float32, device=device)
            bytes_per_element = tensor.element_size()
            total_bytes = size_elements * bytes_per_element

            # Ensure all ranks are ready
            dist.barrier()

            # Warmup
            for _ in range(num_warmup):
                dist.all_reduce(tensor)

            # Synchronize before timing
            torch.cuda.synchronize()
            dist.barrier()

            # Time the operation
            start = time.perf_counter()

            for _ in range(num_iterations):
                dist.all_reduce(tensor)

            torch.cuda.synchronize()

            # Ensure all ranks complete
            dist.barrier()

            end = time.perf_counter()

            # Calculate bandwidth
            elapsed_seconds = (end - start) / num_iterations

            # Allreduce algorithmic bandwidth: 2 * (n-1) / n * data_size
            algo_bytes = 2 * (world_size - 1) / world_size * total_bytes
            algo_bandwidth_gbps = (algo_bytes / elapsed_seconds) / 1e9

            result = {
                "size_mb": size_mb,
                "time_ms": elapsed_seconds * 1000,
                "bandwidth_gbps": algo_bandwidth_gbps,
            }

            if rank == 0:
                results.append(result)
                logger.info(f"Allreduce {size_mb}MB: {algo_bandwidth_gbps:.2f} GB/s")

            # Clean up tensor
            del tensor
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                logger.warning(f"Out of memory at size {size_mb}MB")
            # Ensure all ranks sync even on error
            try:
                dist.barrier()
            except Exception:
                pass
            break
        except Exception as e:
            if rank == 0:
                logger.error(f"Error at size {size_mb}MB: {e}")
            # Ensure all ranks sync even on error
            try:
                dist.barrier()
            except Exception:
                pass
            break

    return results if rank == 0 else []


def print_benchmark_results(results: dict[str, Any]) -> None:
    """Pretty print benchmark results."""
    print()
    print_box_header("NCCL BANDWIDTH BENCHMARKS", include_rank=False)

    # P2P bandwidth
    if "p2p_bandwidth" in results:
        p2p = results["p2p_bandwidth"]
        print(f"\n  📊 P2P BANDWIDTH (Rank {p2p['src_rank']} → Rank {p2p['dst_rank']}):")
        print(f"    Message Size : {p2p['message_size_mb']} MB")
        print(f"    Bandwidth    : {p2p['bandwidth_gbps']:.2f} GB/s")
        print(f"    Time         : {p2p['time_ms']:.2f} ms")

    # Allreduce bandwidth
    if "allreduce_bandwidth" in results:
        print("\n  📊 ALLREDUCE BANDWIDTH:")
        print(f"    {'Size (MB)':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}")
        print(f"    {'-' * 12} {'-' * 12} {'-' * 15}")

        for r in results["allreduce_bandwidth"]:
            print(f"    {r['size_mb']:<12} {r['time_ms']:<12.2f} {r['bandwidth_gbps']:<15.2f}")

        # Report peak
        best_result = max(results["allreduce_bandwidth"], key=lambda x: x["bandwidth_gbps"])
        print(f"\n  🚀 Peak Allreduce: {best_result['bandwidth_gbps']:.2f} GB/s at {best_result['size_mb']}MB")


def print_box_header(title: str, width: int = 75, include_rank: bool = True) -> None:
    """Print a formatted box header with centered title."""
    # Add rank info if requested and available
    if include_rank and "RANK" in os.environ:
        rank = int(os.environ.get("RANK", 0))
        _world_size = int(os.environ.get("WORLD_SIZE", 1))
        node_rank = int(os.environ.get("NODE_RANK", os.environ.get("NODE_INDEX", 0)))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        title = f"{title} (Rank {rank}, Node {node_rank}, GPU {local_rank})"

    # Ensure title fits with padding
    max_title_width = width - 4  # Account for borders and spacing
    if len(title) > max_title_width:
        title = title[: max_title_width - 3] + "..."

    # Calculate padding for centering
    padding = width - len(title)
    left_pad = padding // 2
    right_pad = padding - left_pad

    print(f"╔{'═' * width}╗")
    print(f"║{' ' * left_pad}{title}{' ' * right_pad}║")
    print(f"╚{'═' * width}╝")


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


def get_system_diagnostics() -> dict[str, Any]:
    """Collect comprehensive system diagnostics."""
    diagnostics = {}

    # Cluster configuration
    diagnostics["cluster"] = {
        "NUM_GPUS": os.environ.get("NUM_GPUS", "1"),
        "NUM_NODES": os.environ.get("NUM_NODES", "1"),
        "MASTER_ADDR": os.environ.get("MASTER_ADDR", "localhost"),
        "NODE_INDEX": os.environ.get("NODE_INDEX", "0"),
        "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
    }

    # System limits and mounts
    diagnostics["system"] = {}

    # ulimit -l
    code, stdout, stderr = run_command(["bash", "-c", "ulimit -l"])
    diagnostics["system"]["ULIMIT"] = stdout.strip() if code == 0 else "Error getting ulimit"

    # /dev/shm mount
    code, stdout, stderr = run_command(["mount"])
    if code == 0:
        shm_mount = [line for line in stdout.split("\n") if "/dev/shm" in line]
        diagnostics["system"]["SHM_MOUNT"] = shm_mount[0] if shm_mount else "No /dev/shm mount found"

    # Route to master
    master_addr = diagnostics["cluster"]["MASTER_ADDR"]
    code, stdout, stderr = run_command(["ip", "-o", "route", "get", master_addr])
    diagnostics["system"]["ROUTE_TO_MASTER"] = stdout.strip() if code == 0 else f"No route to {master_addr}"

    # Network interface
    iface = os.environ.get("NCCL_SOCKET_IFNAME", "enp39s0")
    code, stdout, stderr = run_command(["ip", "-o", "addr", "show", iface])
    diagnostics["system"]["NETWORK_INTERFACE"] = stdout.strip() if code == 0 else f"Interface {iface} not found"

    # IPC namespace
    try:
        ipc_ns = os.readlink("/proc/1/ns/ipc")
        diagnostics["system"]["IPC"] = ipc_ns
    except Exception:
        diagnostics["system"]["IPC"] = "Could not read IPC namespace"

    # SHM disk usage
    code, stdout, stderr = run_command(["df", "-h", "/dev/shm"])
    if code == 0:
        lines = stdout.strip().split("\n")
        diagnostics["system"]["SHM_DF"] = lines[-1] if len(lines) > 1 else "No SHM info"

    # Docker userns
    code, stdout, stderr = run_command(["docker", "info"])
    if code == 0:
        userns_lines = [line for line in stdout.split("\n") if "userns" in line.lower()]
        diagnostics["system"]["USERNS"] = userns_lines[0].strip() if userns_lines else "No userns info"

    # umask
    code, stdout, stderr = run_command(["bash", "-c", "umask"])
    diagnostics["system"]["UMASK"] = stdout.strip() if code == 0 else "Error getting umask"

    # SHM detailed info
    diagnostics["shm"] = {}

    code, stdout, stderr = run_command(["ls", "-ld", "/dev/shm"])
    diagnostics["shm"]["permissions"] = stdout.strip() if code == 0 else "Error"

    code, stdout, stderr = run_command(["ipcs", "-m"])
    if code == 0:
        lines = stdout.strip().split("\n")[:20]
        diagnostics["shm"]["ipcs"] = "\n".join(lines)

    # NCCL environment
    diagnostics["nccl_env"] = {
        k: v
        for k, v in os.environ.items()
        if any(pattern in k for pattern in ["NCCL_", "MASTER_", "RANK", "LOCAL_RANK", "WORLD_SIZE"])
    }

    return diagnostics


def print_system_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Print system diagnostics in a clean format."""
    # Cluster configuration
    print_box_header("CLUSTER CONFIGURATION")
    for k, v in diagnostics["cluster"].items():
        print(f"  {k:<15} : {v}")

    # System diagnostics
    print()
    print_box_header("SYSTEM DIAGNOSTICS")
    for k, v in diagnostics["system"].items():
        if k in ["ROUTE_TO_MASTER", "NETWORK_INTERFACE"]:
            # Truncate long lines
            v_str = str(v)
            if len(v_str) > 50:
                v_str = v_str[:47] + "..."
            print(f"  {k:<15} : {v_str}")
        else:
            print(f"  {k:<15} : {v}")

    # SHM info
    print()
    print_box_header("SHARED MEMORY (SHM) INFO")
    print(f"  Mount      : {diagnostics['system'].get('SHM_MOUNT', 'N/A')}")
    print(f"  Usage      : {diagnostics['system'].get('SHM_DF', 'N/A')}")
    print(f"  Permissions: {diagnostics['shm'].get('permissions', 'N/A')}")
    if diagnostics["shm"].get("ipcs"):
        print("  IPC Status :")
        for i, line in enumerate(diagnostics["shm"]["ipcs"].split("\n")[:5]):
            if i == 0:
                print(f"    {line}")  # Header
            else:
                print(f"    {line[:70]}...")  # Truncate long lines

    # NCCL environment
    print()
    print_box_header("NCCL ENVIRONMENT")
    nccl_vars = sorted([(k, v) for k, v in diagnostics["nccl_env"].items() if k.startswith("NCCL_")])
    other_vars = sorted([(k, v) for k, v in diagnostics["nccl_env"].items() if not k.startswith("NCCL_")])

    for k, v in nccl_vars:
        print(f"  {k:<25} : {v}")
    if other_vars:
        print("  ---")
        for k, v in other_vars:
            print(f"  {k:<25} : {v}")

    # NCCL Configuration Summary
    print()
    print_box_header("NCCL CONFIGURATION SUMMARY")

    # Extract values
    master_addr = diagnostics["cluster"]["MASTER_ADDR"]
    master_port = diagnostics["cluster"]["MASTER_PORT"]
    nccl_socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME", "enp39s0")
    nccl_socket_family = os.environ.get("NCCL_SOCKET_FAMILY", "AF_INET")
    nccl_port_range = os.environ.get("NCCL_PORT_RANGE", "43000-43063")
    nccl_debug = os.environ.get("NCCL_DEBUG", "VERSION")
    nccl_debug_subsys = os.environ.get("NCCL_DEBUG_SUBSYS", "")
    nccl_p2p_disable = int(os.environ.get("NCCL_P2P_DISABLE", "0"))
    nccl_shm_disable = int(os.environ.get("NCCL_SHM_DISABLE", "0"))
    nccl_ib_disable = int(os.environ.get("NCCL_IB_DISABLE", "1"))
    nccl_min_nchannels = os.environ.get("NCCL_MIN_NCHANNELS", "4")
    nccl_max_nchannels = os.environ.get("NCCL_MAX_NCHANNELS", "8")

    # Print in a clean table format
    print(f"  Rendezvous Endpoint    : {master_addr}:{master_port}")
    print(f"  Network Interface      : {nccl_socket_ifname}")
    print(f"  Socket Family          : {nccl_socket_family}")
    print(f"  Port Range             : {nccl_port_range}")

    debug_str = nccl_debug
    if nccl_debug_subsys:
        debug_str += f" (subsys: {nccl_debug_subsys})"
    print(f"  Debug Level            : {debug_str}")

    print(f"  Channels (min-max)     : {nccl_min_nchannels}-{nccl_max_nchannels}")
    print("\n  Communication Modes:")
    print(f"    • P2P (GPU Direct)   : {'✓ Enabled' if not nccl_p2p_disable else '✗ Disabled'}")
    print(f"    • Shared Memory      : {'✓ Enabled' if not nccl_shm_disable else '✗ Disabled'}")
    print(f"    • InfiniBand/EFA     : {'✓ Enabled' if not nccl_ib_disable else '✗ Disabled'}")


def print_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Pretty print GPU diagnostics information."""
    print()
    print_box_header("GPU DIAGNOSTICS")

    # Basic info in a clean table
    print(f"  PyTorch Version        : {diagnostics['torch_version']}")
    print(f"  PyTorch CUDA Available : {diagnostics['pytorch_cuda_available']}")
    print(f"  PyTorch CUDA Version   : {diagnostics['pytorch_cuda_version']}")
    print(f"  CUDA Version           : {diagnostics['cuda_version']}")
    print(f"  NCCL Version           : {diagnostics['nccl_version']}")
    print(f"  CUDA_VISIBLE_DEVICES   : {diagnostics['cuda_visible_devices']}")
    print(f"  GPU Count              : {diagnostics['gpu_count']}")

    if diagnostics["errors"]:
        print("\n  ⚠️  Errors encountered:")
        for error in diagnostics["errors"]:
            print(f"    • {error}")

    # nvidia-smi output (if available)
    if diagnostics["nvidia_smi"]:
        print("\n  NVIDIA-SMI Output:")
        print("  " + "-" * 70)
        for line in diagnostics["nvidia_smi"].strip().split("\n"):
            print(f"  {line}")
        print("  " + "-" * 70)


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


def setup_nccl_debug_env(master_addr: str | None = None) -> None:
    """Set sane NCCL defaults for test runs, with optional verbose mode via METTA_NCCL_DEBUG=1."""
    if not master_addr:
        master_addr = os.environ.get("MASTER_ADDR")

    debug_mode = os.environ.get("METTA_NCCL_DEBUG", "0") == "1"

    defaults = {
        "NCCL_DEBUG": "INFO" if debug_mode else "VERSION",
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
        defaults["NCCL_DEBUG_SUBSYS"] = "ALL"
        defaults["CUDA_LAUNCH_BLOCKING"] = "1"

    for k, v in defaults.items():
        os.environ.setdefault(k, v)
        logger.info(f"{k}={os.environ.get(k)}")

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
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=300))

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


def is_distributed_environment() -> bool:
    """Check if we're in a distributed environment."""
    # Check for torchrun environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return True

    # Check for custom environment variables that might indicate multi-node/GPU setup
    num_nodes = int(os.environ.get("NUM_NODES", "1"))
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))

    return num_nodes > 1 or num_gpus > 1


def launch_distributed_test() -> int:
    """Launch distributed test using torchrun."""
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    num_nodes = int(os.environ.get("NUM_NODES", "1"))
    node_index = int(os.environ.get("NODE_INDEX", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    logger.info(f"Launching distributed test with {num_gpus} GPUs on {num_nodes} nodes")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_index}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        __file__,
        "--distributed-worker",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Distributed test failed with return code {e.returncode}")
        return e.returncode


def extract_ip_from_interface(interface_info: str) -> str:
    """Extract IP address from interface info string."""
    try:
        if "inet" in interface_info:
            parts = interface_info.split()
            for i, part in enumerate(parts):
                if part == "inet" and i + 1 < len(parts):
                    return parts[i + 1].split("/")[0]
    except Exception:
        pass
    return "unknown"


def main():
    """Main function to run all diagnostics and tests."""
    # Check if we're a distributed worker
    if "--distributed-worker" in sys.argv:
        # We're inside a torchrun worker, run the actual tests
        logger.info("Running as distributed worker")
    else:
        # We're the main process, check if we need to launch distributed
        if is_distributed_environment() and "RANK" not in os.environ:
            logger.info("Detected distributed environment, launching with torchrun...")
            return launch_distributed_test()

    # Determine our position in the cluster
    # Standardize on NODE_INDEX (your launch script seems to use this)
    node_index = int(os.environ.get("NODE_INDEX", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    num_gpus_per_node = int(os.environ.get("NUM_GPUS", 1))

    # Define our roles
    IS_GPU0 = local_rank == 0
    IS_MASTER = (node_index == 0) and IS_GPU0  # Master is GPU 0 of node 0

    # Check if we're in a distributed environment
    is_distributed = (num_nodes > 1) or (num_gpus_per_node > 1) or (world_size > 1)

    if is_distributed and IS_GPU0:
        # Quick node status line
        print(f"[Node {node_index}] Initialized - {num_gpus_per_node} GPUs detected")

    # Print header (master only)
    if IS_MASTER:
        print("\n" + "═" * 75)
        print("                      NCCL DIAGNOSTICS AND TESTING")
        if is_distributed:
            print(f"                    Nodes: {num_nodes}, GPUs/node: {num_gpus_per_node}, Total: {world_size}")
        print("═" * 75)

    # Collect system diagnostics
    logger.info("Collecting system diagnostics...")
    system_diagnostics = get_system_diagnostics()

    # Cluster-wide configuration (master only)
    if IS_MASTER:
        print_box_header("CLUSTER CONFIGURATION", include_rank=False)
        cluster_info = system_diagnostics["cluster"]
        print(f"  NUM_GPUS        : {cluster_info['NUM_GPUS']}")
        print(f"  NUM_NODES       : {cluster_info['NUM_NODES']}")
        print(f"  MASTER_ADDR     : {cluster_info['MASTER_ADDR']}")
        print(f"  MASTER_PORT     : {cluster_info['MASTER_PORT']}")

    # Node-specific system diagnostics (GPU 0 of each node only)
    if IS_GPU0:
        # Get this node's IP
        node_ip = extract_ip_from_interface(system_diagnostics["system"].get("NETWORK_INTERFACE", ""))
        print()
        print_box_header(f"NODE {node_index} SYSTEM DIAGNOSTICS (IP: {node_ip})", include_rank=False)
        sys_info = system_diagnostics["system"]
        print(f"  ULIMIT          : {sys_info['ULIMIT']}")
        print(f"  SHM_MOUNT       : {sys_info['SHM_MOUNT']}")
        print(f"  NETWORK_INTERFACE : {str(sys_info.get('NETWORK_INTERFACE', 'N/A'))[:50]}...")
        print(f"  IPC             : {sys_info['IPC']}")
        print(f"  SHM_DF          : {sys_info['SHM_DF']}")
        print(f"  UMASK           : {sys_info['UMASK']}")

    # GPU diagnostics - one per node (GPU 0 only)
    if IS_GPU0:
        logger.info("Collecting GPU diagnostics...")
        gpu_diagnostics = get_gpu_diagnostics()
        print()
        print_box_header(f"NODE {node_index} GPU DIAGNOSTICS", include_rank=False)
        print(f"  PyTorch Version        : {gpu_diagnostics['torch_version']}")
        print(f"  PyTorch CUDA Available : {gpu_diagnostics['pytorch_cuda_available']}")
        print(f"  PyTorch CUDA Version   : {gpu_diagnostics['pytorch_cuda_version']}")
        print(f"  CUDA Version           : {gpu_diagnostics['cuda_version']}")
        print(f"  NCCL Version           : {gpu_diagnostics['nccl_version']}")
        print(f"  CUDA_VISIBLE_DEVICES   : {gpu_diagnostics['cuda_visible_devices']}")
        print(f"  GPU Count              : {gpu_diagnostics['gpu_count']}")

        # Only show nvidia-smi from master to avoid duplication
        if IS_MASTER and gpu_diagnostics["nvidia_smi"]:
            print("\n  NVIDIA-SMI Output:")
            print("  " + "-" * 70)
            for line in gpu_diagnostics["nvidia_smi"].strip().split("\n"):
                print(f"  {line}")
            print("  " + "-" * 70)
    else:
        # Non-GPU0 ranks still need GPU diagnostics for tests
        gpu_diagnostics = get_gpu_diagnostics()

    # NCCL environment - print from each rank as it can differ
    print()
    print_box_header("NCCL ENVIRONMENT", include_rank=True)
    nccl_env = system_diagnostics["nccl_env"]
    nccl_vars = sorted([(k, v) for k, v in nccl_env.items() if k.startswith("NCCL_")])

    # Only show key NCCL vars from non-GPU0 ranks to reduce noise
    if not IS_GPU0:
        key_vars = ["NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY"]
        nccl_vars = [(k, v) for k, v in nccl_vars if k in key_vars]

    for k, v in nccl_vars:
        print(f"  {k:<25} : {v}")

    # Setup debug environment
    setup_nccl_debug_env()

    # Run tests - each rank runs but only logs key info
    print()
    print_box_header("RUNNING TESTS", include_rank=True)

    all_passed = True
    test_results = []

    # Single GPU test
    if gpu_diagnostics["pytorch_cuda_available"]:
        if IS_GPU0:
            print(f"\n  🔧 Node {node_index}: Running single GPU test...")
        if test_single_gpu():
            test_results.append(("Single GPU Test", "✓ PASSED"))
        else:
            test_results.append(("Single GPU Test", "✗ FAILED"))
            all_passed = False
            if not IS_GPU0:  # Only print errors from non-GPU0 ranks
                print(f"  ✗ Rank {rank}: Single GPU test failed")
    else:
        logger.warning("Skipping GPU tests - CUDA not available")
        test_results.append(("Single GPU Test", "⚠ SKIPPED (No CUDA)"))
        all_passed = False

    # NCCL communication test
    if "RANK" in os.environ and world_size > 1:
        if IS_MASTER:
            print("\n  🔧 Running NCCL communication test across all ranks...")
        if test_nccl_communication():
            test_results.append(("NCCL Communication Test", "✓ PASSED"))
        else:
            test_results.append(("NCCL Communication Test", "✗ FAILED"))
            all_passed = False
            if not IS_GPU0:  # Only print errors from non-GPU0 ranks
                print(f"  ✗ Rank {rank}: NCCL test failed")
    else:
        logger.info("Not in distributed environment, skipping NCCL communication test")
        test_results.append(("NCCL Communication Test", "⚠ SKIPPED (Not distributed)"))

    # NCCL benchmarks
    if "RANK" in os.environ and world_size > 1:
        if IS_MASTER:
            print("\n  📊 Running bandwidth and latency benchmarks...")
        if test_nccl_benchmarks():
            test_results.append(("NCCL Benchmarks", "✓ PASSED"))
        else:
            test_results.append(("NCCL Benchmarks", "✗ FAILED"))
            all_passed = False

    all_ranks_passed = all_passed  # default for non-distributed

    # Synchronize results if distributed
    if is_distributed and "RANK" in os.environ:
        if dist.is_initialized():
            dist.barrier()
        # Check if we're still initialized (test might have destroyed it)
        if dist.is_initialized():
            # Gather results from all ranks
            all_passed_tensor = torch.tensor([1.0 if all_passed else 0.0], dtype=torch.float32).cuda()
            dist.all_reduce(all_passed_tensor)
            all_ranks_passed = all_passed_tensor.item() == float(world_size)
        else:
            # If process group was destroyed, we can't aggregate
            # In this case, we just report our local status
            logger.warning("Process group not initialized for result aggregation")

    # Summary - only from master
    if IS_MASTER:
        print()
        print_box_header("TEST SUMMARY", include_rank=False)

        for test_name, result in test_results:
            print(f"  {test_name:<30} : {result}")

        if is_distributed:
            print(f"\n  Overall: {'✓ All ranks passed' if all_ranks_passed else '✗ Some ranks failed'}")

        print("\n" + "═" * 75)
        if all_ranks_passed:
            print("                    ✓ ALL TESTS PASSED! ✓")
        else:
            print("                    ✗ SOME TESTS FAILED ✗")
        print("═" * 75 + "\n")

    return_code = 0 if all_passed else 1

    # Add after tests, before destroying process group
    if is_distributed and dist.is_initialized() and not all_ranks_passed:
        # Collect error details from failed ranks only
        if not all_passed:
            error_info = f"Rank {rank} (Node {node_index}, GPU {local_rank}): "
            error_info += ", ".join([f"{name} {result}" for name, result in test_results if "FAILED" in result])
        else:
            error_info = None

        if world_size <= 8:  # Only for small clusters
            error_list = [None] * world_size
            dist.all_gather_object(error_list, error_info)

            if IS_MASTER:
                errors = [err for err in error_list if err]  # Filter out None values
                if errors:
                    print("\n  Error Summary:")
                    for err in errors:
                        print(f"    {err}")

    # Clean up process group last
    if dist.is_initialized():
        dist.destroy_process_group()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
