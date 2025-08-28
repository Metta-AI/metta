#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import io
import os
import socket
import subprocess
import sys
import time
from typing import Any, Optional

import torch
import torch.distributed as dist


def measure_p2p_bandwidth(
    device: torch.device,
    src_rank: int = 0,
    dst_rank: int = 1,
    message_size_mb: int = 64,
    num_iterations: int = 10,
    num_warmup: int = 5,
) -> Optional[dict[str, float]]:
    """Measure point-to-point bandwidth between two specific ranks."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size < 2:
        return None

    # Only participating ranks do the work
    if rank in [src_rank, dst_rank]:
        # Create tensor
        message_size = (message_size_mb * 1024 * 1024) // 4
        tensor = torch.randn(message_size, dtype=torch.float32, device=device)
        bytes_transferred = tensor.numel() * tensor.element_size()

        # Warmup
        for _ in range(num_warmup):
            if rank == src_rank:
                dist.send(tensor, dst=dst_rank)
            elif rank == dst_rank:
                dist.recv(tensor, src=src_rank)

        torch.cuda.synchronize()

        # Measure bandwidth
        start = time.perf_counter()

        for _ in range(num_iterations):
            if rank == src_rank:
                dist.send(tensor, dst=dst_rank)
            elif rank == dst_rank:
                dist.recv(tensor, src=src_rank)

        torch.cuda.synchronize()
        end = time.perf_counter()

        if rank == dst_rank:
            elapsed_seconds = (end - start) / num_iterations
            bandwidth_gbps = (bytes_transferred / elapsed_seconds) / 1e9

            result = {
                "src_rank": src_rank,
                "dst_rank": dst_rank,
                "message_size_mb": message_size_mb,
                "bandwidth_gbps": bandwidth_gbps,
                "time_ms": elapsed_seconds * 1000,
            }
        else:
            result = None
    else:
        result = None

    # Single barrier at the end for all ranks
    dist.barrier()
    return result


def measure_allreduce_bandwidth(
    device: torch.device,
    sizes_mb: list[int] | None = None,
    num_iterations: int = 10,
    num_warmup: int = 5,
) -> list[dict[str, float]]:
    """Measure allreduce bandwidth with minimal synchronization."""
    if sizes_mb is None:
        sizes_mb = [1, 4, 16, 64]

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    results = []

    # Single barrier at start
    dist.barrier()

    for size_mb in sizes_mb:
        size_elements = (size_mb * 1024 * 1024) // 4

        try:
            # Check available memory
            free_memory = torch.cuda.mem_get_info(device.index)[0]
            required_memory = size_elements * 4 * 2

            if free_memory < required_memory:
                if rank == 0:
                    print(f"Skipping size {size_mb}MB - insufficient memory")
                continue

            # Allocate tensor
            tensor = torch.randn(size_elements, dtype=torch.float32, device=device)
            bytes_per_element = tensor.element_size()
            total_bytes = size_elements * bytes_per_element

            # Warmup (no barrier needed - allreduce is already collective)
            for _ in range(num_warmup):
                dist.all_reduce(tensor)

            torch.cuda.synchronize()

            # Time the operation
            start = time.perf_counter()

            for _ in range(num_iterations):
                dist.all_reduce(tensor)

            torch.cuda.synchronize()
            end = time.perf_counter()

            # Calculate bandwidth
            elapsed_seconds = (end - start) / num_iterations
            algo_bytes = 2 * (world_size - 1) / world_size * total_bytes
            algo_bandwidth_gbps = (algo_bytes / elapsed_seconds) / 1e9

            result = {
                "size_mb": size_mb,
                "time_ms": elapsed_seconds * 1000,
                "bandwidth_gbps": algo_bandwidth_gbps,
            }

            if rank == 0:
                results.append(result)

            # Clean up
            del tensor
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"Out of memory at size {size_mb}MB")
            break
        except Exception as e:
            if rank == 0:
                print(f"Error at size {size_mb}MB: {e}")
            break

    # Single barrier at end
    dist.barrier()
    return results if rank == 0 else []


def collect_nccl_benchmarks() -> dict[str, Any] | None:
    """Run NCCL bandwidth benchmarks with minimal synchronization."""
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            print("Process group not initialized, cannot run benchmarks")
            return None

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Initial sync
        dist.barrier()

        results = {}

        # P2P Bandwidth Test
        if world_size >= 2:
            p2p_result = measure_p2p_bandwidth(device)  # Has its own barrier

            # Gather results
            if rank == 1 and p2p_result:
                dist.send(torch.tensor([p2p_result["bandwidth_gbps"]], dtype=torch.float32, device=device), dst=0)
            elif rank == 0 and world_size >= 2:
                bandwidth_tensor = torch.zeros(1, device=device)
                dist.recv(bandwidth_tensor, src=1)
                results["p2p_bandwidth"] = {
                    "src_rank": 0,
                    "dst_rank": 1,
                    "message_size_mb": 64,
                    "bandwidth_gbps": float(bandwidth_tensor.item()),
                    "time_ms": 0,
                }

        allreduce_results = measure_allreduce_bandwidth(device)  # Has its own barriers
        if allreduce_results:
            results["allreduce_bandwidth"] = allreduce_results
        return results

    except Exception:
        return None


def format_benchmark_results(results: dict[str, Any]) -> str:
    """Format benchmark results as a string instead of printing directly."""
    output = io.StringIO()

    output.write("\n")
    output.write(format_box_header("NCCL BANDWIDTH BENCHMARKS", include_rank=False))

    # P2P bandwidth
    if "p2p_bandwidth" in results:
        p2p = results["p2p_bandwidth"]
        output.write(f"\n  📊 P2P BANDWIDTH (Rank {p2p['src_rank']} → Rank {p2p['dst_rank']}):\n")
        output.write(f"    Message Size : {p2p['message_size_mb']} MB\n")
        output.write(f"    Bandwidth    : {p2p['bandwidth_gbps']:.2f} GB/s\n")
        output.write(f"    Time         : {p2p['time_ms']:.2f} ms\n")

    # Allreduce bandwidth
    if "allreduce_bandwidth" in results:
        output.write("\n  📊 ALLREDUCE BANDWIDTH:\n")
        output.write(f"    {'Size (MB)':<12} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}\n")
        output.write(f"    {'-' * 12} {'-' * 12} {'-' * 15}\n")

        for r in results["allreduce_bandwidth"]:
            output.write(f"    {r['size_mb']:<12} {r['time_ms']:<12.2f} {r['bandwidth_gbps']:<15.2f}\n")

        # Report peak
        best_result = max(results["allreduce_bandwidth"], key=lambda x: x["bandwidth_gbps"])
        output.write(f"\n  🚀 Peak Allreduce: {best_result['bandwidth_gbps']:.2f} GB/s at {best_result['size_mb']}MB\n")

    return output.getvalue()


def print_benchmark_results(results: dict[str, Any], topology: dict[str, Any] | None = None) -> None:
    """Pretty print benchmark results with topology-aware interpretation."""
    output = format_benchmark_results(results)

    # Add topology-aware interpretation
    if topology and "p2p_bandwidth" in results:
        p2p = results["p2p_bandwidth"]
        src = f"GPU{p2p['src_rank']}"
        dst = f"GPU{p2p['dst_rank']}"

        # Check connection type
        conn_type = topology.get("matrix", {}).get(src, {}).get(dst, "SYS")

        expected_bandwidth = {
            "NV1": 25.0,  # NVLink 1.0
            "NV2": 50.0,  # NVLink 2.0
            "NV3": 300.0,  # NVLink 3.0
            "PIX": 16.0,  # PCIe switch
            "SYS": 8.0,  # System/CPU
        }.get(conn_type[:3], 8.0)

        efficiency = (p2p["bandwidth_gbps"] / expected_bandwidth) * 100

        output += "\n  📊 Topology Analysis:\n"
        output += f"    Connection Type: {conn_type}\n"
        output += f"    Expected BW: ~{expected_bandwidth:.0f} GB/s\n"
        output += f"    Efficiency: {efficiency:.0f}%\n"

    print(output)


def format_box_header(title: str, width: int = 75, include_rank: bool = True) -> str:
    """Format a box header as a string instead of printing directly."""
    output = io.StringIO()

    # Prepare the title with rank info if requested
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("NODE_INDEX", 0)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    display_title = f"{title} (Rank {rank}, Node {node_rank}, GPU {local_rank})"

    if not include_rank:
        display_title = f"{title}"

    # Ensure title fits with padding
    max_title_width = width - 4  # Account for borders and spacing
    if len(display_title) > max_title_width:
        display_title = display_title[: max_title_width - 3] + "..."

    # Calculate padding for centering
    padding = width - len(display_title)
    left_pad = padding // 2
    right_pad = padding - left_pad

    output.write(f"╔{'═' * width}╗\n")
    output.write(f"║{' ' * left_pad}{display_title}{' ' * right_pad}║\n")
    output.write(f"╚{'═' * width}╝\n")
    return output.getvalue()


def print_box_header(title: str, width: int = 75, include_rank: bool = True) -> None:
    """Print a formatted box header with centered title."""
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return  # Only print from rank 0 in distributed mode

    print(format_box_header(title, width, include_rank), end="")


def run_command(cmd: list[str], check: bool = False) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr
    except Exception as e:
        return -1, "", str(e)


def parse_gpu_topology(topo_output: str) -> dict[str, Any]:
    """Parse nvidia-smi topology matrix output."""
    lines = topo_output.strip().split("\n")
    topology = {
        "matrix": {},
        "connections": [],
        "numa_affinity": {},
        "cpu_affinity": {},
    }

    # Find the header line
    header_idx = -1
    for i, line in enumerate(lines):
        if "GPU0" in line or "GPU1" in line:
            header_idx = i
            break

    if header_idx == -1:
        return topology

    # Parse header to get GPU indices
    header_parts = lines[header_idx].split()
    gpu_indices = [p for p in header_parts if p.startswith("GPU")]

    # Parse the matrix
    for i, gpu_src in enumerate(gpu_indices):
        if header_idx + i + 1 >= len(lines):
            break

        line = lines[header_idx + i + 1]
        parts = line.split()
        if len(parts) < len(gpu_indices) + 1:
            continue

        topology["matrix"][gpu_src] = {}
        for j, gpu_dst in enumerate(gpu_indices):
            if j + 1 < len(parts):
                conn_type = parts[j + 1]
                topology["matrix"][gpu_src][gpu_dst] = conn_type

                # Extract meaningful connections
                if conn_type not in ["X", "SYS"] and "NV" in conn_type:
                    topology["connections"].append(
                        {
                            "src": gpu_src,
                            "dst": gpu_dst,
                            "type": "NVLink",
                            "links": int(conn_type[2]) if len(conn_type) > 2 else 1,
                        }
                    )

        # Parse CPU/NUMA affinity if present
        if "CPU Affinity" in lines[header_idx]:
            # Extract affinity info from the line
            affinity_parts = parts[len(gpu_indices) + 1 :]
            if affinity_parts:
                topology["cpu_affinity"][gpu_src] = affinity_parts[0]
            if len(affinity_parts) > 1:
                topology["numa_affinity"][gpu_src] = affinity_parts[1]

    return topology


def analyze_topology_performance(topology: dict[str, Any]) -> dict[str, Any]:
    """Analyze topology for performance implications."""
    analysis = {"nvlink_pairs": [], "isolated_gpus": [], "topology_type": "unknown", "recommendations": []}

    if not topology or "matrix" not in topology:
        return analysis

    matrix = topology["matrix"]
    gpu_list = list(matrix.keys())

    # Find NVLink pairs
    for gpu_src in gpu_list:
        for gpu_dst in gpu_list:
            if gpu_src < gpu_dst:  # Avoid duplicates
                conn = matrix.get(gpu_src, {}).get(gpu_dst, "")
                if "NV" in conn:
                    analysis["nvlink_pairs"].append((gpu_src, gpu_dst))

    # Find isolated GPUs (no NVLink)
    for gpu in gpu_list:
        has_nvlink = False
        for other_gpu in gpu_list:
            if gpu != other_gpu:
                conn = matrix.get(gpu, {}).get(other_gpu, "")
                if "NV" in conn:
                    has_nvlink = True
                    break
        if not has_nvlink:
            analysis["isolated_gpus"].append(gpu)

    # Determine topology type
    num_gpus = len(gpu_list)
    num_nvlink_pairs = len(analysis["nvlink_pairs"])

    if num_nvlink_pairs == 0:
        analysis["topology_type"] = "PCIe-only"
        analysis["recommendations"].append("Consider using gradient accumulation to reduce communication overhead")
    elif num_nvlink_pairs == num_gpus * (num_gpus - 1) // 2:
        analysis["topology_type"] = "Fully-connected (DGX-style)"
        analysis["recommendations"].append("Excellent topology for model parallelism and large batch training")
    elif num_nvlink_pairs == num_gpus // 2:
        analysis["topology_type"] = "Paired"
        analysis["recommendations"].append("Consider using hybrid parallelism with pairs as pipeline stages")
    else:
        analysis["topology_type"] = "Hybrid"

    # NUMA recommendations
    numa_nodes = set(topology.get("numa_affinity", {}).values())
    if len(numa_nodes) > 1:
        analysis["recommendations"].append("GPUs span multiple NUMA nodes - ensure proper CPU affinity")

    return analysis


def get_gpu_diagnostics(node_index: int) -> dict[str, Any]:
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
        "gpu_topology": None,
        "errors": [],
    }

    # Get nvidia-smi output
    print(f"Running nvidia-smi on node {node_index}...")
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

    if code == 0:  # If nvidia-smi worked
        print(f"Checking GPU topology on node {node_index}...")
        topo_code, topo_stdout, topo_stderr = run_command(["nvidia-smi", "topo", "-m"])
        if topo_code == 0:
            diagnostics["gpu_topology"] = parse_gpu_topology(topo_stdout)
        else:
            diagnostics["errors"].append(f"Failed to get GPU topology: {topo_stderr}")

    # Get CUDA version from nvcc
    print(f"Checking CUDA version on node {node_index}...")
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
    print(f"Checking NCCL version on node {node_index}...")
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

    # Get network interface info
    if code == 0 and stdout.strip():
        # Extract interface name from route output
        try:
            # Format: "8.8.8.8 via X.X.X.X dev eth0 src Y.Y.Y.Y"
            parts = stdout.strip().split()
            if "dev" in parts:
                dev_idx = parts.index("dev")
                if dev_idx + 1 < len(parts):
                    interface_name = parts[dev_idx + 1]
                    # Get interface details
                    code2, stdout2, stderr2 = run_command(["ip", "addr", "show", interface_name])
                    if code2 == 0:
                        diagnostics["system"]["NETWORK_INTERFACE"] = stdout2.strip()
        except Exception:
            pass

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


def format_system_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Format system diagnostics as a string instead of printing directly."""
    output = io.StringIO()

    # Cluster configuration
    output.write(format_box_header("CLUSTER CONFIGURATION"))
    for k, v in diagnostics["cluster"].items():
        output.write(f"  {k:<15} : {v}\n")

    # System diagnostics
    output.write("\n")
    output.write(format_box_header("SYSTEM DIAGNOSTICS"))
    for k, v in diagnostics["system"].items():
        if k in ["ROUTE_TO_MASTER", "NETWORK_INTERFACE"]:
            # Truncate long lines
            v_str = str(v)
            if len(v_str) > 50:
                v_str = v_str[:47] + "..."
            output.write(f"  {k:<15} : {v_str}\n")
        else:
            output.write(f"  {k:<15} : {v}\n")

    # SHM info
    output.write("\n")
    output.write(format_box_header("SHARED MEMORY (SHM) INFO"))
    output.write(f"  Mount      : {diagnostics['system'].get('SHM_MOUNT', 'N/A')}\n")
    output.write(f"  Usage      : {diagnostics['system'].get('SHM_DF', 'N/A')}\n")
    output.write(f"  Permissions: {diagnostics['shm'].get('permissions', 'N/A')}\n")
    if diagnostics["shm"].get("ipcs"):
        output.write("  IPC Status :\n")
        for i, line in enumerate(diagnostics["shm"]["ipcs"].split("\n")[:5]):
            if i == 0:
                output.write(f"    {line}\n")  # Header
            else:
                output.write(f"    {line[:70]}...\n")  # Truncate long lines

    # NCCL environment
    output.write("\n")
    output.write(format_box_header("NCCL ENVIRONMENT"))
    nccl_vars = sorted([(k, v) for k, v in diagnostics["nccl_env"].items() if k.startswith("NCCL_")])
    other_vars = sorted([(k, v) for k, v in diagnostics["nccl_env"].items() if not k.startswith("NCCL_")])

    for k, v in nccl_vars:
        output.write(f"  {k:<25} : {v}\n")
    if other_vars:
        output.write("  ---\n")
        for k, v in other_vars:
            output.write(f"  {k:<25} : {v}\n")

    # NCCL Configuration Summary
    output.write("\n")
    output.write(format_box_header("NCCL CONFIGURATION SUMMARY"))

    # Extract values
    master_addr = diagnostics["cluster"]["MASTER_ADDR"]
    master_port = diagnostics["cluster"]["MASTER_PORT"]
    nccl_socket_family = os.environ.get("NCCL_SOCKET_FAMILY", "AF_INET")
    nccl_port_range = os.environ.get("NCCL_PORT_RANGE", "43000-43063")
    nccl_debug = os.environ.get("NCCL_DEBUG", "VERSION")
    nccl_debug_subsys = os.environ.get("NCCL_DEBUG_SUBSYS", "")
    nccl_p2p_disable = int(os.environ.get("NCCL_P2P_DISABLE", "0"))
    nccl_shm_disable = int(os.environ.get("NCCL_SHM_DISABLE", "0"))
    nccl_ib_disable = int(os.environ.get("NCCL_IB_DISABLE", "1"))

    # Print in a clean table format
    output.write(f"  Rendezvous Endpoint    : {master_addr}:{master_port}\n")
    output.write(f"  Socket Family          : {nccl_socket_family}\n")
    output.write(f"  Port Range             : {nccl_port_range}\n")

    debug_str = nccl_debug
    if nccl_debug_subsys:
        debug_str += f" (subsys: {nccl_debug_subsys})"
    output.write(f"  Debug Level            : {debug_str}\n")

    output.write("\n  Communication Modes:\n")
    output.write(f"    • P2P (GPU Direct)   : {'✓ Enabled' if not nccl_p2p_disable else '✗ Disabled'}\n")
    output.write(f"    • Shared Memory      : {'✓ Enabled' if not nccl_shm_disable else '✗ Disabled'}\n")
    output.write(f"    • InfiniBand/EFA     : {'✓ Enabled' if not nccl_ib_disable else '✗ Disabled'}\n")

    return output.getvalue()


def print_system_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Print system diagnostics in a clean format - kept for backward compatibility."""
    print(format_system_diagnostics(diagnostics))


def format_gpu_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Format GPU diagnostics as a string instead of printing directly."""
    output = io.StringIO()

    output.write("\n")
    output.write(format_box_header("GPU DIAGNOSTICS"))

    # Basic info in a clean table
    output.write(f"  PyTorch Version        : {diagnostics['torch_version']}\n")
    output.write(f"  PyTorch CUDA Available : {diagnostics['pytorch_cuda_available']}\n")
    output.write(f"  PyTorch CUDA Version   : {diagnostics['pytorch_cuda_version']}\n")
    output.write(f"  CUDA Version           : {diagnostics['cuda_version']}\n")
    output.write(f"  NCCL Version           : {diagnostics['nccl_version']}\n")
    output.write(f"  CUDA_VISIBLE_DEVICES   : {diagnostics['cuda_visible_devices']}\n")
    output.write(f"  GPU Count              : {diagnostics['gpu_count']}\n")

    if diagnostics["errors"]:
        output.write("\n  ⚠️  Errors encountered:\n")
        for error in diagnostics["errors"]:
            output.write(f"    • {error}\n")

    # nvidia-smi output (if available)
    if diagnostics["nvidia_smi"]:
        output.write("\n  NVIDIA-SMI Output:\n")
        output.write("  " + "-" * 70 + "\n")
        for line in diagnostics["nvidia_smi"].strip().split("\n"):
            output.write(f"  {line}\n")
        output.write("  " + "-" * 70 + "\n")

    # Add topology section if available
    if diagnostics.get("gpu_topology") and diagnostics["gpu_topology"].get("matrix"):
        topology = diagnostics["gpu_topology"]
        output.write("\n  GPU Topology:\n")
        output.write("  " + "-" * 50 + "\n")

        # Show connection matrix
        matrix = topology["matrix"]
        gpu_list = sorted(matrix.keys())

        if gpu_list:
            # Header
            output.write("       ")
            for gpu in gpu_list:
                output.write(f"{gpu:>6}")
            output.write("\n")

            # Matrix rows
            for gpu_src in gpu_list:
                output.write(f"  {gpu_src:>5}")
                for gpu_dst in gpu_list:
                    conn = matrix.get(gpu_src, {}).get(gpu_dst, "?")
                    output.write(f"{conn:>6}")
                output.write("\n")

        # Show analysis
        analysis = analyze_topology_performance(topology)
        output.write(f"\n  Topology Type: {analysis['topology_type']}\n")

        if analysis["nvlink_pairs"]:
            output.write("  NVLink Connections:\n")
            for src, dst in analysis["nvlink_pairs"]:
                output.write(f"    • {src} ↔ {dst}\n")

        if analysis["isolated_gpus"]:
            output.write("  ⚠️  PCIe-only GPUs: " + ", ".join(analysis["isolated_gpus"]) + "\n")

        if analysis["recommendations"]:
            output.write("  Recommendations:\n")
            for rec in analysis["recommendations"]:
                output.write(f"    • {rec}\n")

    return output.getvalue()


def print_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Pretty print GPU diagnostics information - kept for backward compatibility."""
    print(format_gpu_diagnostics(diagnostics))


def setup_nccl_debug_env(master_addr: str | None = None) -> None:
    """Set minimal NCCL settings for test runs."""
    if not master_addr:
        master_addr = os.environ.get("MASTER_ADDR")

    os.environ["NCCL_DEBUG"] = "VERSION"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,IPC"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Log current NCCL settings
    print("Setting NCCL Environment for testing:")
    for key, value in sorted(os.environ.items()):
        if key.startswith("NCCL_"):
            print(f"  {key} = {value}")

    print(f"MASTER_ADDR={master_addr or os.environ.get('MASTER_ADDR', '<unset>')}")


def test_nccl_communication() -> bool:
    """Test NCCL communication in distributed setting."""

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((master_addr, master_port))
        sock.close()
        assert result == 0
    except Exception:
        return False

    try:
        # Check if we're in a distributed environment
        if "RANK" not in os.environ:
            print("RANK not set, skipping distributed NCCL test")
            return True

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        # Initialize process group
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=300))

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Test 1: All-reduce
        tensor = torch.ones(1).to(device) * (rank + 1)
        dist.all_reduce(tensor)
        expected = world_size * (world_size + 1) // 2
        if abs(tensor.item() - expected) > 1e-6:
            raise ValueError(f"All-reduce failed: expected {expected}, got {tensor.item()}")

        # Test 2: Broadcast
        tensor = torch.zeros(1).to(device)
        if rank == 0:
            tensor.fill_(42)
        dist.broadcast(tensor, 0)
        if abs(tensor.item() - 42) > 1e-6:
            raise ValueError(f"Broadcast failed: expected 42, got {tensor.item()}")

        # Test 3: Barrier
        dist.barrier()
        return True

    except Exception:
        return False


def test_single_gpu() -> bool:
    """Test single GPU functionality."""
    try:
        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda:0")

        # Test 1: Basic tensor operations
        tensor = torch.ones(100, 100).to(device)
        result = torch.matmul(tensor, tensor)
        if result[0, 0].item() != 100:
            raise ValueError("Matrix multiplication failed")

        # Test 2: Memory allocation
        large_tensor = torch.zeros(1000, 1000, 100).to(device)
        del large_tensor
        torch.cuda.empty_cache()

        return True

    except Exception:
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

    print(f"Launching distributed test with {num_gpus} GPUs on {num_nodes} nodes")

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

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Use Popen to have more control over the process
        process = subprocess.Popen(cmd)
        returncode = process.wait()

        if returncode != 0:
            print(f"[WARNING] torch.distributed.run exited with code {returncode}")
            # Give a grace period for cleanup
            time.sleep(2)

        return returncode
    except Exception as e:
        print(f"[ERROR] Failed to run distributed test: {e}")
        return 1


def extract_ip_from_interface(interface_info: str) -> str:
    """Extract IP address from interface info string."""
    try:
        # Look for inet (IPv4) addresses
        lines = interface_info.split("\n")
        for line in lines:
            if "inet " in line and "scope global" in line:
                # Format: "inet 172.31.4.248/20 brd ... scope global ..."
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if part == "inet" and i + 1 < len(parts):
                        ip_with_mask = parts[i + 1]
                        return ip_with_mask.split("/")[0]
    except Exception:
        pass

    # Fallback: try to extract from ROUTE_TO_MASTER if available
    try:
        # Check if this is a route output format
        if "src" in interface_info:
            parts = interface_info.split()
            for i, part in enumerate(parts):
                if part == "src" and i + 1 < len(parts):
                    return parts[i + 1]
    except Exception:
        pass

    return "unknown"


def main():
    """Main function to run all diagnostics and tests."""
    # Check if we're a distributed worker
    if "--distributed-worker" in sys.argv:
        # We're inside a torchrun worker, run the actual tests
        print("Running as distributed worker")
    else:
        # We're the main process, check if we need to launch distributed
        if is_distributed_environment():
            print("Detected distributed environment, launching with torchrun...")
            return launch_distributed_test()

    # Determine our position in the cluster
    # Standardize on NODE_INDEX (your launch script seems to use this)
    node_index = int(os.environ.get("NODE_INDEX", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    num_gpus_per_node = int(os.environ.get("NUM_GPUS", 1))
    world_size = num_nodes * num_gpus_per_node
    os.environ["WORLD_SIZE"] = str(world_size)

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

    # delay to de-synchronize printing
    time.sleep(0.02 * rank)

    # Collect system diagnostics
    print(f"Collecting system diagnostics on rank {rank}...")
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
        # Get this node's IP - try network interface first, then route as fallback
        node_ip = extract_ip_from_interface(system_diagnostics["system"].get("NETWORK_INTERFACE", ""))
        if node_ip == "unknown":
            # Try extracting from route
            node_ip = extract_ip_from_interface(system_diagnostics["system"].get("ROUTE_TO_MASTER", ""))

        print_box_header(f"NODE {node_index} SYSTEM DIAGNOSTICS (IP: {node_ip})", include_rank=False)
        sys_info = system_diagnostics["system"]
        print(f"  ULIMIT          : {sys_info['ULIMIT']}")
        print(f"  SHM_MOUNT       : {sys_info['SHM_MOUNT']}")
        print(f"  NETWORK_INTERFACE : {str(sys_info.get('NETWORK_INTERFACE', 'N/A'))[:50]}...")
        print(f"  IPC             : {sys_info['IPC']}")
        print(f"  SHM_DF          : {sys_info['SHM_DF']}")
        print(f"  UMASK           : {sys_info['UMASK']}")

    if IS_GPU0:
        print(f"Collecting GPU diagnostics on node {node_index}...")
        gpu_diagnostics = get_gpu_diagnostics(node_index)
        print_box_header(f"NODE {node_index} GPU DIAGNOSTICS", include_rank=False)
        print(f"  PyTorch Version        : {gpu_diagnostics['torch_version']}")
        print(f"  PyTorch CUDA Available : {gpu_diagnostics['pytorch_cuda_available']}")
        print(f"  PyTorch CUDA Version   : {gpu_diagnostics['pytorch_cuda_version']}")
        print(f"  CUDA Version           : {gpu_diagnostics['cuda_version']}")
        print(f"  NCCL Version           : {gpu_diagnostics['nccl_version']}")
        print(f"  CUDA_VISIBLE_DEVICES   : {gpu_diagnostics['cuda_visible_devices']}")
        print(f"  GPU Count              : {gpu_diagnostics['gpu_count']}")

        if gpu_diagnostics["nvidia_smi"]:
            print("\n  NVIDIA-SMI Output:")
            print("  " + "-" * 70)
            for line in gpu_diagnostics["nvidia_smi"].strip().split("\n"):
                print(f"  {line}")
            print("  " + "-" * 70)

    # NCCL environment - print from each rank as it can differ
    print_box_header("NCCL ENVIRONMENT", include_rank=True)
    nccl_env = system_diagnostics["nccl_env"]
    nccl_vars = sorted([(k, v) for k, v in nccl_env.items() if k.startswith("NCCL_")])
    for k, v in nccl_vars:
        print(f"  {k:<25} : {v}")

    # Setup debug environment
    setup_nccl_debug_env()

    # Run tests - each rank runs but only master logs key info
    print_box_header("RUNNING TESTS", include_rank=True)

    all_passed = True
    test_results = []

    # Single GPU test
    if torch.cuda.is_available():
        if IS_GPU0:
            print(f"\n  🔧 Node {node_index}: Running single GPU tests...")
        if test_single_gpu():
            test_results.append((f"Single GPU Test [Rank {rank}]", "✓ PASSED"))
        else:
            test_results.append((f"Single GPU Test [Rank {rank}]", "✗ FAILED"))
            all_passed = False
    else:
        test_results.append((f"Single GPU Test [Rank {rank}]", "⚠ FAILED (No CUDA)"))
        all_passed = False

    # NCCL communication test
    if all_passed and world_size > 1:
        if IS_GPU0:
            print(f"\n  🔧 Node {node_index}: Running NCCL communication tests...")

        if test_nccl_communication():
            test_results.append((f"NCCL Communication Test [Rank {rank}]", "✓ PASSED"))
        else:
            test_results.append((f"NCCL Communication Test [Rank {rank}]", "✗ FAILED"))
            all_passed = False
    else:
        test_results.append((f"NCCL Communication Test [Rank {rank}]", "⚠ SKIPPED (Not distributed)"))

    # NCCL benchmarks
    benchmark_results = None
    if all_passed and world_size > 1:
        if IS_GPU0:
            print(f"\n  📊 Node {node_index}: Running benchmarks...")

        bench_result = collect_nccl_benchmarks()

        # Only rank 0 gets results
        if bench_result is not None:
            benchmark_results = bench_result
            test_results.append((f"NCCL Benchmarks [Rank {rank}]", "✓ PASSED"))
        else:
            test_results.append((f"NCCL Benchmarks [Rank {rank}]", "✗ FAILED"))
            all_passed = False

    all_ranks_passed = all_passed

    # Synchronize results if distributed
    if all_ranks_passed and is_distributed:
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
            print("Process group not initialized for result aggregation")

    # Summary - only from master
    if IS_MASTER and benchmark_results:
        # Get topology from diagnostics if available
        topology = gpu_diagnostics.get("gpu_topology", {}) if "gpu_diagnostics" in locals() else {}
        print_benchmark_results(benchmark_results, topology)

    if IS_MASTER:
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

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
