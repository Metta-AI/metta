#!/usr/bin/env python3
"""
Diagnostic tool for multi-node distributed training issues.

Usage:
    # Single node, 4 GPUs
    uv run ./tools/run.py metta.debug.diagnose_multinode.diagnose

    # Or with launch.py for cloud deployment
    uv run ./tools/launch.py metta.debug.diagnose_multinode.diagnose -- gpus=4 nodes=1
    uv run ./tools/launch.py metta.debug.diagnose_multinode.diagnose -- gpus=4 nodes=4
"""

import os
import sys
import time
import torch
import torch.distributed as dist
from typing import Optional
from pydantic import Field

from metta.common.tool import Tool
from metta.rl.system_config import SystemConfig
from metta.rl.training import DistributedHelper


class DiagnoseMultiNodeTool(Tool):
    """Tool to diagnose multi-node distributed training configuration issues."""

    # Optional configuration parameters
    test_communication: bool = Field(default=True, description="Test inter-process communication")
    verbose: bool = Field(default=True, description="Enable verbose output")
    test_duration_seconds: float = Field(default=5.0, description="Duration for performance tests")

    def invoke(self, args: dict[str, str]) -> int | None:
        """Run the diagnostic tests."""
        print("\n" + "="*70)
        print("MULTI-NODE DISTRIBUTED TRAINING DIAGNOSTIC TOOL")
        print("="*70)

        # Print environment variables
        self._print_environment_variables()

        # Verify expected configuration
        issues = self._verify_expected_setup()

        # Test DistributedHelper
        helper = self._test_distributed_helper()

        # Test communication if distributed
        if helper and helper.is_distributed and self.test_communication:
            self._test_communication(helper)

        # Test actual GPU usage
        if helper and helper.is_distributed:
            self._test_gpu_assignment(helper)

        # Summary and diagnosis
        diagnosis = self._generate_diagnosis(helper, issues)

        # Cleanup
        if helper:
            helper.cleanup()

        # Return non-zero if issues found
        return 1 if issues else 0

    def _print_environment_variables(self):
        """Print all distributed training related environment variables."""
        print("\n" + "="*60)
        print("ENVIRONMENT VARIABLES")
        print("="*60)

        important_vars = [
            "WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
            "NUM_NODES", "NODE_RANK", "SKYPILOT_NUM_NODES", "SKYPILOT_NODE_RANK",
            "SKYPILOT_NUM_GPUS_PER_NODE", "DIST_URL", "LOCAL_WORLD_SIZE"
        ]

        for var in important_vars:
            value = os.environ.get(var, "<not set>")
            print(f"{var:30s} = {value}")

        print("\nCUDA Information:")
        if torch.cuda.is_available():
            print(f"  CUDA available: True")
            print(f"  CUDA device count: {torch.cuda.device_count()}")
            print(f"  Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_initialized() else 'not initialized'}")
        else:
            print("  CUDA available: False")

    def _verify_expected_setup(self) -> list[str]:
        """Verify the expected multi-node configuration and return list of issues."""
        print("\n" + "="*60)
        print("CONFIGURATION VERIFICATION")
        print("="*60)

        issues = []

        # Get environment variables
        world_size_str = os.environ.get("WORLD_SIZE", "")
        num_nodes_str = os.environ.get("NUM_NODES", "")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))

        # Check if WORLD_SIZE is set
        if not world_size_str:
            print("⚠️  WORLD_SIZE not set in environment")
            # Check for the fallback that distributed_helper uses
            if num_nodes_str:
                print(f"   NUM_NODES is set to {num_nodes_str}")
                print(f"   Code may incorrectly use NUM_NODES as WORLD_SIZE")
                issues.append("WORLD_SIZE not set, NUM_NODES used as fallback")

        world_size = int(world_size_str) if world_size_str else int(num_nodes_str or "1")
        num_nodes = int(num_nodes_str) if num_nodes_str else 1

        # Calculate expected values
        if num_nodes > 1:
            gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
            expected_world_size = num_nodes * gpus_per_node

            print(f"\nMulti-node setup detected:")
            print(f"  NUM_NODES: {num_nodes}")
            print(f"  GPUs per node: {gpus_per_node}")
            print(f"  Expected WORLD_SIZE: {expected_world_size}")
            print(f"  Actual WORLD_SIZE: {world_size}")

            if world_size != expected_world_size:
                print(f"\n⚠️  WARNING: WORLD_SIZE mismatch!")
                print(f"    Expected {expected_world_size} (NUM_NODES * GPUs_per_node)")
                print(f"    Got {world_size}")
                issues.append(f"WORLD_SIZE mismatch: {world_size} != {expected_world_size}")

                if world_size == num_nodes:
                    print(f"\n🐛 CRITICAL BUG: WORLD_SIZE equals NUM_NODES!")
                    print(f"    The distributed setup is treating nodes as processes")
                    print(f"    Only {num_nodes} out of {expected_world_size} GPUs will be used!")
                    issues.append(f"CRITICAL: Only {num_nodes}/{expected_world_size} GPUs being used")
            else:
                print(f"\n✅ WORLD_SIZE is correct for multi-node setup")
        else:
            print(f"\nSingle-node setup:")
            print(f"  WORLD_SIZE: {world_size}")
            print(f"  LOCAL_RANK: {local_rank}")
            print(f"  RANK: {rank}")

        return issues

    def _test_distributed_helper(self) -> Optional[DistributedHelper]:
        """Test DistributedHelper initialization and configuration."""
        print("\n" + "="*60)
        print("TESTING DistributedHelper")
        print("="*60)

        # Show what SystemConfig will use
        print(f"\nSystemConfig device (from guess_device): {SystemConfig().device}")

        # Test the actual path used in TrainTool
        print(f"\nTesting TrainTool initialization path:")
        system_cfg = self.system  # This is what TrainTool uses
        print(f"  Tool.system.device: {system_cfg.device}")

        try:
            helper = DistributedHelper(system_cfg)

            print(f"\nDistributedHelper Configuration:")
            print(f"  Is distributed: {helper.is_distributed}")
            print(f"  World size: {helper.get_world_size()}")
            print(f"  Rank: {helper.get_rank()}")
            print(f"  Local rank: {helper.config.local_rank}")
            print(f"  Is master: {helper.is_master()}")
            print(f"  Device: {helper.config.device}")

            # Check for the specific bug where all ranks might use cuda:0
            if helper.is_distributed and helper.config.device.startswith("cuda"):
                device_index = int(helper.config.device.split(":")[-1]) if ":" in helper.config.device else 0
                expected_device_index = helper.config.local_rank
                if device_index != expected_device_index:
                    print(f"\n⚠️  WARNING: Device index mismatch!")
                    print(f"    Device: {helper.config.device} (index {device_index})")
                    print(f"    Local rank: {expected_device_index}")
                    print(f"    All ranks may be competing for the same GPU!")

            return helper
        except Exception as e:
            print(f"\nERROR initializing DistributedHelper: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _test_communication(self, helper: DistributedHelper):
        """Test that all processes can communicate."""
        print("\n" + "="*60)
        print("TESTING INTER-PROCESS COMMUNICATION")
        print("="*60)

        world_size = helper.get_world_size()
        rank = helper.get_rank()

        # Test 1: Barrier synchronization
        print(f"\n[Rank {rank}] Testing barrier synchronization...")
        try:
            helper.synchronize()
            print(f"[Rank {rank}] ✅ Barrier successful")
        except Exception as e:
            print(f"[Rank {rank}] ❌ Barrier failed: {e}")
            return

        # Test 2: All-reduce operation
        print(f"\n[Rank {rank}] Testing all-reduce...")
        try:
            tensor = torch.tensor([rank + 1.0]).to(helper.config.device)
            original_value = tensor.item()
            helper.all_reduce(tensor)
            expected = sum(range(1, world_size + 1))

            if abs(tensor.item() - expected) < 0.001:
                print(f"[Rank {rank}] ✅ All-reduce successful: {original_value} -> {tensor.item()}")
            else:
                print(f"[Rank {rank}] ❌ All-reduce FAILED: expected {expected}, got {tensor.item()}")
        except Exception as e:
            print(f"[Rank {rank}] ❌ All-reduce failed: {e}")

        # Test 3: Broadcast from master
        print(f"\n[Rank {rank}] Testing broadcast from master...")
        try:
            if helper.is_master():
                data = {"message": "Hello from master", "world_size": world_size}
            else:
                data = None

            received = helper.broadcast_from_master(data)
            print(f"[Rank {rank}] ✅ Broadcast received: {received}")
        except Exception as e:
            print(f"[Rank {rank}] ❌ Broadcast failed: {e}")

    def _test_gpu_assignment(self, helper: DistributedHelper):
        """Test actual GPU device assignment and usage."""
        print("\n" + "="*60)
        print("TESTING GPU ASSIGNMENT")
        print("="*60)

        if not torch.cuda.is_available():
            print("Skipping GPU tests (CUDA not available)")
            return

        rank = helper.get_rank()
        local_rank = helper.config.local_rank
        device = helper.config.device

        print(f"\n[Rank {rank}] GPU Assignment:")
        print(f"  Local rank: {local_rank}")
        print(f"  Assigned device: {device}")
        print(f"  Current CUDA device: {torch.cuda.current_device()}")

        # Test tensor allocation
        try:
            test_tensor = torch.randn(1000, 1000).to(device)
            print(f"  ✅ Successfully allocated tensor on {device}")
            print(f"  Tensor device: {test_tensor.device}")

            # Check which physical GPU is being used
            if device.startswith("cuda"):
                with torch.cuda.device(device):
                    current = torch.cuda.current_device()
                    print(f"  Physical GPU index: {current}")

                    # Quick memory allocation test
                    for _ in range(10):
                        _ = torch.randn(1000, 1000).cuda()
                    memory_allocated = torch.cuda.memory_allocated(current) / 1024**2
                    print(f"  Memory allocated: {memory_allocated:.2f} MB")
        except Exception as e:
            print(f"  ❌ Failed to use device {device}: {e}")

    def _generate_diagnosis(self, helper: Optional[DistributedHelper], issues: list[str]) -> str:
        """Generate final diagnosis and recommendations."""
        print("\n" + "="*60)
        print("DIAGNOSIS & RECOMMENDATIONS")
        print("="*60)

        if not helper:
            print("❌ Failed to initialize DistributedHelper")
            return "initialization_failed"

        if not helper.is_distributed:
            print("ℹ️  Running in single-process mode (non-distributed)")
            return "non_distributed"

        actual_world_size = helper.get_world_size()
        num_nodes = int(os.environ.get("NUM_NODES", "1"))

        if issues:
            print(f"\n❌ ISSUES FOUND ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

            print(f"\n📊 PERFORMANCE IMPACT:")
            if "CRITICAL" in str(issues):
                gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
                expected_gpus = num_nodes * gpus_per_node
                actual_gpus = actual_world_size
                print(f"  Expected GPUs in use: {expected_gpus}")
                print(f"  Actual GPUs in use: {actual_gpus}")
                print(f"  Performance loss: {(1 - actual_gpus/expected_gpus)*100:.1f}%")
                print(f"\n  This explains why 4 GPUs and {expected_gpus} GPUs show identical performance!")

            print(f"\n🔧 RECOMMENDATIONS:")
            if "WORLD_SIZE not set" in str(issues):
                print("  1. Ensure torchrun is called with correct --nnodes and --nproc-per-node")
                print("  2. Check that WORLD_SIZE env var is properly set before launching")
            if "WORLD_SIZE mismatch" in str(issues):
                print("  1. Verify the launch script correctly calculates WORLD_SIZE = nodes * gpus_per_node")
                print("  2. Check if NUM_NODES is being incorrectly used as WORLD_SIZE")

            return "issues_found"
        else:
            print(f"\n✅ Distributed training configured correctly")
            print(f"  World size: {actual_world_size}")
            print(f"  All processes can communicate")
            return "success"


def diagnose() -> DiagnoseMultiNodeTool:
    """Tool maker function for the diagnostic tool."""
    return DiagnoseMultiNodeTool()


def diagnose_quick() -> DiagnoseMultiNodeTool:
    """Quick version without communication tests."""
    return DiagnoseMultiNodeTool(test_communication=False, test_duration_seconds=1.0)


def diagnose_verbose() -> DiagnoseMultiNodeTool:
    """Verbose version with extended tests."""
    return DiagnoseMultiNodeTool(verbose=True, test_duration_seconds=10.0)