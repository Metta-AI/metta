#!/usr/bin/env -S uv run
"""
Performance benchmarking script for Metta AI.
Measures training throughput, memory usage, and system utilization.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List
from contextlib import contextmanager
import statistics

import torch
import psutil
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Comprehensive performance benchmarking for training components."""
    
    def __init__(self):
        self.results = {}
        self.baseline_results = None
        
    @contextmanager
    def measure_time(self, operation_name: str):
        """Context manager to measure execution time."""
        start_time = time.perf_counter()
        start_memory = psutil.virtual_memory().used if psutil else 0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated()
        else:
            start_gpu_memory = 0
            
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated()
            else:
                end_gpu_memory = 0
                
            end_time = time.perf_counter()
            end_memory = psutil.virtual_memory().used if psutil else 0
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            self.results[operation_name] = {
                "execution_time": execution_time,
                "memory_delta_mb": memory_delta / (1024 * 1024),
                "gpu_memory_delta_mb": gpu_memory_delta / (1024 * 1024),
                "timestamp": time.time(),
            }
    
    def benchmark_tensor_operations(self, device: str = "cuda", batch_size: int = 16384) -> Dict[str, float]:
        """Benchmark basic tensor operations."""
        console.print(f"[cyan]Benchmarking tensor operations on {device}...[/cyan]")
        
        device_obj = torch.device(device)
        results = {}
        
        # Test different tensor sizes
        sizes = [
            (batch_size, 256),      # Small
            (batch_size, 1024),     # Medium
            (batch_size, 4096),     # Large
        ]
        
        for size_name, (rows, cols) in zip(["small", "medium", "large"], sizes):
            # Matrix multiplication
            with self.measure_time(f"matmul_{size_name}_{device}"):
                a = torch.randn(rows, cols, device=device_obj)
                b = torch.randn(cols, rows, device=device_obj)
                for _ in range(100):
                    c = torch.matmul(a, b)
                if device == "cuda":
                    torch.cuda.synchronize()
            
            # Element-wise operations
            with self.measure_time(f"elementwise_{size_name}_{device}"):
                x = torch.randn(rows, cols, device=device_obj)
                for _ in range(1000):
                    y = torch.relu(x) + torch.tanh(x)
                if device == "cuda":
                    torch.cuda.synchronize()
        
        return results
    
    def benchmark_mixed_precision(self, device: str = "cuda", batch_size: int = 8192) -> Dict[str, float]:
        """Benchmark mixed precision training performance."""
        if device != "cuda":
            console.print("[yellow]Mixed precision benchmarking only available on CUDA[/yellow]")
            return {}
            
        console.print("[cyan]Benchmarking mixed precision training...[/cyan]")
        
        device_obj = torch.device(device)
        
        # Simple model for testing
        model = torch.nn.Sequential(
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
        ).to(device_obj)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler()
        
        # Benchmark FP32
        model.train()
        with self.measure_time(f"training_fp32_{device}"):
            for _ in range(50):
                x = torch.randn(batch_size, 512, device=device_obj)
                y = torch.randn(batch_size, 256, device=device_obj)
                
                optimizer.zero_grad()
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, y)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
        
        # Benchmark FP16 (mixed precision)
        model.train()
        with self.measure_time(f"training_fp16_{device}"):
            for _ in range(50):
                x = torch.randn(batch_size, 512, device=device_obj)
                y = torch.randn(batch_size, 256, device=device_obj)
                
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device):
                    output = model(x)
                    loss = torch.nn.functional.mse_loss(output, y)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
        
        return {}
    
    def benchmark_torch_compile(self, device: str = "cuda", batch_size: int = 8192) -> Dict[str, float]:
        """Benchmark torch.compile performance."""
        console.print("[cyan]Benchmarking torch.compile...[/cyan]")
        
        device_obj = torch.device(device)
        
        # Test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(512, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 256),
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Regular model
        model_regular = TestModel().to(device_obj)
        
        # Compiled model
        try:
            model_compiled = torch.compile(TestModel().to(device_obj), mode="reduce-overhead")
            compile_available = True
        except Exception as e:
            console.print(f"[yellow]torch.compile not available: {e}[/yellow]")
            compile_available = False
        
        # Warm up and benchmark regular model
        x = torch.randn(batch_size, 512, device=device_obj)
        
        # Warmup
        for _ in range(10):
            _ = model_regular(x)
        
        with self.measure_time(f"inference_regular_{device}"):
            for _ in range(100):
                output = model_regular(x)
            if device == "cuda":
                torch.cuda.synchronize()
        
        # Benchmark compiled model if available
        if compile_available:
            # Warmup compiled model (compilation happens here)
            for _ in range(10):
                _ = model_compiled(x)
            
            with self.measure_time(f"inference_compiled_{device}"):
                for _ in range(100):
                    output = model_compiled(x)
                if device == "cuda":
                    torch.cuda.synchronize()
        
        return {}
    
    def benchmark_vectorization(self, num_envs: int = 16) -> Dict[str, float]:
        """Benchmark vectorization performance."""
        console.print("[cyan]Benchmarking vectorization...[/cyan]")
        
        # Simulate environment step operations
        def simulate_env_step(batch_size: int, num_steps: int = 1000):
            """Simulate environment stepping."""
            for _ in range(num_steps):
                # Simulate observation processing
                obs = np.random.rand(batch_size, 84, 84, 3).astype(np.float32)
                
                # Simulate reward calculation
                rewards = np.random.rand(batch_size).astype(np.float32)
                
                # Simulate action selection
                actions = np.random.randint(0, 8, size=batch_size)
                
                # Simulate some computation
                processed_obs = obs.mean(axis=(1, 2))
        
        # Benchmark serial processing
        with self.measure_time("vectorization_serial"):
            for _ in range(num_envs):
                simulate_env_step(1, 100)
        
        # Benchmark vectorized processing
        with self.measure_time("vectorization_batch"):
            simulate_env_step(num_envs, 100)
        
        return {}
    
    def benchmark_memory_operations(self, device: str = "cuda") -> Dict[str, float]:
        """Benchmark memory-intensive operations."""
        console.print(f"[cyan]Benchmarking memory operations on {device}...[/cyan]")
        
        device_obj = torch.device(device)
        
        # Large tensor allocation
        with self.measure_time(f"large_tensor_alloc_{device}"):
            tensors = []
            for _ in range(100):
                tensor = torch.randn(1024, 1024, device=device_obj)
                tensors.append(tensor)
            del tensors
            if device == "cuda":
                torch.cuda.empty_cache()
        
        # Memory copying
        if device == "cuda":
            with self.measure_time("memory_copy_cpu_to_gpu"):
                cpu_tensor = torch.randn(10000, 1000)
                for _ in range(50):
                    gpu_tensor = cpu_tensor.to(device_obj)
                torch.cuda.synchronize()
            
            with self.measure_time("memory_copy_gpu_to_cpu"):
                gpu_tensor = torch.randn(10000, 1000, device=device_obj)
                for _ in range(50):
                    cpu_tensor = gpu_tensor.cpu()
                torch.cuda.synchronize()
        
        return {}
    
    def run_full_benchmark(self, devices: List[str] = None, save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        if devices is None:
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices.append("mps")
        
        console.print(Panel.fit("🚀 Metta AI Performance Benchmark", style="bold blue"))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        ) as progress:
            
            total_tasks = len(devices) * 5  # 5 benchmark categories per device
            main_task = progress.add_task("Running benchmarks...", total=total_tasks)
            
            for device in devices:
                # Tensor operations
                progress.update(main_task, description=f"Tensor ops ({device})")
                self.benchmark_tensor_operations(device)
                progress.advance(main_task)
                
                # Mixed precision (CUDA only)
                progress.update(main_task, description=f"Mixed precision ({device})")
                if device == "cuda":
                    self.benchmark_mixed_precision(device)
                progress.advance(main_task)
                
                # Torch compile
                progress.update(main_task, description=f"Torch compile ({device})")
                self.benchmark_torch_compile(device)
                progress.advance(main_task)
                
                # Memory operations
                progress.update(main_task, description=f"Memory ops ({device})")
                self.benchmark_memory_operations(device)
                progress.advance(main_task)
                
                # Vectorization (CPU-centric, run once)
                if device == "cpu":
                    progress.update(main_task, description="Vectorization")
                    self.benchmark_vectorization()
                progress.advance(main_task)
        
        # Save results
        if save_results:
            timestamp = int(time.time())
            results_file = Path(f"benchmark_results_{timestamp}.json")
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            console.print(f"[green]Results saved to {results_file}[/green]")
        
        self.print_results()
        return self.results
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        if not self.results:
            console.print("[yellow]No benchmark results to display[/yellow]")
            return
        
        table = Table(title="🏁 Benchmark Results")
        table.add_column("Operation", style="cyan")
        table.add_column("Time (s)", style="green")
        table.add_column("Memory Δ (MB)", style="yellow")
        table.add_column("GPU Memory Δ (MB)", style="red")
        table.add_column("Throughput", style="magenta")
        
        for operation, metrics in self.results.items():
            time_s = f"{metrics['execution_time']:.4f}"
            memory_mb = f"{metrics['memory_delta_mb']:.1f}"
            gpu_memory_mb = f"{metrics['gpu_memory_delta_mb']:.1f}"
            
            # Calculate throughput where applicable
            if "matmul" in operation:
                ops_per_sec = 100 / metrics['execution_time']
                throughput = f"{ops_per_sec:.1f} ops/s"
            elif "training" in operation:
                steps_per_sec = 50 / metrics['execution_time']
                throughput = f"{steps_per_sec:.1f} steps/s"
            elif "inference" in operation:
                inferences_per_sec = 100 / metrics['execution_time']
                throughput = f"{inferences_per_sec:.1f} inf/s"
            else:
                throughput = "N/A"
            
            table.add_row(operation, time_s, memory_mb, gpu_memory_mb, throughput)
        
        console.print(table)
    
    def compare_with_baseline(self, baseline_file: Path):
        """Compare current results with baseline."""
        if not baseline_file.exists():
            console.print(f"[yellow]Baseline file {baseline_file} not found[/yellow]")
            return
        
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        console.print("\n[bold blue]📊 Performance Comparison vs Baseline[/bold blue]")
        
        comparison_table = Table()
        comparison_table.add_column("Operation", style="cyan")
        comparison_table.add_column("Current (s)", style="green")
        comparison_table.add_column("Baseline (s)", style="yellow")
        comparison_table.add_column("Improvement", style="magenta")
        
        for operation in self.results:
            if operation in baseline:
                current_time = self.results[operation]['execution_time']
                baseline_time = baseline[operation]['execution_time']
                improvement = (baseline_time - current_time) / baseline_time * 100
                
                improvement_str = f"{improvement:+.1f}%"
                if improvement > 0:
                    improvement_str = f"[green]{improvement_str}[/green]"
                elif improvement < -5:  # More than 5% slower
                    improvement_str = f"[red]{improvement_str}[/red]"
                
                comparison_table.add_row(
                    operation,
                    f"{current_time:.4f}",
                    f"{baseline_time:.4f}",
                    improvement_str
                )
        
        console.print(comparison_table)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Metta AI Performance Benchmark")
    parser.add_argument("--devices", nargs="+", help="Devices to benchmark (cpu, cuda, mps)")
    parser.add_argument("--baseline", type=Path, help="Baseline results file for comparison")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark(
        devices=args.devices,
        save_results=not args.no_save
    )
    
    if args.baseline:
        benchmark.compare_with_baseline(args.baseline)


if __name__ == "__main__":
    main()