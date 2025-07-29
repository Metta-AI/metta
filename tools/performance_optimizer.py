#!/usr/bin/env -S uv run
"""
Performance optimization utility for Metta AI training.
Analyzes system capabilities and recommends optimal settings.
"""

import logging
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import psutil
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()


class PerformanceOptimizer:
    """Analyzes system capabilities and generates optimized configurations."""
    
    def __init__(self):
        self.system_info = self._gather_system_info()
        
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information."""
        info = {
            "cpu": {
                "count": multiprocessing.cpu_count(),
                "physical_cores": psutil.cpu_count(logical=False),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "arch": platform.machine(),
                "platform": platform.system(),
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
            },
            "gpu": self._get_gpu_info(),
            "pytorch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            }
        }
        return info
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {"count": 0, "devices": []}
        
        if torch.cuda.is_available():
            gpu_info["count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "name": props.name,
                    "memory": props.total_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count,
                })
        
        return gpu_info
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance and generate recommendations."""
        recommendations = {}
        
        # CPU recommendations
        cpu_cores = self.system_info["cpu"]["physical_cores"] or self.system_info["cpu"]["count"]
        recommendations["cpu_threads"] = min(max(cpu_cores // 2, 1), 8)
        
        # Memory recommendations
        memory_gb = self.system_info["memory"]["total"] / (1024**3)
        if memory_gb < 8:
            recommendations["batch_size"] = 8192
            recommendations["minibatch_size"] = 1024
        elif memory_gb < 16:
            recommendations["batch_size"] = 32768
            recommendations["minibatch_size"] = 2048
        elif memory_gb < 32:
            recommendations["batch_size"] = 131072
            recommendations["minibatch_size"] = 8192
        else:
            recommendations["batch_size"] = 524288  # Default
            recommendations["minibatch_size"] = 16384  # Default
        
        # GPU recommendations
        if self.system_info["gpu"]["count"] > 0:
            recommendations["device"] = "cuda"
            recommendations["mixed_precision"] = True
            recommendations["compile"] = True
            
            # Adjust batch size based on GPU memory
            gpu_memory = self.system_info["gpu"]["devices"][0]["memory"] / (1024**3)
            if gpu_memory < 6:  # Less than 6GB
                recommendations["batch_size"] = min(recommendations["batch_size"], 32768)
                recommendations["minibatch_size"] = min(recommendations["minibatch_size"], 2048)
        elif self.system_info["pytorch"]["mps_available"]:
            recommendations["device"] = "mps"
            recommendations["mixed_precision"] = False  # MPS doesn't fully support it yet
            recommendations["compile"] = True
        else:
            recommendations["device"] = "cpu"
            recommendations["mixed_precision"] = False
            recommendations["compile"] = False
        
        # Vectorization recommendations
        if cpu_cores >= 8 and memory_gb >= 16:
            recommendations["vectorization"] = "multiprocessing"
            recommendations["num_workers"] = min(cpu_cores // 2, 8)
        else:
            recommendations["vectorization"] = "serial"
            recommendations["num_workers"] = 1
        
        return recommendations
    
    def generate_config(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate optimized configuration."""
        recommendations = self.analyze_performance()
        
        config = {
            "# Auto-generated optimized configuration": None,
            "device": recommendations["device"],
            "vectorization": recommendations["vectorization"],
            
            "env_vars": {
                "OMP_NUM_THREADS": str(recommendations["cpu_threads"]),
                "MKL_NUM_THREADS": str(recommendations["cpu_threads"]),
                "OPENBLAS_NUM_THREADS": str(recommendations["cpu_threads"]),
            },
            
            "trainer": {
                "batch_size": recommendations["batch_size"],
                "minibatch_size": recommendations["minibatch_size"],
                "num_workers": recommendations["num_workers"],
                "compile": recommendations["compile"],
                "use_mixed_precision": recommendations["mixed_precision"],
                "grad_scaler_enabled": recommendations["mixed_precision"],
                
                # Memory optimizations
                "pin_memory": True,
                "channels_last_memory_format": recommendations["device"] == "cuda",
                "use_memory_efficient_attention": True,
            }
        }
        
        # Add device-specific optimizations
        if recommendations["device"] == "cuda":
            config["env_vars"].update({
                "TORCH_CUDNN_V8_API_ENABLED": "1",
                "TORCH_CUDNN_BENCHMARK": "1",
                "CUDA_LAUNCH_BLOCKING": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,garbage_collection_threshold:0.8",
            })
        elif recommendations["device"] == "mps":
            config["env_vars"].update({
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.0",
            })
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                # Remove the comment key before saving
                clean_config = {k: v for k, v in config.items() if not k.startswith("#")}
                yaml.dump(clean_config, f, default_flow_style=False, sort_keys=False)
            console.print(f"[green]Generated optimized config: {output_path}[/green]")
        
        return config
    
    def print_analysis(self):
        """Print detailed system analysis."""
        console.print(Panel.fit("🚀 Metta AI Performance Analysis", style="bold blue"))
        
        # System info table
        table = Table(title="System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="white")
        
        # CPU info
        cpu = self.system_info["cpu"]
        table.add_row("CPU Cores", f"{cpu['count']} logical, {cpu['physical_cores']} physical")
        table.add_row("CPU Architecture", f"{cpu['arch']} ({cpu['platform']})")
        
        # Memory info
        memory = self.system_info["memory"]
        memory_gb = memory["total"] / (1024**3)
        table.add_row("Memory", f"{memory_gb:.1f} GB total, {memory['percent']:.1f}% used")
        
        # GPU info
        gpu = self.system_info["gpu"]
        if gpu["count"] > 0:
            for i, device in enumerate(gpu["devices"]):
                gpu_memory_gb = device["memory"] / (1024**3)
                table.add_row(f"GPU {i}", f"{device['name']} ({gpu_memory_gb:.1f} GB)")
        else:
            table.add_row("GPU", "None detected")
        
        # PyTorch info
        pt = self.system_info["pytorch"]
        table.add_row("PyTorch", f"v{pt['version']}")
        table.add_row("CUDA Support", "✅" if pt["cuda_available"] else "❌")
        table.add_row("MPS Support", "✅" if pt["mps_available"] else "❌")
        
        console.print(table)
        
        # Recommendations
        recommendations = self.analyze_performance()
        
        rec_table = Table(title="🎯 Performance Recommendations")
        rec_table.add_column("Setting", style="yellow")
        rec_table.add_column("Recommended Value", style="green")
        rec_table.add_column("Reason", style="white")
        
        rec_table.add_row("Device", recommendations["device"], "Best available compute device")
        rec_table.add_row("Batch Size", str(recommendations["batch_size"]), "Optimized for available memory")
        rec_table.add_row("CPU Threads", str(recommendations["cpu_threads"]), "Prevents CPU oversubscription")
        rec_table.add_row("Vectorization", recommendations["vectorization"], "Best for current hardware")
        rec_table.add_row("Mixed Precision", str(recommendations["mixed_precision"]), "Faster training with compatible hardware")
        rec_table.add_row("Torch Compile", str(recommendations["compile"]), "JIT compilation for speed")
        
        console.print(rec_table)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Metta AI Performance Optimizer")
    parser.add_argument("--output", "-o", type=Path, help="Output path for generated config")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    optimizer = PerformanceOptimizer()
    optimizer.print_analysis()
    
    if args.output:
        optimizer.generate_config(args.output)
    else:
        console.print("\n[yellow]Use --output to generate a config file[/yellow]")


if __name__ == "__main__":
    main()